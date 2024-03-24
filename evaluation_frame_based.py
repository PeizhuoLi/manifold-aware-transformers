import pickle
import torch
import numpy as np
from mesh_utils import SymbolicPoissonSolver, translation_alignment, create_sdf_info, PenetrationSolver, get_local_frames
from tqdm import tqdm
from utils import to_device, reshape_past, to_tensor
from models.matrix_analysis import clamp_singular_value
from my_profiler import Profiler


def compose_batched_data(dataset, network_input, results, sdf, cfg=None, body_pos=None, body_pos0=None,
                         body_g_velo=None, profiler=None):
    if profiler is not None: profiler.tick("repr from pos")
    all_input = dataset.get_repr_from_pos(network_input, results, sdf, cfg, body_pos, body_pos0,
                                          body_g_velo=body_g_velo, use_gpu=True)
    if profiler is not None: profiler.tock("repr from pos")
    if dataset.geodesic is not None:
        all_input['geodesic'] = dataset.geodesic
        all_input['geodesic'] = all_input['geodesic'].to(all_input['f'].device)
    if cfg is None:
        cfg = dataset.cfg
    if True:
        split_size = cfg.split_data if cfg.split_data else 1
        step_length = all_input['f'].shape[0] // split_size
        for k, v in all_input.items():
            if v.dim() == 3:
                # Per-face data
                all_input[k] = v.reshape((split_size, -1) + v.shape[1:])
            elif k == 'mask' or k == 'geodesic':
                res = [v[i * step_length:(i + 1) * step_length, i * step_length:(i + 1) * step_length] for i in range(split_size)]
                res = torch.stack(res, dim=0)
                all_input[k] = res
            elif v.dim() == 2:
                # Per-shape data
                v = v.unsqueeze(0)
                v = v.repeat(split_size, 1, 1)
                all_input[k] = v

    return all_input


def vanilla_evaluate(dataset, t_model, device, autoregressive=False, cond_length=1,
                     n_frames=40, noise_level=0.0, requires_jacobian=False,
                     global_alignment='g_velo', delayed_start=0,
                     keep_collision_result=True, write_all_inputs=False, requires_attn_func=None):
    if n_frames is not None and n_frames > dataset.n_frames:
        print('Warning: n_frames is larger than the dataset length. Setting n_frames to dataset length.')
        n_frames = dataset.n_frames

    cfg = dataset.cfg
    results = [torch.from_numpy(dataset.vert_pos_gt[i]) for i in range(cond_length + delayed_start)]
    jacobian_basis = []
    attns = {}
    results_collided = [results[i] for i in range(cond_length)]
    network_input = [torch.from_numpy(dataset.vert_pos[i]) for i in range(cond_length + delayed_start)]
    all_jacobians = []
    raw_res = [f for f in results]
    batchify_split = 1
    handle_indicator = None
    sdfs = None
    n_remain = 0
    cond_length = dataset.cfg.cond_length
    cfg = dataset.cfg
    use_persistent_solver = True

    all_inputs = []

    import os.path as osp
    if not write_all_inputs and osp.exists('./results/debug.pt'):
        with open('./results/debug.pt', 'rb') as f:
            all_inputs = pickle.load(f)

    expected_diff = []
    gt_diff = []

    frames = list(range(cond_length + delayed_start, n_frames))
    
    profiler = Profiler()

    if autoregressive:
        if dataset.cfg.use_sdf:
            sdfs = [dataset.sdf[:, i] for i in range(cond_length)]

    poisson_solver = SymbolicPoissonSolver(dataset.rest_pos, dataset.faces, dataset.get_poisson_list(),
                                           persistent=use_persistent_solver)

    penetration_solver = PenetrationSolver(dataset.rest_pos, dataset.faces, [4., 2., 1e-3], eps=4e-3)
    frame_failed = -1

    all_delta = []
    for f, frame in tqdm(enumerate(frames), total=len(frames)):
        profiler.tick('all')
        profiler.tick('collect')
        if autoregressive and frame > 1:
            sdfs_tensor = to_tensor(np.stack(sdfs, axis=1)) if dataset.cfg.use_sdf else None

            if cfg.add_base_deformation:
                body_pos = to_tensor(dataset.body_pos[f + cfg.cond_length])
                body_pos0 = to_tensor(dataset.body_pos[f + cfg.cond_length - 1])
            else:
                body_pos = None
                body_pos0 = None

            if cfg.add_base_velocity:
                body_g_velo = to_tensor(dataset.body_g_velo[f + 1: f + cfg.cond_length])
            else:
                body_g_velo = None

            all_input = compose_batched_data(dataset, torch.stack(network_input[-cond_length:], dim=0),
                                             torch.stack(results[-cond_length:], dim=0), sdfs_tensor,
                                             body_pos=body_pos, body_pos0=body_pos0, body_g_velo=body_g_velo, profiler=profiler)

            profiler.tock('collect')

        else:
            if dataset.cfg.split_data:
                all_input = [dataset[f * dataset.cfg.split_data + i][0] for i in range(dataset.cfg.split_data)]
                batched_input = {}
                for k in all_input[0].keys():
                    batched_input[k] = torch.stack([in_f[k] for in_f in all_input], dim=0)
                all_input = batched_input
            else:
                all_input = dataset[f][0]
                for k, v in all_input.items():
                    if isinstance(v, torch.Tensor):
                        all_input[k] = v.unsqueeze(0)
            profiler.tock('collect')

        profiler.tick('nn')
        if batchify_split:
            all_input_bak = {}
            if write_all_inputs:
                for k, v in all_input.items():
                    all_input_bak[k] = v.clone()
                all_inputs.append(all_input_bak)
            all_input = dataset.cfg.normalize_input(all_input)
            to_device(all_input, device)
            reshape_past(all_input)

            requires_attn = requires_attn_func is not None and requires_attn_func(f)
            if requires_attn:
                res, attn = t_model(all_input, requires_attn=True)
                attns[f] = attn
            else:
                res = t_model(all_input)
            for k in res.keys():
                res[k] = res[k].detach()
            res = dataset.cfg.denorm_output(res)
            res_dict = dataset.cfg.parse_output(res)
            singular_value = res_dict['singular_value'].reshape(-1, 3).cpu() if 'singular_value' in res_dict else None
            jacobians = res_dict['jacobians'].reshape((-1, 3, 3)).cpu()

        else:
            raise Exception('Abandoned')

        profiler.tock('nn')
        all_jacobians.append(jacobians)

        prev_vert = results[-1] if autoregressive else dataset.vert_pos_gt[frame - 1]
        jacobian_basis.append(prev_vert)

        if use_persistent_solver:
            prev_vert_stretch = get_local_frames(prev_vert, dataset.faces) @ dataset.inv_rest_local_frame
            jacobians = jacobians @ prev_vert_stretch

            profiler.tick('singular value')
            jacobians = clamp_singular_value(jacobians, None, None, singular_value, use_gpu=True)
            profiler.tock('singular value')
            prev_vert = dataset.rest_pos

        if dataset.cfg.use_heuristic_boundary:
            constraints = dataset.vert_pos_gt[len(results)][dataset.get_poisson_list()]
        else:
            constraints = None

        profiler.tick('Poisson solver')
        res = poisson_solver.forward(prev_vert, jacobians, constraints=constraints)
        profiler.tock('Poisson solver')
        failed = res is None

        if failed:
            n_remain = len(frames) - f
            results += [results[-1]] * n_remain
            frame_failed = frame
            print(f'Poisson solver failed at frame {frame}, skipping the rest of the frames')
            break

        res = torch.from_numpy(res).to(torch.float32)

        if dataset.cfg.use_heuristic_boundary:
            pass
        elif global_alignment == 'gt' or (dataset.global_velo is None):
            res = translation_alignment(res, dataset.vert_pos_gt[len(results)])
        elif 'velo' in global_alignment and dataset.global_velo is not None:
            g_velo = res_dict['g_velo'].reshape((-1, 3)).detach().cpu().numpy()
            if g_velo.ndim > 1:
                g_velo = g_velo.mean(axis=tuple(range(g_velo.ndim - 1)))
            g_velo_gt = dataset.global_velo[len(results) - 1]
            delta = g_velo_gt - g_velo
            all_delta.append(delta)
            res = translation_alignment(res, results[-1]) + g_velo
        else:
            raise Exception('Unknown global alignment method')

        raw_res.append(res.clone())
        res_collided = res
        if autoregressive:
            # Don't use the post-process if the result is not in the good global position
            # res_collided, c_mask = dataset.forward_control(res, len(results), eps=1e-3)  # post process for collision
            profiler.tick('collision')
            body_pos, F_body = dataset.load_dict['body_pos'][len(results)], dataset.load_dict['F_body']
            res_collided = penetration_solver.forward(res.numpy(), body_pos, F_body)
            if res_collided is None:
                n_remain = len(frames) - f
                results += [results[-1]] * n_remain
                print(f'Poisson solver failed at frame {f}, skipping the rest of the frames')
                break
            res_collided = torch.from_numpy(res_collided).to(torch.float32)
            profiler.tock('collision')

            if keep_collision_result:
                res = res_collided

        results.append(res.clone())
        results_collided.append(res_collided.clone())
        if len(results) < dataset.vert_pos.shape[0] and autoregressive:
            res, collision_face = dataset.forward_control(res, len(results))
            if handle_indicator is not None:
                handle_indicator = handle_indicator[1:] + [collision_face.unsqueeze(-1)]
            if cfg.use_sdf:
                profiler.tick('SDF')
                new_sdf0 = create_sdf_info(dataset.get_centroid(res).numpy(), dataset.load_dict['body_pos'][len(results) - 1], dataset.load_dict['F_body'])
                profiler.tock('SDF')
                new_sdf = [new_sdf0]
                new_sdf = np.stack(new_sdf, axis=0)
                sdfs = sdfs[1:] + [new_sdf]

        network_input.append(res)
        profiler.tock('all')

    if len(expected_diff) > 0:
        import matplotlib.pyplot as plt
        plt.plot(expected_diff)
        plt.legend(['Expected velo diff'])
        plt.show()
        plt.plot(gt_diff)
        plt.legend(['GT velo diff'])
        plt.show()

    if write_all_inputs:
        with open('./results/debug.pt', 'wb') as f:
            pickle.dump(all_inputs, f)

    eff = profiler.report()

    results = torch.stack(results, dim=0)
    raw_res = torch.stack(raw_res, dim=0)
    results_collided = torch.stack(results_collided, dim=0)
    jacobian_basis = to_tensor(np.stack(jacobian_basis, axis=0))
    all_jacobians = torch.stack(all_jacobians, dim=0).to('cpu')
    if requires_jacobian:
        return results, all_jacobians
    else:
        network_input = torch.stack(network_input, dim=0)
        return {'res': results, 'network_input': network_input, 'jacobians': all_jacobians,
                'compare_length': n_frames - n_remain, 'res_collided': results_collided,
                'jacobian_basis': jacobian_basis,
                'raw_res': raw_res, 'attns': attns, 'frame_failed': frame_failed, 'profiling': eff}
