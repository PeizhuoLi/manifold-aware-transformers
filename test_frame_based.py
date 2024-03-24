import igl
import torch
import numpy as np
import os.path as osp
from dataset.handle_dataset import MultipleDataset
from models.basics import load_dataset_from_args
from models.transformer import create_transformer_model_from_args
from mesh_utils import translation_alignment, mean_vertex_error
import pickle
from utils import count_parameters
from option import TrainOptionParser, TestOptionParser, override_args
from evaluation_frame_based import vanilla_evaluate
from torch.backends.cuda import sdp_kernel, SDPBackend
import os


def multiple_dataset(eval_func):
    def wrapper(dataset, path, *args, **kwargs):
        if isinstance(dataset, MultipleDataset):
            results = []
            for i, d in enumerate(dataset.datasets):
                print(f'Processing dataset {i}')
                results.append(eval_func(d, osp.join(path, f'{i:03d}'), *args, **kwargs))

            final_res = {}
            for key in results[0].keys():
                final_res[key] = np.mean([r[key] for r in results]) if key != 'profiling' else [r[key] for r in results]
            return final_res
        else:
            return eval_func(dataset, path, *args, **kwargs)

    return wrapper


@multiple_dataset
def metrics(dataset, save_path, t_model, device, n_frames=None, batch_size=-1):
    to_save = {}
    cond_length = dataset.cfg.cond_length
    cfg = dataset.cfg

    if n_frames is None:
        n_frames = dataset.vert_pos.shape[0] #- cond_length + cfg.intra_frame_jacobian

    with torch.no_grad():
        results_dict = vanilla_evaluate(dataset, t_model, device, n_frames=n_frames, cond_length=cond_length,
                                        batch_size=batch_size, global_alignment='velo')
        results = results_dict['res']
        results = results.to('cpu')
        results = translation_alignment(results, dataset.vert_pos_gt[:results.shape[0]])

    gt_vert = dataset.vert_pos_gt[:n_frames]


    obj_path = osp.join(save_path, 'debug')
    os.makedirs(obj_path, exist_ok=True)
    dataset.write_vert_pos_pickle(osp.join(obj_path, 'results.pkl'), results)

    extend_static = 10

    criterion = mean_vertex_error

    n_ignore = cond_length - cfg.intra_frame_jacobian
    to_save['all_loss'] = criterion(results[n_ignore:], gt_vert[n_ignore:]).item()
    to_save['loss_first'] = criterion(results[n_ignore], gt_vert[n_ignore]).item()
    to_save['loss_ten'] = criterion(results[extend_static], gt_vert[extend_static]).item()
    if results_dict['frame_failed'] == -1:
        to_save['difference'] = criterion(results[n_ignore:], results_dict['jacobian_basis'],
                                          translate_align=True).item()
    else:
        to_save['difference'] = -1

    dataset.write_vert_pos_pickle(osp.join(obj_path, 'input.pkl'), torch.cat([results[:n_ignore], results_dict['jacobian_basis']], dim=0))
    dataset.write_vert_pos_pickle(osp.join(obj_path, 'network_input.pkl'), dataset.vert_pos[:n_ignore + n_frames])

    obj_path = osp.join(save_path, 'obj_result')
    os.makedirs(obj_path, exist_ok=True)
    interested_frames = [extend_static - 1, extend_static, extend_static + 1]
    for i, i_frame in enumerate(interested_frames):
        if i_frame >= results.shape[0]:
            break
        igl.write_triangle_mesh(osp.join(obj_path, f'{i_frame:03d}_results.obj'), results[i_frame].cpu().numpy(),
                                dataset.faces)
        igl.write_triangle_mesh(osp.join(obj_path, f'{i_frame:03d}_gt.obj'), dataset.vert_pos_gt[i_frame],
                                dataset.faces)
        igl.write_triangle_mesh(osp.join(obj_path, f'{i_frame:03d}_input.obj'),
                                dataset.vert_pos[i_frame - 1], dataset.faces)
        igl.write_triangle_mesh(osp.join(obj_path, f'{i_frame:03d}_input_gt.obj'),
                                dataset.vert_pos_gt[i_frame - 1], dataset.faces)

    return to_save


@multiple_dataset
def sequence(dataset, save_path, t_model, device, autoregressive, n_frames, requires_attn_func=None):
    to_save = {}

    obj_path = osp.join(save_path, 'sequence')
    cond_length = dataset.cfg.cond_length

    if osp.exists(obj_path):
        os.system(f'rm -rf {obj_path}')
    os.makedirs(obj_path, exist_ok=True)

    dataset.write_vert_pos_pickle(osp.join(obj_path, 'gt.pkl'), dataset.vert_pos_gt)
    dataset.write_vert_pos_pickle(osp.join(obj_path, 'body.pkl'), dataset.load_dict['body_pos'],
                                  dataset.load_dict['F_body'])

    # with torch.no_grad():
    results_dict = vanilla_evaluate(dataset, t_model, device, autoregressive=autoregressive,
                                    cond_length=cond_length, n_frames=n_frames,
                                    global_alignment='g_velo', keep_collision_result=True,
                                    requires_attn_func=requires_attn_func)
    results = results_dict['res']
    results = results.to('cpu')
    compare_length = min(results.shape[0], dataset.vert_pos.shape[0])
    compare_length = min(compare_length, results_dict['compare_length'])
    to_save['compare_length'] = compare_length
    to_save['profiling'] = results_dict['profiling']
    to_save['n_faces'] = dataset.faces.shape[0]

    dataset.write_vert_pos_pickle(osp.join(obj_path, 'prediction.pkl'), results)

    attns_dict = results_dict['attns']
    attns_dict['faces'] = dataset.faces_complete
    attns_dict['split_data'] = dataset.cfg.split_data
    attns_dict['sep_point'] = dataset.sep_point if hasattr(dataset, 'sep_point') else None
    with open(osp.join(obj_path, 'attns.pkl'), 'wb') as f:
        pickle.dump(attns_dict, f)
    loss = mean_vertex_error(results[cond_length:compare_length], dataset.vert_pos_gt[cond_length:compare_length])
    to_save['vert_diff'] = loss.item()

    return to_save


def main():
    test_parser = TestOptionParser()
    test_args = test_parser.parse_args()
    train_parser = TrainOptionParser()
    args = train_parser.load(osp.join(test_args.save_path, 'args.txt'))
    args.save_path = test_args.save_path
    args.device = test_args.device

    device = torch.device(args.device)
    override_args(test_args, args)

    # args.split_data = 0

    training_dataset_z_up = False
    if 'cloth3d' in args.multiple_dataset or 'cloth3d' in args.dataset_path:
        if not args.set_cloth3d_y_up:
            training_dataset_z_up = True

    if test_args.another_dataset:
        args.dataset_path = test_args.another_dataset
        args.multiple_dataset = ''

        if 'cloth3d' not in args.dataset_path and training_dataset_z_up:
            args.convert_z_up = True
            print('Converting to Z-up')

    dataset_control_configs = ['slowdown_ratio', 'reverse', 'static_pose', 'start_frame',
                               'use_heuristic_boundary', 'scale_vert']

    args.use_mmap = 0
    args.fixed_downsample = 0
    for key in dataset_control_configs:
        if not hasattr(args, key):
            setattr(args, key, getattr(test_args, key))


    dataset = load_dataset_from_args(args)
    t_model = create_transformer_model_from_args(args, dataset.cfg).to(device)
    n_params = count_parameters(t_model)
    t_model.load_from_prefix(test_args.save_path, test_args.epoch)
    t_model.eval()

    if test_args.n_frames > dataset.n_frames:
        test_args.n_frames = dataset.n_frames

    if test_args.debug:
        dataset.vert_pos = dataset.vert_pos[:50]

    to_save = {}
    if test_args.mode == 'metrics':
        to_save = metrics(dataset, test_args.save_path, t_model, device, test_args.n_frames,
                          test_args.runtime_batch_size)
        to_save['n_params'] = n_params

    elif test_args.mode == 'sequence':
        os.makedirs(test_args.export_path, exist_ok=True)
        to_save = sequence(dataset, test_args.export_path, t_model, device, test_args.autoregressive,
                           test_args.n_frames, requires_attn_func=None)

    if test_args.export_path is None:
        test_args.export_path = test_args.save_path
    if test_args.print_final_result:
        print(to_save)
    if not test_args.not_save_to_file:
        with open(osp.join(test_args.export_path, 'summary_data.pickle'), 'wb') as handle:
            pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    backend_map = {
        SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
        SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False, "enable_flash": False, "enable_mem_efficient": True
        }
    }

    with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
        main()
