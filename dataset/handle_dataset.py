import torch
import numpy as np
import os
import os.path as osp
from torch.utils.data import Dataset
from mesh_utils import get_local_frames, body_cloth_collision, write_obj, \
    write_vert_pos_pickle, get_face_normal, cut_mesh_with_vertex_mask, create_face_laplace, \
    heuristic_boundary_cloth3d, coordinate_transform_cloth3d
from utils import get_std_mean_fast, get_unique_name, \
    to_tensor, numpy_wrapper, DynamicMmap, interpolate_t, auto_cut, signed_distance, smooth_t
from models.matrix_analysis import batched_svd, batched_svd_torch
import random
from tqdm import tqdm



class DataReprConfig:
    __keys = ['use_jacobian', 'cond_length', 'use_static_centroid',
              'normalization', 'extend_static', 'sample_rate', 'split_data',
              'remove_translation',
              'predict_g_velo', 'use_relative_stretch', 'mean_var_path',
              'use_sdf', 'normalize_sdf',
              'use_face_orientation', 'face_orientation_usage',
              'fixed_downsample', 'add_base_deformation',
              'normalize_base_deformation', 'predict_singular_value', 'add_base_velocity',
              'gaussian_filter_sigma', 'geodesic', 'geodesic_power', 'set_cloth3d_y_up']
    __optional_keys = {'slowdown_ratio': -1, 'reverse': 0, 'static_pose': 0, 'start_frame': 0,
                       'use_heuristic_boundary': 0, 'convert_z_up': 0, 'scale_vert': -1}
    __abandoned_keys = []

    def __init__(self, **kargs):
        for key in self.__keys + self.__abandoned_keys:
            setattr(self, key, kargs[key])
            kargs.pop(key)

        for key in self.__optional_keys:
            if key not in kargs:
                setattr(self, key, self.__optional_keys[key])
            else:
                setattr(self, key, kargs[key])
                kargs.pop(key)

        for key in kargs:
            # warnings.warn('Unused key: {}'.format(key))
            pass

        if osp.exists(self.mean_var_path):
            mean_var_dict = torch.load(self.mean_var_path)
            if not isinstance(mean_var_dict, dict):
                self.std_mean_input = None
                self.std_mean_output = None
                print('Existing mean and var out of date')
            else:
                self.std_mean_input = mean_var_dict['std_mean_input']
                self.std_mean_output = mean_var_dict['std_mean_output']
                print('Existing mean and var loaded')
        else:
            self.std_mean_input = None
            self.std_mean_output = None

    def normalize_pair(self, input, output):
        return self.normalize_input(input), self.normalize_output(output)

    def normalize_input(self, input):
        if self.normalization:
            if isinstance(input, dict):
                for key in self.std_mean_input:
                    device = input[key].device
                    input[key] = (input[key] - self.std_mean_input[key][1].to(device)) / self.std_mean_input[key][0].to(
                        device)
            else:
                input = (input - self.std_mean_input[1].to(input.device)) / self.std_mean_input[0].to(input.device)
        return input

    def normalize_output(self, output, strict=True):
        if self.normalization:
            if isinstance(output, dict):
                for key in self.std_mean_output:
                    device = output[key].device
                    std, mean = self.std_mean_output[key]
                    if not strict:
                        std = std[..., :output[key].shape[-1]]
                        mean = mean[..., :output[key].shape[-1]]
                    output[key] = (output[key] - mean.to(device)) / std.to(device)
            else:
                std, mean = self.std_mean_output[0].to(output.device), self.std_mean_output[1].to(output.device)
                if not strict:
                    std = std[..., :output.shape[-1]]
                    mean = mean[..., :output.shape[-1]]
                output = (output - mean) / std
        return output

    def denorm_output(self, output, keys=None):
        if self.normalization:
            if isinstance(output, dict):
                if keys is None:
                    keys = self.std_mean_output.keys()
                for key in keys:
                    device = output[key].device
                    output[key] = output[key] * self.std_mean_output[key][0].to(device) + self.std_mean_output[key][
                        1].to(device)
            else:
                output = output * self.std_mean_output[0].to(output.device) + self.std_mean_output[1].to(output.device)
        return output

    @property
    def n_channel(self):
        n_channels = 3
        if self.use_relative_stretch:
            n_channels += 9
        if self.use_sdf:
            n_channels += 4 * self.use_sdf
        if self.use_face_orientation:
            if self.face_orientation_usage == 'concat':
                n_channels += 3
            elif self.face_orientation_usage == 'cosine':
                n_channels += 1
        if self.add_base_deformation:
            n_channels += 12
        if self.predict_singular_value:
            n_channels += 3
        return n_channels

    @property
    def n_channel_total(self):
        res = self.n_channel * self.cond_length
        return res

    @property
    def n_channel_output(self):
        res = 9
        if self.predict_singular_value:
            res += 3
        return res

    @property
    def n_channel_g_velo(self):
        if not self.predict_g_velo:
            return 0
        else:
            return 3 * (self.cond_length - 1) * (1 + self.add_base_velocity)

    def parse_output(self, output_dict, key=None):
        assert key in ['jacobians', 'face_velo', 'g_velo', 'sdf', 'singular_value', None]
        res = {}
        if isinstance(output_dict, dict):
            output = output_dict['f']
            if 'g_velo' in output_dict:
                res['g_velo'] = output_dict['g_velo']
        else:
            output = output_dict
        res['jacobians'] = output[..., :9]
        output = output[..., 9:]
        if self.predict_singular_value:
            res['singular_value'] = output[..., :3]
            output = output[..., 3:]
        if key is not None:
            return res[key]
        return res

    def parse_input(self, input_dict, key=None):
        assert key in ['relative_stretch', 'centroids', 'sdf', 'base_deformation', 'face_velo', None]
        res = {}
        if isinstance(input_dict, dict):
            input = input_dict['f']
        else:
            input = input_dict

        input_shape = input.shape
        if input.shape[0] > self.split_data:  # The first dimension is not the batched dimension
            input = input.reshape(input_shape[:1] + (self.cond_length, -1))
        else:
            input = input.reshape(input_shape[:2] + (self.cond_length, -1))

        res['relative_stretch'] = input[..., :9]
        input = input[..., 9:]

        if self.use_face_orientation:
            n_orientation_channels = 3 if self.face_orientation_usage == 'concat' else 1
            res['face_orientation'] = input[..., :n_orientation_channels]
            input = input[..., n_orientation_channels:]

        res['centroids'] = input[..., :3]
        input = input[..., 3:]

        if self.use_sdf:
            n_sdf_channels = 4 * self.use_sdf
            sdf_part = input[..., :n_sdf_channels]
            sdf_part = sdf_part.reshape(input_shape[:2] + (self.cond_length, 4, self.use_sdf))
            res['sdf'] = sdf_part
            input = input[..., n_sdf_channels:]

        if self.add_base_deformation:
            base_deform = input[..., :12]
            input = input[..., 12:]
            for i in range(1, base_deform.shape[-2]):
                assert torch.allclose(base_deform[..., i, :], base_deform[..., 0, :])
            res['base_deformation'] = base_deform[..., 0, :]

        assert input.shape[-1] == 0

        if key is not None:
            return res[key]
        return input


class BasePhysicsDataset:
    """
    Principle
        vert_pos_i ----predict------> vert_pos_gt_i+1
        vert_pos_gt_i ---add control from frame i+1 --> vert_pos_i
        Important!!: vert_pos_i contains control from frame i+1.
    """

    def __init__(self, prefix, cfg, debug=False, device='cpu', **kargs):
        vert_pos, vert_pos_gt, rest_pos, faces, masks, load_dict = load_base_data(prefix, cfg)

        self.cfg = cfg

        # Store data for training
        self.vert_pos = vert_pos.astype(np.float32)
        self.vert_pos_gt = vert_pos_gt.astype(np.float32)
        self.rest_pos = rest_pos.astype(np.float32)
        self.faces = faces
        self.masks_vert = masks
        self.load_dict = load_dict
        self.output_keys = ['f']
        self.base_dataset_name = prefix.split('/')[-1]
        self.dataset_y_up = 'cloth3d' not in prefix or cfg.set_cloth3d_y_up

        if cfg.convert_z_up:
            self.dataset_y_up = False

        if 'dg_base' in self.load_dict:
            self.load_dict.pop('dg_base')
        if cfg.use_sdf:
            self.sdf = load_dict['sdf']
        else:
            self.sdf = None
        if 'sdf' in load_dict:
            load_dict.pop('sdf')
        if 'vt' in load_dict:
            self.vt = load_dict['vt']
            self.ftc = load_dict['ftc']
        else:
            self.vt = None
            self.ftc = None

        if cfg.add_base_deformation:
            self.Is = load_dict['Is']
            load_dict.pop('Is')
        else:
            self.Is = None

        if cfg.geodesic:
            print('geodesic power is', cfg.geodesic_power)
            geodesic = to_tensor(load_dict['distances'])
            load_dict.pop('distances')
            geodesic = torch.max(geodesic, dim=1, keepdim=True)[0] - geodesic
            geodesic = geodesic ** cfg.geodesic_power
            geodesic = torch.softmax(geodesic, dim=1)
            self.geodesic = geodesic
        else:
            self.geodesic = None

        # Store configurations
        self.debug = debug
        self.device = device

        self.n_frames = self.vert_pos.shape[0]

        if debug:
            self.vert_pos = self.vert_pos[:debug]

        if cfg.split_data:
            if self.faces.shape[0] % cfg.split_data:
                print(f'Warning: faces.shape[0] % self.split_data != 0, discarding {self.faces.shape[0] % cfg.split_data} faces')
            self.faces_complete = self.faces.copy()
            self.faces = self.faces[:self.faces.shape[0] // cfg.split_data * cfg.split_data]
            random.seed(0)
            perm = list(range(self.faces.shape[0]))
            random.shuffle(perm)
            perm_ori = perm.copy() + list(range(self.faces.shape[0], self.faces_complete.shape[0]))
            self.faces_complete = self.faces_complete[perm_ori]
            # todo: the sdf feature is stored in the default face permutation and the permutation should be also applied to it
            self.faces = np.ascontiguousarray(self.faces[perm])
            if self.sdf is not None:
                self.sdf = self.sdf[:, :, :self.faces.shape[0]]
                self.sdf = np.ascontiguousarray(self.sdf[:, :, perm])
            if self.ftc is not None:
                self.ftc = self.ftc[perm]
            if self.Is is not None:
                self.Is = np.ascontiguousarray(self.Is[:, perm])
            if self.geodesic is not None:
                new_geodesic = self.geodesic[perm][:, perm]
                self.geodesic = new_geodesic

            self.perm = perm
            sep_point = [i * self.faces.shape[0] // cfg.split_data for i in range(cfg.split_data)]
            sep_point.append(self.faces.shape[0])
            self.sep_point = sep_point
        else:
            self.faces_complete = self.faces

        if cfg.add_base_deformation:
            self.body_pos = self.load_dict['body_pos'].astype(np.float32)
            self.body_faces = self.load_dict['F_body']
        if cfg.add_base_velocity:
            if self.load_dict['translation'] is None:
                global_pos = np.zeros((self.load_dict['body_pos'].shape[0], 3), dtype=np.float32)
            else:
                global_pos = self.load_dict['translation'].astype(np.float32)
            self.body_g_velo = global_pos[1:] - global_pos[:-1]
            body_mean_pos = self.body_pos.mean(axis=1)
            body_g_velo2 = body_mean_pos[1:] - body_mean_pos[:-1]
            delta = self.body_g_velo - body_g_velo2
        else:
            self.body_g_velo = None

        if cfg.remove_translation:
            self.rest_pos = self.rest_pos - self.rest_pos.mean(axis=-2, keepdims=True)

        if cfg.remove_translation:
            vert_pos_gt_mean = self.vert_pos_gt.mean(axis=-2, keepdims=True)
            vert_pos_gt = self.vert_pos_gt - vert_pos_gt_mean
            vert_pos = self.vert_pos - self.vert_pos.mean(axis=-2, keepdims=True)
        else:
            vert_pos = self.vert_pos
            vert_pos_gt = self.vert_pos_gt

        if cfg.predict_g_velo:
            vert_mean_pos_gt = self.vert_pos_gt.mean(axis=-2)
            self.global_velo = vert_mean_pos_gt[1:] - vert_mean_pos_gt[:-1]
            self.output_keys.append('g_velo')
        else:
            self.global_velo = None

        if cfg.use_static_centroid:
            self.centroids = self.get_centroid(self.rest_pos)[None]
            self.centroids = np.broadcast_to(self.centroids, (vert_pos.shape[0], *self.centroids.shape[1:]))
            self.centroids_gt = self.centroids
        else:
            self.centroids = self.get_centroid(vert_pos)
            self.centroids_gt = self.get_centroid(vert_pos_gt)

        if cfg.use_face_orientation:
            self.vert_pos_orientation = get_face_normal(self.vert_pos, self.faces)
            self.vert_pos_gt_orientation = get_face_normal(self.vert_pos_gt, self.faces)

            if cfg.face_orientation_usage == 'cosine':
                gravity = np.array([0, -1, 0], dtype=np.float32)[None, None]
                self.vert_pos_orientation = np.sum(self.vert_pos_orientation * gravity, axis=-1, keepdims=True)
                self.vert_pos_gt_orientation = np.sum(self.vert_pos_gt_orientation * gravity, axis=-1, keepdims=True)
        else:
            self.vert_pos_orientation = None
            self.vert_pos_gt_orientation = None

    def get_face_orient_repr(self, vert_pos, faces=None, cfg=None):
        if faces is None:
            faces = self.faces
        if cfg is None:
            cfg = self.cfg

        orientation = get_face_normal(vert_pos, faces)
        if cfg.face_orientation_usage == 'cosine':
            gravity = np.array([0, -1, 0], dtype=np.float32)[None, None]
            orientation = (orientation * gravity).sum(axis=-1, keepdims=True)

        return orientation

    def get_centroid(self, vert_pos):
        centroids = vert_pos[..., self.faces, :].mean(axis=-2)
        return centroids

    def write_obj_vert_pos(self, filename, vert_pos):
        if self.cfg.split_data:
            faces = self.faces_complete
        else:
            faces = self.faces
        if not self.dataset_y_up:
            vert_pos = coordinate_transform_cloth3d(vert_pos)
        write_obj(filename, vert_pos, faces, self.vt, self.ftc)

    def write_vert_pos_pickle(self, filename, vert_pos, faces=None):
        if faces is None:
            if self.cfg.split_data:
                faces = self.faces_complete
            else:
                faces = self.faces
        if not self.dataset_y_up:
            vert_pos = coordinate_transform_cloth3d(vert_pos)
        write_vert_pos_pickle(filename, vert_pos, faces)

    def convert_large_numpy_to_mmap(self, threshold=int(1e6)):
        tmp_prefix = osp.join(os.environ['TMPDIR'], 'large_numpy_arrays')
        os.makedirs(tmp_prefix, exist_ok=True)
        variables = vars(self)
        sizes = []
        for key in variables:
            v = getattr(self, key)
            if isinstance(v, np.ndarray):
                sizes.append(v.size)
            if isinstance(v, np.ndarray) and v.size > threshold:
                target_file = get_unique_name(tmp_prefix, key + '.dat')
                mmap_writer = np.memmap(target_file, dtype=v.dtype, mode='w+', shape=v.shape)
                mmap_writer[:] = v[:]
                mmap_writer.flush()
                mmap_reader = DynamicMmap(target_file, v.dtype, 'r', v.shape)
                setattr(self, key, mmap_reader)


def load_base_data(prefix, cfg):
    return load_body_dataset(prefix, cfg)

def load_body_dataset(prefix, cfg=None):
    load_dict = np.load(osp.join(prefix, 'base_physics_format_fixed_uv.npy'), allow_pickle=True).item()

    temporal_related = ['vert_pos', 'vert_pos_gt', 'masks', 'body_pos', 'dg_base',
                        'sdf', 'Is', 'pose', 'translation']
    vertex_related = ['vert_pos', 'vert_pos_gt', 'rest_pos']
    faces_related = ['Is', 'sdf']
    need_smooth = ['vert_pos', 'vert_pos_gt']
    coord_transform_related = ['vert_pos', 'vert_pos_gt', 'rest_pos', 'body_pos', 'translation', 'sdf']
    vertex_scale_related = ['vert_pos', 'vert_pos_gt', 'rest_pos', 'body_pos', 'translation', 'sdf', 'distances']

    original_length = load_dict['vert_pos_gt'].shape[0]
    reverse = cfg.reverse if cfg is not None else False
    static_pose = cfg.static_pose if cfg is not None else False
    slowdown_ratio = cfg.slowdown_ratio if cfg is not None else -1
    target_length = int(original_length / slowdown_ratio) if slowdown_ratio > 0 else original_length
    gaussian_filter_sigma = cfg.gaussian_filter_sigma if cfg is not None else -1
    start_frame = cfg.start_frame if cfg is not None else 0
    set_cloth3d_y_up = cfg.set_cloth3d_y_up if cfg is not None else False
    convert_z_up = cfg.convert_z_up if cfg is not None else False
    scale_vert = cfg.scale_vert if cfg is not None else -1

    faces = load_dict['faces'].astype(np.int64)
    load_dict.pop('faces')

    if gaussian_filter_sigma > 0:
        for k in need_smooth:
            if k in load_dict and load_dict[k] is not None:
                v = load_dict[k]
                v2 = smooth_t(v, gaussian_filter_sigma, source_length=original_length)
                load_dict[k] = v2

    for k in temporal_related:
        if k in load_dict and load_dict[k] is not None:
            v = load_dict[k]
            mode = 'linear' if v.dtype == np.float32 or v.dtype == np.float64 else 'nearest'
            v2 = interpolate_t(v, target_length, reverse=reverse, mode=mode, source_length=original_length,
                               make_static=static_pose, start_frame=start_frame)
            load_dict[k] = v2

    if 'hood' in prefix:
        v = load_dict['rest_pos']
        rot_mat = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        v = rot_mat @ v[..., None]
        v = v[..., 0]
        load_dict['rest_pos'] = v
        # rot = np.array([0., 1., 0.]) * np.pi / 2
        # rot = aa2mat

    if ('cloth3d' in prefix and set_cloth3d_y_up) or convert_z_up:
        for k in coord_transform_related:
            if k in load_dict and load_dict[k] is not None:
                v = load_dict[k][..., -3:]
                v2 = coordinate_transform_cloth3d(v, invert=convert_z_up)
                load_dict[k][..., -3:] = v2

    if scale_vert > 0:
        for k in vertex_scale_related:
            if k in load_dict and load_dict[k] is not None:
                if k != 'sdf':
                    load_dict[k] = load_dict[k] * scale_vert
                else:
                    load_dict[k][0] = load_dict[k][0] * scale_vert

    vert_pos = load_dict['vert_pos']
    load_dict.pop('vert_pos')
    vert_pos_gt = load_dict['vert_pos_gt']
    load_dict.pop('vert_pos_gt')
    rest_pos = load_dict['rest_pos']
    load_dict.pop('rest_pos')

    masks = load_dict['masks']
    load_dict.pop('masks')

    if 'F_body' in load_dict:
        load_dict['F_body'] = load_dict['F_body'].astype(np.int64)
    if 'body_pos' in load_dict:
        load_dict['body_pos'] = load_dict['body_pos'].astype(np.float32)

    if vert_pos is None:
        # For backward compatibility.
        vert_pos = vert_pos_gt

    return vert_pos, vert_pos_gt, rest_pos, faces, masks, load_dict



class HandleControlledSequence(BasePhysicsDataset, Dataset):
    def __init__(self, prefix, cfg, device='cpu', debug=False, use_mmap=False, face_num_limit=None):
        super(HandleControlledSequence, self).__init__(prefix, cfg, debug, device)

        if cfg.use_jacobian:
            local_frames_gt = get_local_frames(self.vert_pos_gt, self.faces)
            inv_rest_local_frame = np.linalg.inv(get_local_frames(self.rest_pos, self.faces))
            self.inv_rest_local_frame = inv_rest_local_frame

            inv_local_frames_gt = np.linalg.inv(local_frames_gt)
            rhs = inv_local_frames_gt

            self.jacobians = local_frames_gt[1:] @ rhs[:-1]

            self.stretch_base = self.repr_get_relative_stretch(self.vert_pos, self.vert_pos_gt, self.cfg)

            self.centroids_base = self.repr_get_centroids(self.vert_pos, self.vert_pos_gt, self.cfg)

            self.orients_base = self.repr_get_orients(None, self.vert_pos_gt, self.cfg)

            self.singular_value, uv = self.repr_get_singular_value(self.stretch_base, self.cfg)

            self.L = create_face_laplace(self.faces, use_torch=True)

            if cfg.add_base_deformation:
                self.base_deformations = self.repr_get_base_deform(None, self.body_pos, self.cfg, self.Is[:-1])

            else:
                self.base_deformation = None

        if cfg.use_heuristic_boundary:
            up_axis = 1 if self.dataset_y_up else 2
            self.heuristic_boundary = np.where(heuristic_boundary_cloth3d(self.rest_pos, self.faces, up_axis=up_axis))[0].tolist()

        if cfg.normalization:
            self.prepare_normalization()
        if use_mmap:
            self.convert_large_numpy_to_mmap()
            
    def repr_get_g_velo(self, vert_pos, vert_pos_gt, cfg):
        vert_mean_pos_gt = vert_pos_gt.mean(axis=-2)
        global_velo = vert_mean_pos_gt[1:] - vert_mean_pos_gt[:-1]
        return global_velo

    @numpy_wrapper
    def repr_get_relative_stretch(self, vert_pos, vert_pos_gt, cfg):
        if vert_pos is not None and vert_pos.device != torch.device('cpu'):
            inv_rest_local_frame = torch.from_numpy(self.inv_rest_local_frame).to(vert_pos.device)
        else:
            inv_rest_local_frame = self.inv_rest_local_frame
        vert_pos_base = vert_pos_gt
        local_frames_base = get_local_frames(vert_pos_base, self.faces)
        relative_stretch = local_frames_base @ inv_rest_local_frame
        return relative_stretch

    def repr_get_singular_value(self, relative_stretch, cfg):
        if cfg is None:
            cfg = self.cfg
        if not cfg.predict_singular_value:
            return None, None
        if isinstance(relative_stretch, torch.Tensor):
            U, sigma, V = batched_svd_torch(relative_stretch)
        else:
            U, sigma, V = batched_svd(relative_stretch)
            if isinstance(relative_stretch, torch.Tensor):
                sigma = torch.from_numpy(sigma).to(torch.float32)
                U = torch.from_numpy(U).to(torch.float32)
                V = torch.from_numpy(V).to(torch.float32)
        return sigma, None

    def repr_get_centroids(self, vert_pos, vert_pos_gt, cfg):
        vert_pos_base = vert_pos_gt
        if cfg.remove_translation:
            vert_pos_base = vert_pos_base - vert_pos_base.mean(axis=-2, keepdims=True)
        n_frames = vert_pos_base.shape[0]
        centroids = self.get_centroid(vert_pos_base) if not cfg.use_static_centroid else self.centroids[:n_frames]
        return centroids

    def repr_get_orients(self, vert_pos, vert_pos_gt, cfg):
        if cfg.use_face_orientation:
            orients = self.get_face_orient_repr(vert_pos_gt, cfg=cfg)
        else:
            orients = None
        return orients

    @numpy_wrapper
    def repr_get_base_deform(self, vert_pos_gt, body_pos, cfg, I=None):
        if cfg.add_base_deformation:
            body_centroid = body_pos[..., self.body_faces, :].mean(axis=-2)
            if I is None:
                centroids = self.get_centroid(vert_pos_gt)
                Is = []
                for i in range(centroids.shape[0]):
                    _, I, _ = signed_distance(centroids[i].cpu().numpy(), body_pos[i].cpu().numpy(),
                                                  self.body_faces)
                    Is.append(I)
                Is = np.stack(Is)
                I = torch.from_numpy(Is)
            body_local_frame = get_local_frames(body_pos, self.body_faces)
            body_jacobians = body_local_frame[1:] @ torch.linalg.inv(body_local_frame[:-1])
            body_velo = body_centroid[1:] - body_centroid[:-1]

            idx_x = torch.arange(body_jacobians.shape[0]).unsqueeze(-1)
            body_jacobians = body_jacobians[idx_x, I]
            body_velo = body_velo[idx_x, I]

            base_deformation = torch.cat((body_jacobians.reshape(body_jacobians.shape[:-2] + (9, )), body_velo), dim=-1)
        else:
            base_deformation = None
        return base_deformation

    def get_face_velo(self, vert_pos, vert_pos_gt, cfg):
        pass

    def prepare_normalization(self, dataset=None):
        cfg = self.cfg
        if dataset is None:
            dataset = self
        if cfg.std_mean_input is None:
            cfg.std_mean_input, cfg.std_mean_output = get_std_mean_fast(dataset)
            to_save_dict = {'std_mean_input': cfg.std_mean_input, 'std_mean_output': cfg.std_mean_output}
            torch.save(to_save_dict, cfg.mean_var_path)

    def __len__(self):
        n_frames = self.vert_pos.shape[0] - self.cfg.cond_length
        if self.cfg.split_data != 0 and not self.cfg.fixed_downsample:
            return n_frames * self.cfg.split_data
        else:
            return n_frames

    @staticmethod
    def compose_repr(relative_stretch=None, centroids=None, sdf=None,
                     orients=None, base_deform=None, singular_value=None,
                     requires_readable_repr=False, global_velo=None):
        to_concat = []

        if requires_readable_repr:
            readable_repr = []
            offset = 0

            def append_readable_repr(name, tensor):
                nonlocal offset
                readable_repr.append([name, offset, offset + tensor.shape[-1]])
                offset += tensor.shape[-1]
        else:
            append_readable_repr = lambda name, tensor: None

        if relative_stretch is not None:
            relative_stretch = relative_stretch.reshape(relative_stretch.shape[:-2] + (9,))
            to_concat.append(relative_stretch)
            append_readable_repr('relative_stretch', relative_stretch)
        if orients is not None:
            to_concat.append(orients)
            append_readable_repr('orients', orients)
        if centroids is not None:
            to_concat.append(centroids)
            append_readable_repr('centroids', centroids)
        if sdf is not None:
            sdf = sdf.permute(1, 2, 3, 0)
            sdf = sdf.reshape(sdf.shape[:-2] + (sdf.shape[-1] * sdf.shape[-2],))
            to_concat.append(sdf)
            append_readable_repr('sdf', sdf)
        if base_deform is not None:
            base_deform = base_deform.unsqueeze(0).expand(to_concat[-1].shape[0], -1, -1)
            to_concat.append(base_deform)
            append_readable_repr('base_deform', base_deform)
        if singular_value is not None:
            to_concat.append(singular_value)
            append_readable_repr('singular_value', singular_value)
        if global_velo is not None:
            to_concat.append(global_velo)
            append_readable_repr('global_velo', global_velo)
        in_f = torch.cat(to_concat, dim=-1)
        in_f = in_f.permute(1, 0, 2)
        
        if requires_readable_repr:
            print(readable_repr)
        return in_f

    @property
    def sample_size(self):
        if self.cfg.split_data:
            return self.stretch_base.shape[1] // self.cfg.split_data
        else:
            return self.stretch_base.shape[1]

    def __getitem__(self, idx):
        """
        Pre-computed variables:

        stretch_base
        centroids_base
        handle_indicator
        orients_base
        sdf
        base_deformation
        singular_value
        """
        cfg = self.cfg
        if cfg.split_data:
            if cfg.fixed_downsample:
                split_idx = 0
            else:
                split_idx = idx % cfg.split_data
                idx = idx // cfg.split_data
            sli = slice(self.sep_point[split_idx], self.sep_point[split_idx + 1])
            # lst = list(range(self.faces.shape[0]))
            # random.shuffle(lst)
            # sli = lst[:self.faces.shape[0] // cfg.split_data]
        else:
            sli = slice(None)

        sli_frames = slice(idx, idx + cfg.cond_length)
        centroids = to_tensor(self.centroids_base[sli_frames, sli])
        relative_stretch = to_tensor(self.stretch_base[sli_frames, sli])
        orients = to_tensor(self.orients_base[sli_frames, sli]) if self.orients_base is not None else None

        if cfg.add_base_deformation:
            frame_idx = idx + cfg.cond_length - 1
            base_deform = self.base_deformations[frame_idx, sli]
            base_deform = to_tensor(base_deform)
        else:
            base_deform = None

        if self.global_velo is not None:
            global_velo = to_tensor(self.global_velo[idx: idx + cfg.cond_length - 1])
        else:
            global_velo = None

        if cfg.use_sdf:
            sdf = to_tensor(self.sdf[:cfg.use_sdf, sli_frames, sli])
        else:
            sdf = None

        if cfg.predict_singular_value:
            singular_value = to_tensor(self.singular_value[sli_frames, sli])
        else:
            singular_value = None

        if cfg.add_base_velocity:
            body_g_velo = to_tensor(self.body_g_velo[idx + 1: idx + cfg.cond_length])
        else:
            body_g_velo = None

        in_dict2 = self.get_repr_from_pos(relative_stretch=relative_stretch, centroids=centroids, orients=orients,
                                          base_deformation=base_deform, global_velo=global_velo,
                                          sdf=sdf,
                                          singular_value=singular_value,
                                          body_g_velo=body_g_velo)

        out_f = to_tensor(self.jacobians[idx + cfg.cond_length - 1, sli]).reshape((-1, 9))
        if cfg.predict_singular_value:
            out_f = torch.cat([out_f, to_tensor(self.singular_value[idx + cfg.cond_length, sli])], dim=-1)

        out_dict = {'f': out_f}

        if self.global_velo is not None:
            out_dict['g_velo'] = to_tensor(self.global_velo[idx + cfg.cond_length - 1])

        if self.cfg.geodesic:
            in_dict2["geodesic"] = self.geodesic[sli, sli]

        return in_dict2, out_dict

    def get_body_params(self, dest_frame):
        cfg = self.cfg
        idx = dest_frame - cfg.cond_length
        body_pose = to_tensor(self.body_pose[idx: idx + cfg.cond_length + 1]).reshape(-1)
        body_shape = to_tensor(self.body_shape[0]).reshape(-1)
        body_velo = to_tensor(self.body_velocity[idx: idx + cfg.cond_length + 1]).reshape(-1)
        return torch.cat([body_pose, body_shape, body_velo], dim=-1)

    def compose_global_velo_perface(self, g_velo):
        g_velo = torch.cat([g_velo[:1], g_velo], axis=0)
        return g_velo

    def get_repr_from_pos(self, vert_pos=None, vert_pos_gt=None, sdf=None, cfg=None,
                          body_pos=None, body_pos0=None, requires_readable_repr=False,
                          relative_stretch=None, centroids=None, orients=None, base_deformation=None,
                          global_velo=None, singular_value=None, body_g_velo=None,
                          use_gpu=False):
        if cfg is None:
            cfg = self.cfg

        device = 'cuda' if use_gpu else 'cpu'

        if vert_pos is not None:
            vert_pos = vert_pos.to(device)
        if vert_pos_gt is not None:
            vert_pos_gt = vert_pos_gt.to(device)
        if relative_stretch is not None:
            relative_stretch = relative_stretch.to(device)
        if centroids is not None:
            centroids = centroids.to(device)
        if base_deformation is not None:
            base_deformation = base_deformation.to(device)
        if body_pos is not None:
            body_pos = body_pos.to(device)
        if body_pos0 is not None:
            body_pos0 = body_pos0.to(device)
        if sdf is not None:
            sdf = sdf.to(device)
        if body_g_velo is not None:
            body_g_velo = body_g_velo.to(device)

        if relative_stretch is None:
            relative_stretch = self.repr_get_relative_stretch(vert_pos, vert_pos_gt, cfg=cfg)
        if centroids is None:
            centroids = self.repr_get_centroids(vert_pos, vert_pos_gt, cfg=cfg)
        if base_deformation is None:
            body_pos_combined = torch.stack([body_pos0, body_pos], dim=0)
            base_deformation = self.repr_get_base_deform(vert_pos_gt[-1:], body_pos_combined, cfg=cfg)[0]
        if orients is None:
            orients = self.repr_get_orients(vert_pos, vert_pos_gt, cfg=cfg)
        if singular_value is None:
            singular_value, uv = self.repr_get_singular_value(relative_stretch, cfg=cfg)

        in_f = self.compose_repr(relative_stretch, centroids,sdf,
                                 base_deform=base_deformation, requires_readable_repr=requires_readable_repr,
                                 orients=orients, singular_value=singular_value, global_velo=None)

        in_dict = {'f': in_f}

        if cfg.predict_g_velo:
            if global_velo is None:
                global_velo = self.repr_get_g_velo(None, vert_pos_gt, cfg)
            if body_g_velo is not None:
                global_velo = torch.cat([global_velo, body_g_velo], dim=-1)

            in_dict['g_velo'] = global_velo

        return in_dict

    def get_poisson_list(self):
        if self.cfg.use_heuristic_boundary:
            return self.heuristic_boundary
        else:
            return []

    def forward_control(self, vert_pos, f_idx, eps=0.):
        body_vert = self.load_dict['body_pos'][f_idx]
        F_body = self.load_dict['F_body']
        new_vert, collision_vert = body_cloth_collision(vert_pos, body_vert, F_body, eps=eps)
        collision_face = collision_vert[self.faces].any(dim=-1)
        return new_vert, collision_face

    def find_nn(self, vert_pos, continuous=False, vert_pos_gt=None):
        if vert_pos_gt is None:
            vert_pos_gt = self.vert_pos_gt
        if not continuous:
            diff = vert_pos.unsqueeze(1) - vert_pos_gt
            diff = diff.pow(2).sum(dim=-1).mean(dim=-1)
            return diff.argmin(dim=-1)
        else:
            raise Exception('Not implemented')


class MultipleDataset(Dataset):
    def __init__(self, prefixes, face_num_limit=None, **kargs):
        cfg = kargs['cfg']
        normalization = cfg.normalization

        if normalization:
            cfg.normalization = 0
            mean_var_path = cfg.mean_var_path

        base_data_class = HandleControlledSequence

        self.datasets = []
        self.sample_sizes = set()
        self.Ls = []
        names = []
        failed_dataset = []
        n_faces = []
        for prefix in tqdm(prefixes):
            print(f'Loading {prefix}')
            new_dataset = base_data_class(prefix, **kargs)
            l = len(new_dataset) # check if the dataset is empty
            assert l >= 0
            if face_num_limit is not None:
                if new_dataset.faces.shape[0] > face_num_limit:
                    continue
            self.datasets.append(new_dataset)
            self.Ls.append(new_dataset.L)
            self.sample_sizes.add(self.datasets[-1].sample_size)
            n_faces.append(new_dataset.faces.shape[0])
            names.append(prefix)
        self.lengths = [len(d) for d in self.datasets]
        self.use_incremental_statistics = 0

        if len(failed_dataset) > 0:
            print(f'Failed to load:\n{failed_dataset}')

        if normalization:
            cfg.normalization = 1
            self.datasets[0].prepare_normalization(self)

    @property
    def multiple_sample_size(self):
        return len(self.sample_sizes) > 1

    def __getitem__(self, idx):
        for i, l in enumerate(self.lengths):
            if idx < l:
                res = self.datasets[i][idx]
                return res, i
            idx -= l
        raise IndexError

    def __len__(self):
        return sum(self.lengths)

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        return getattr(self.datasets[0], item)

    def find_nn(self, vert_pos, continuous=False):
        return self.datasets[0].find_nn(vert_pos, continuous=continuous, vert_pos_gt=self.all_vert_pos_gt)
