import os
from mesh_utils import write_vert_pos_pickle, create_sdf_info_from_SC, construct_appr_geodesic_matrix
from models.downsample import create_downsample_sequence, downsample_by_sequence
import numpy as np
from tqdm import tqdm
import igl
from utils import signed_distance


def prepare_single_sequence(rest_pos, faces, cloth_pos, body_pos, F_body, translation=None,
                            decimate_ratio=None, export_dir=None):

    F_body = F_body.astype(np.int64)
    faces = faces.astype(np.int64)

    if decimate_ratio is not None:
        down_seq = create_downsample_sequence(rest_pos, faces, ratio=decimate_ratio)
    else:
        down_seq = None

    cloth_pos = cloth_pos.astype(np.float32)

    body_pos = body_pos.astype(np.float32)

    if down_seq is not None:
        vert_pos_gt = cloth_pos
        faces_ori = faces.copy()
        rest_pos, faces = downsample_by_sequence(rest_pos, faces_ori, down_seq)
        vert_pos_gt_down = []
        for j in tqdm(range(vert_pos_gt.shape[0])):
            vert_pos_gt_down.append(downsample_by_sequence(vert_pos_gt[j], faces_ori, down_seq)[0])
        vert_pos_gt = np.stack(vert_pos_gt_down, axis=0)
        cloth_pos = vert_pos_gt

    vert_pos_gt = cloth_pos

    Is = []
    vert_pos = None

    centroids = cloth_pos[..., faces, :].mean(axis=-2)
    sdfs = []
    offsets = [0]

    for offset in offsets:
        sdf = []
        for i in tqdm(range(cloth_pos.shape[0] - offset)):
            if offset == 0:
                S, I, C = signed_distance(centroids[i], body_pos[i], F_body)
                sdf.append(create_sdf_info_from_SC(S, C, centroids[i]))
                Is.append(I)
            elif offset == 1:
                S, I, C = igl.signed_distance(centroids[i], body_pos[i + 1], F_body)
                sdf.append(create_sdf_info_from_SC(S, C, centroids[i]))
                Is.append(I)
            else:
                raise Exception('Deprecated')
        for i in range(offset):
            sdf.append(sdf[-1])
        sdfs.append(np.stack(sdf, axis=0))

    sdfs = np.stack(sdfs, axis=0)
    Is = np.stack(Is, axis=0)

    if translation is None:
        translation = np.zeros((cloth_pos.shape[0], 3), dtype=np.float32)
        body_mean_pos = body_pos.mean(axis=1)
        # translation[1:] = body_mean_pos[1:] - body_mean_pos[:-1]
        translation = body_mean_pos

    geodesic_mat = construct_appr_geodesic_matrix(rest_pos, faces, t=1e-1, on_face=True)
    # geodesic_mat = None

    save_dict = {'vert_pos': vert_pos, 'vert_pos_gt': vert_pos_gt, 'rest_pos': rest_pos, 'faces': faces,
                 'masks': None, 'body_pos': body_pos, 'F_body': F_body, 'translation': translation, 'Is': Is,
                 'distances': geodesic_mat, 'sdf': sdfs}

    body_pos_extended = np.concatenate([vert_pos_gt, rest_pos[None]], axis=0)
    if export_dir is not None:
        os.makedirs(export_dir, exist_ok=True)
        write_vert_pos_pickle(os.path.join(export_dir, "body.pkl"), body_pos, F_body)
        write_vert_pos_pickle(os.path.join(export_dir, "body_collider.pkl"), body_pos[1:], F_body)
        write_vert_pos_pickle(os.path.join(export_dir, "garment.pkl"), vert_pos_gt, faces)
        write_vert_pos_pickle(os.path.join(export_dir, "garment-extend.pkl"), body_pos_extended, faces)
        if vert_pos is not None:
            write_vert_pos_pickle(os.path.join(export_dir, "vert_input.pkl"), vert_pos, faces)

        np.save(os.path.join(export_dir, 'base_physics_format_fixed_uv.npy'), save_dict, allow_pickle=True)

    return save_dict
