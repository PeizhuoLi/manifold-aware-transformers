import argparse
import pickle
import os
import torch
from dataset.smpl import SMPL_Layer
from mesh_utils import write_vert_pos_pickle, create_sdf_info_from_SC, construct_appr_geodesic_matrix
from models.downsample import create_downsample_sequence, downsample_by_sequence
import numpy as np
from tqdm import tqdm
import igl
import os.path as osp
from utils import signed_distance


def save_obj(filename, vertices, faces):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in (faces + 1):  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    print("Saved:", filename)


def parse_single_sequence(data_path, export_dir, smpl_layer, rest_pos, requires_sdf=False,
                          sdf_on_face=True, down_seq=None):
    if os.path.exists(os.path.join(export_dir, 'base_physics_format_fixed_uv.npy')):
        print('Result exists, skip')
        return
    os.makedirs(export_dir, exist_ok=True)

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    cloth_pos = data['vertices']
    vert_pos_gt = cloth_pos

    faces = data['faces'].astype(int)

    if down_seq is not None:
        # down_seq = create_downsample_sequence(rest_pos, faces, ratio=0.2)
        faces_ori = faces.copy()
        rest_pos, faces = downsample_by_sequence(rest_pos, faces_ori, down_seq)
        vert_pos_gt_down = []
        for j in tqdm(range(vert_pos_gt.shape[0])):
            vert_pos_gt_down.append(downsample_by_sequence(vert_pos_gt[j], faces_ori, down_seq)[0])
        vert_pos_gt = np.stack(vert_pos_gt_down, axis=0)
        cloth_pos = vert_pos_gt

    collision_mask = np.zeros((cloth_pos.shape[0], cloth_pos.shape[1]), dtype=bool)

    body_vertices, _, T = smpl_layer(torch.from_numpy(data['pose']), torch.from_numpy(data['shape']), requires_mat=True)
    body_vertices = body_vertices.detach().numpy() + data['translation'][:, None]
    body_pos = body_vertices
    F_body = smpl_layer.faces.astype(int)
    Is = []

    vert_pos = None

    if requires_sdf:
        if sdf_on_face:
            centroids = cloth_pos[..., faces, :].mean(axis=-2)
        else:
            centroids = cloth_pos
        sdfs = []
        offsets = [0]
        for offset in offsets:
            sdf = []
            for i in tqdm(range(cloth_pos.shape[0] - offset)):
                if offset == 0:
                    S, I, C = signed_distance(centroids[i], body_vertices[i], F_body)
                    sdf.append(create_sdf_info_from_SC(S, C, centroids[i]))
                    Is.append(I)
                elif offset == 1:
                    # Not used
                    S, I, C = igl.signed_distance(centroids[i], body_vertices[i + 1], F_body)
                    sdf.append(create_sdf_info_from_SC(S, C, centroids[i]))
                    Is.append(I)
            for i in range(offset):
                sdf.append(sdf[-1])
            sdfs.append(np.stack(sdf, axis=0))

        sdfs = np.stack(sdfs, axis=0)
        Is = np.stack(Is, axis=0)

    write_vert_pos_pickle(os.path.join(export_dir, "body.pkl"), body_pos, F_body)
    write_vert_pos_pickle(os.path.join(export_dir, "body_collider.pkl"), body_pos[1:], F_body)
    write_vert_pos_pickle(os.path.join(export_dir, "garment.pkl"), vert_pos_gt, faces)
    if vert_pos is not None:
        write_vert_pos_pickle(os.path.join(export_dir, "vert_input.pkl"), vert_pos, faces)

    geodesic_mat = construct_appr_geodesic_matrix(rest_pos, faces, t=1e-1, on_face=True)

    save_dict = {'vert_pos': vert_pos, 'vert_pos_gt': vert_pos_gt, 'rest_pos': rest_pos, 'faces': faces,
                 'masks': collision_mask, 'body_pos': body_pos, 'F_body': F_body, 'pose': data['pose'],
                 'shape': data['shape'], 'translation': data['translation'], 'Is': Is,
                 'distances': geodesic_mat}

    # Only vert_pos_gt, rest_pos, faces, body_pos, F_body, translation, Is are still in use.

    if requires_sdf:
        save_dict['sdf'] = sdfs

    np.save(os.path.join(export_dir, 'base_physics_format_fixed_uv.npy'), save_dict, allow_pickle=True)


def main():
    parser = argparse.ArgumentParser(
        description='Extract meshes from .pkl and save as .obj'
    )

    parser.add_argument(
        '--data_path_prefix',
        type=str,
    )
    parser.add_argument(
        '--decimate_ratio',
        type=float,
        default=0.2
    )
    parser.add_argument(
        '--save_path',
        type=str,
    )

    args = parser.parse_args()
    save_path = args.save_path
    files = os.listdir(args.data_path_prefix)
    files = [f for f in files if f.endswith('.pkl')]
    files.sort()

    if 'dress' in args.data_path_prefix:
        static_obj_path = osp.join(args.data_path_prefix, 'dress.obj')
    else:
        static_obj_path = osp.join(args.data_path_prefix, 'tshirt.obj')

    order_list = [f for f in os.listdir(osp.join(args.data_path_prefix, 'simulations')) if f.endswith('.pkl') ]
    order_list.sort()

    smpl_layer = SMPL_Layer(gender='female')

    rest_pos, _, _, faces, _, _ = igl.read_obj(static_obj_path)
    decimate_ratio = args.decimate_ratio

    if decimate_ratio < 1:
        print('Simplifying mesh with ratio', decimate_ratio)
        down_seq = create_downsample_sequence(rest_pos, faces, ratio=decimate_ratio)
    else:
        down_seq = None

    for i, f in enumerate(order_list):
        print(f'[{i}/{len(order_list)}]')
        parse_single_sequence(os.path.join(args.data_path_prefix, 'simulations', f),
                              os.path.join(save_path, f.split('/')[-1][:-4]),
                              smpl_layer, rest_pos, requires_sdf=True, sdf_on_face=True,
                              down_seq=down_seq)


if __name__ == '__main__':
    main()
