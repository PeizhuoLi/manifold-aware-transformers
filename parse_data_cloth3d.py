import argparse
import os
import os.path as osp
import numpy as np
import igl
from dataset.create_npy_format import prepare_single_sequence


import sys
sys.path.append('./data/cloth3d/DataReader')
from read import DataReader


def save_obj(filename, vertices, faces):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for f in (faces + 1):  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    print("Saved:", filename)


def quads2tris(F):
    out = []
    for f in F:
        if len(f) == 3: out += [f]
        elif len(f) == 4: out += [[f[0],f[1],f[2]],
                                [f[0],f[2],f[3]]]
        else: print("This should not happen...")
    return np.array(out, np.int32)


def parse_cloth3d(data_path, export_dir_o, decimate_ratio):
    reader = DataReader()

    info = reader.read_info(data_path)
    n_frames = info['trans'].shape[1]

    _, F_body = reader.read_human(data_path, 0)
    F_body = np.array(F_body, dtype=np.int64)

    for garment in info['outfit']:
        rest_pos, _, _, _, _, _ = igl.read_obj(osp.join(data_path, garment + '.obj'))
        faces = reader.read_garment_topology(data_path, garment)
        faces = quads2tris(faces)

        export_dir = export_dir_o + '_' + garment + '_' + info['outfit'][garment]['fabric']

        translation = info['trans'].astype(np.float32).T
        
        if info['outfit'][garment]['fabric'] != 'cotton':
            # We use cotton material only, and skip the other materials
            continue

        if os.path.exists(os.path.join(export_dir, 'base_physics_format_fixed_uv.npy')):
            print('Result exists, skip')
            continue
            pass

        cloth_pos = []
        body_vertices = []
        for i in range(n_frames):
            cloth_pos.append(reader.read_garment_vertices(data_path, garment, i))
            body_vertices.append(reader.read_human(data_path, i)[0])
        cloth_pos = np.stack(cloth_pos, axis=0)
        body_vertices = np.stack(body_vertices, axis=0)
        body_vertices = body_vertices.astype(np.float32)

        prepare_single_sequence(rest_pos, faces, cloth_pos, body_vertices, F_body, export_dir=export_dir,
                                decimate_ratio=decimate_ratio, translation=translation)


def main():
    parser = argparse.ArgumentParser(
        description='Extract meshes from cloth3d format and save as .obj'
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
    files = [f for f in files if osp.isdir(osp.join(args.data_path_prefix, f))]
    files.sort()

    os.makedirs(save_path, exist_ok=True)

    for i, f in enumerate(files):
        print(f'[{i}/{len(files)}]')
        parse_cloth3d(os.path.join(args.data_path_prefix, f), os.path.join(save_path, f),
                      args.decimate_ratio)


if __name__ == '__main__':
    main()
