import os
import pickle
import sys
import os.path as osp
import numpy as np
import argparse


def evaluate(boundary_garments, dataset_path, model_path, export_path):
    python_parse = sys.executable

    os.makedirs(export_path, exist_ok=True)
    all_names = [x for x in os.listdir(dataset_path) if osp.isdir(osp.join(dataset_path, x))]
    all_names.sort()
    all_res = []

    for all_name in all_names:
        args_run = f'--save_path={model_path} --device=cuda:0 --debug=0 --mode=sequence --test_noise_level=0 --print_final_result=1 --n_frames=300 --override_extend_static=0 --autoregressive=1 --static_pose=0'
        final_export = os.path.join(export_path, all_name)
        garment = all_name.split('_')[1]
        args_run += f' --another_dataset={osp.join(dataset_path, all_name)}'
        args_run += f' --export_path={final_export}'
        args_run += f' --use_heuristic_boundary={int(garment in boundary_garments)}'
        cmd = f'{python_parse} test_frame_based.py {args_run}'
        os.system(cmd)

        try:
            with open(os.path.join(final_export, 'summary_data.pickle'), 'rb') as handle:
                all_res.append(pickle.load(handle)['vert_diff'])
        except:
            print(f'Error: {all_name}')

    all_res = np.array(all_res).mean()
    print('Mean vertex error (cm):', all_res.mean())


def evaluate_cloth3d():
    evaluate(['Skirt', 'Trousers'], 'data/cloth3d-test', './pre-trained/cloth3d', './results/cloth3d-evaluation/')


def evaluate_vto():
    evaluate([], 'data/vto-test', './pre-trained/vto', './results/vto-evaluation/')


def main():
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--dataset', type=str, help='Model to evaluate', required=True, choices=['cloth3d', 'vto'])

    args = parser.parse_args()
    if args.dataset == 'cloth3d':
        evaluate_cloth3d()
    elif args.dataset == 'vto':
        evaluate_vto()


if __name__ == '__main__':
    main()
