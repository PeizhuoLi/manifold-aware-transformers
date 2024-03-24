import argparse
import sys
import os


class BaseOptionParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--save_path', type=str, default='./results/test')
        self.parser.add_argument('--dataset_path', type=str, default='./data/batch1/moving1_topo1_Cotton')
        self.parser.add_argument('--device', type=str, default='cuda')
        self.parser.add_argument('--cond_length', type=int, default=1)
        self.parser.add_argument('--debug', type=int, default=0)
        self.parser.add_argument('--use_jacobian', type=int, default=0)
        self.parser.add_argument('--n_layers', type=int, default=-1)
        self.parser.add_argument('--use_static_centroid', type=int, default=0)
        self.parser.add_argument('--noise_level', type=float, default=0.0)
        self.parser.add_argument('--normalization', type=int, default=0, help='Normalize the input and output')
        self.parser.add_argument('--extend_static', type=int, default=0, help='Extend the static part of the training sequence')
        self.parser.add_argument('--multiple_dataset', type=str, default='', help='Path to multiple data prefix')
        self.parser.add_argument('--part_training_data', type=int, default=-1, help='Use only part of the training data')
        self.parser.add_argument('--attention_head', type=int, default=4)
        self.parser.add_argument('--attention_dropout', type=float, default=0.0)
        self.parser.add_argument('--sample_rate', type=float, default=1)
        self.parser.add_argument('--t_d_model', type=int, default=512)
        self.parser.add_argument('--t_n_layers', type=int, default=6)
        self.parser.add_argument('--t_n_head', type=int, default=8)
        self.parser.add_argument('--t_dim_feedforward', type=int, default=2048)
        self.parser.add_argument('--split_data', type=int, default=0)
        self.parser.add_argument('--remove_translation', type=int, default=0, help='Remove centroid from mesh')
        self.parser.add_argument('--predict_global_velo', type=int, default=0)
        self.parser.add_argument('--use_relative_stretch', type=int, default=1)
        self.parser.add_argument('--use_mmap', type=int, default=0)
        self.parser.add_argument('--use_sdf', type=int, default=0, choices=[0, 1])
        self.parser.add_argument('--normalize_sdf', type=int, default=0, choices=[0, 1])
        self.parser.add_argument('--transformer_activation', type=str, default='relu')
        self.parser.add_argument('--use_face_orientation', type=int, default=0, help='Use face normal orientation')
        self.parser.add_argument('--face_orientation_usage', type=str, default='concat',
                                 choices=['concat', 'cosine'], help='Concat face orientation or just cosine to the gravity direction')
        self.parser.add_argument('--fixed_downsample', type=int, default=0, help='Fixed downsample')
        self.parser.add_argument('--add_base_deformation', type=int, default=0, help='Add base deformation to the input')
        self.parser.add_argument('--normalize_base_deformation', type=int, default=0)
        self.parser.add_argument('--predict_singular_value', type=int, default=0, help='Predict singular value')
        self.parser.add_argument('--add_base_velocity', type=int, default=0, help='Use body velocity as the input for predicting global velocity')
        self.parser.add_argument('--slowdown_ratio', type=float, default=-1, help='Slow down the training')
        self.parser.add_argument('--gaussian_filter_sigma', type=float, default=-1, help='Filtering noise')
        self.parser.add_argument('--geodesic', type=int, default=0, help='Use geodesic distance in predicting vertices non-aligned by the same seam')
        self.parser.add_argument('--geodesic_power', type=float, default=1.0, help='Adjust how far the geodesic distance looks.')
        self.parser.add_argument('--set_cloth3d_y_up', type=int, default=0, help='Set y axis as up for cloth3d')
        self.parser.add_argument('--n_g_heads', type=int, default=2, help='number of attention heads being overwritten with geodesic distance')
        self.parser.add_argument('--g_in_place', type=int, default=0, help='geodesic distance is weight to attn, does not substitute attn')

    @staticmethod
    def save(filename, args_str=None):
        if args_str is None:
            args_str = ' '.join(sys.argv[1:])
        path = '/'.join(filename.split('/')[:-1])
        os.makedirs(path, exist_ok=True)
        with open(filename, 'w') as file:
            file.write(args_str)

    def load(self, filename):
        with open(filename, 'r') as file:
            args_str = file.readline()
        return self.parse_args(args_str.split())

    @staticmethod
    def checker(args):
        if not args.geodesic:
            args.n_g_heads = 0
        return args

    def parse_args(self, args_str=None):
        return self.checker(self.parser.parse_args(args_str))


class TrainOptionParser(BaseOptionParser):
    def __init__(self):
        super(TrainOptionParser, self).__init__()
        self.parser.add_argument('--lr', type=float, default=1e-3)
        self.parser.add_argument('--batch_size', type=int, default=16)
        self.parser.add_argument('--dec_factor', type=float, default=1e-2)
        self.parser.add_argument('--use_tqdm', type=int, default=0)
        self.parser.add_argument('--num_epochs', type=int, default=2000)
        self.parser.add_argument('--continue_train', type=int, default=0)
        self.parser.add_argument('--save_gap', type=int, default=50)
        self.parser.add_argument('--save_freq', type=int, default=30)
        self.parser.add_argument('--gradient_clip', type=float, default=-1)
        self.parser.add_argument('--f_loss_type', type=str, default='L1')
        self.parser.add_argument('--g_velo_loss_type', type=str, default='L1')
        self.parser.add_argument('--lambda_f', type=float, default=1.0)
        self.parser.add_argument('--lambda_g_velo', type=float, default=0.)
        self.parser.add_argument('--ddp', type=int, default=0)
        self.parser.add_argument('--iterative_batch', type=int, default=1)
        self.parser.add_argument('--finetune_pretrained_model', type=str, default=None)
        self.parser.add_argument('--face_num_limit', type=int, default=None, help='Face number limit')

    @staticmethod
    def checker(args):
        args = super(TrainOptionParser, TrainOptionParser).checker(args)
        return args


class TestOptionParser(BaseOptionParser):
    def __init__(self):
        super(TestOptionParser, self).__init__()
        self.save_path = None
        self.parser.add_argument('--epoch', type=str, default=-1)
        self.parser.add_argument('--mode', type=str, default='metrics')
        self.parser.add_argument('--autoregressive', type=int, default=0)
        self.parser.add_argument('--n_frames', type=int, default=40)
        self.parser.add_argument('--test_noise_level', type=float, default=0.0)
        self.parser.add_argument('--print_final_result', type=int, default=0)
        self.parser.add_argument('--another_dataset', type=str, default='')
        self.parser.add_argument('--override_extend_static', type=int, default=None)
        self.parser.add_argument('--override_multiple_dataset', type=str, default=None)
        self.parser.add_argument('--override_part_training_data', type=int, default=None)
        self.parser.add_argument('--not_save_to_file', type=int, default=0)
        self.parser.add_argument('--override_attention_dropout', type=float, default=0.)
        self.parser.add_argument('--override_sample_rate', type=float, default=None)
        self.parser.add_argument('--runtime_batch_size', type=int, default=-1)
        self.parser.add_argument('--path_to_post_model', type=str, default=None)
        self.parser.add_argument('--override_mask_order', type=int, default=None)
        self.parser.add_argument('--override_slowdown_ratio', type=float, default=None)
        self.parser.add_argument('--reverse', type=int, default=0)
        self.parser.add_argument('--clamp_down', type=float, default=None)
        self.parser.add_argument('--clamp_up', type=float, default=None)
        self.parser.add_argument('--static_pose', type=int, default=0)
        self.parser.add_argument('--override_gaussian_filter_sigma', type=float, default=None)
        self.parser.add_argument('--start_frame', type=int, default=0)
        self.parser.add_argument('--use_heuristic_boundary', type=int, default=0)
        self.parser.add_argument('--scale_vert', type=float, default=-1)
        self.parser.add_argument('--override_geodesic_power', type=float, default=None)
        self.parser.add_argument('--override_split_data', type=int, default=None)
        self.parser.add_argument('--export_path', type=str, default=None)

    def parse_args(self, args_str=None):
        res, _ = self.parser.parse_known_args(args_str)
        if self.save_path is None:
            self.save_path = res.save_path
        else:
            res.save_path = self.save_path
        if not res.export_path:
            res.export_path = res.save_path
        return res


def override_args(test_args, args):
    for key in test_args.__dict__.keys():
        if key.startswith('override_'):
            override_key = key.replace('override_', '')
            if test_args.__dict__[key] is not None:
                args.__dict__[override_key] = test_args.__dict__[key]
