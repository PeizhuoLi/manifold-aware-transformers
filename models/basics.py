import torch
import torch.nn as nn
from dataset.handle_dataset import HandleControlledSequence, MultipleDataset, DataReprConfig
import os.path as osp
import copy


def load_config_from_args(args):
    # Deal with backward compatibility
    args_as_dict = copy.copy(vars(args))
    args_as_dict['predict_g_velo'] = args.predict_global_velo
    args_as_dict.pop('predict_global_velo')
    args_as_dict['mean_var_path'] = osp.join(args.save_path, 'mean_var.pt')

    return DataReprConfig(**args_as_dict)


def load_dataset_from_args(args):
    if args.multiple_dataset:
        with open(args.multiple_dataset, 'r') as f:
            dataset_names = f.readlines()
        dataset_names = [x.strip() for x in dataset_names]
        if args.part_training_data != -1:
            dataset_names = dataset_names[:args.part_training_data]
        dataset_class = MultipleDataset
    else:
        dataset_names = args.dataset_path
        dataset_class = HandleControlledSequence

    # Deal with backward compatibility
    data_repr_config = load_config_from_args(args)
    face_num_limit = args.face_num_limit if hasattr(args, 'face_num_limit') else None

    dataset = dataset_class(dataset_names, face_num_limit=face_num_limit, cfg=data_repr_config, debug=args.debug, use_mmap=args.use_mmap)
    return dataset
