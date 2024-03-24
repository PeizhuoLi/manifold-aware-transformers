import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os.path as osp
from models.basics import load_dataset_from_args, load_config_from_args
from models.transformer import create_transformer_model_from_args
from tqdm import tqdm
from loss_recorder import LossRecorder
from utils import to_device, reshape_past, BatchedMultipleDatasetSampler, AbnormalDetector
from option import TrainOptionParser

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from torch.backends.cuda import sdp_kernel, SDPBackend

from dataset.handle_dataset import MultipleDataset

import socket
from contextlib import closing


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def ddp_setup(rank, world_size, port):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    if not isinstance(port, str):
        port = str(port)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def train(rank, world_size, args, t_model, port=None):
    device = torch.device(f'cuda:{rank}')
    t_model = t_model.to(device)
    dataset = load_dataset_from_args(args)

    if world_size > 1:
        if isinstance(dataset, MultipleDataset):
            assert not dataset.multiple_sample_size
        ddp_setup(rank, world_size, port)
        t_model = DDP(t_model, device_ids=[rank], output_device=rank)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                 sampler=DistributedSampler(dataset), num_workers=0)
    else:
        num_workers = 0 if not args.debug else 0
        if isinstance(dataset, MultipleDataset) and dataset.multiple_sample_size:
            data_sampler = BatchedMultipleDatasetSampler(dataset, args.batch_size)
            data_loader = DataLoader(dataset, num_workers=num_workers, batch_sampler=data_sampler)
        else:
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=num_workers)

    def t_model_state_dict():
        return t_model.module.state_dict() if world_size > 1 else t_model.state_dict()

    if rank == 0:
        loss_recorder_path = osp.join(args.save_path, 'tensorboard')
        if not args.continue_train:
            os.system(f'rm -rf {loss_recorder_path}')
        os.makedirs(loss_recorder_path, exist_ok=True)
        summary_writer = SummaryWriter(log_dir=loss_recorder_path)
        loss_recorder = LossRecorder(summary_writer)

    optimizer = torch.optim.Adam(t_model.parameters(), lr=args.lr)
    if hasattr(t_model, 'state_dict_optimizer') and t_model.state_dict_optimizer is not None:
        optimizer.load_state_dict(t_model.state_dict_optimizer)
        t_model.state_dict_optimizer = None

    abnormal_detector = AbnormalDetector(list(t_model.parameters()))
    nan_counter = 0

    criterion = {}
    for key in dataset.output_keys:
        criterion[key] = torch.nn.L1Loss() if getattr(args, f'{key}_loss_type') == 'L1' else torch.nn.MSELoss()
    criterion['rotation'] = torch.nn.L1Loss()

    if args.continue_train:
        checkpoint_filename = t_model.load_from_prefix(args.save_path, load_optimizer=True)
        optimizer.load_state_dict(t_model.state_dict_optimizer)
        del t_model.state_dict_optimizer

        epoch_start = int(checkpoint_filename.split('_')[1])
    else:
        epoch_start = 0

    time_last_save = time.time()
    for epoch in range(epoch_start, args.num_epochs + 1):
        if args.use_tqdm:
            loop = tqdm(enumerate(data_loader), total=len(data_loader))
        else:
            loop = enumerate(data_loader)

        optimizer.zero_grad()
        for it, batch in loop:
            # Prepare data for training
            (in_dict, gt_dict), idx = batch
            in_dict, gt_dict = dataset.cfg.normalize_pair(in_dict, gt_dict)
            to_device(in_dict, device)
            to_device(gt_dict, device)
            reshape_past(in_dict)

            # Forward
            out_dict = t_model(in_dict) ## here is where we forward

            # Compute loss and backward
            losses = {}
            for key in out_dict:
                losses[key] = criterion[key](out_dict[key], gt_dict[key])

            loss_total = sum([losses[k] * getattr(args, f'lambda_{k}') for k in losses])
            loss_total_backward = loss_total / args.iterative_batch
            loss_total_backward.backward()

            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(t_model.parameters(), args.gradient_clip)

            if args.debug and it % 10 == 0 and it != 0:
                print(loss_recorder.losses['loss_f'].loss_step)

            if (it + 1) % args.iterative_batch == 0:
                if abnormal_detector.detect():
                    nan_counter += 1
                    optimizer.zero_grad()
                optimizer.step()
                optimizer.zero_grad()

            # Record loss
            if rank == 0:
                loss_recorder.add_scalar('nan_counter', nan_counter)
                for k in losses:
                    loss_recorder.add_scalar(f'loss_{k}', losses[k].item())
                loss_recorder.add_scalar('loss_total', loss_total.item())

                loss_descript = ' '.join([f'{k}: {v.item():.8f}' for k, v in losses.items()])
                loss_descript = f'total: {loss_total.item():.8f} ' + loss_descript
                if args.use_tqdm:
                    loop.set_description(loss_descript)
                else:
                    if it % 50 == 0:
                        loop_descript = f'[{epoch}/{args.num_epochs}] [{it}/{len(data_loader)}] '
                        print(loop_descript + loss_descript)

                now_time = time.time()
                if now_time - time_last_save > args.save_gap * 60:
                    torch.save(t_model_state_dict(), osp.join(args.save_path, f'transformer_{epoch:05d}_{it:07d}.pth'))
                    torch.save(optimizer.state_dict(), osp.join(args.save_path, f'optimizer_{epoch:05d}_{it:07d}.pth'))
                    time_last_save = now_time

        if rank == 0:
            loss_recorder.epoch()
            if epoch % args.save_freq == 0:
                now_time = time.time()
                if now_time - time_last_save > args.save_gap * 60 or epoch == args.num_epochs:
                    torch.save(t_model_state_dict(), osp.join(args.save_path, f'transformer_{epoch}.pth'))
                    torch.save(optimizer.state_dict(), osp.join(args.save_path, f'optimizer_{epoch}.pth'))
                    time_last_save = now_time

    if world_size > 1:
        destroy_process_group()


def main():
    s_time = time.time()
    parser = TrainOptionParser()
    args = parser.parse_args()

    backend_map = {
        SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
        SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False, "enable_flash": False, "enable_mem_efficient": True
        }
    }

    parser.save(osp.join(args.save_path, 'args.txt'))

    if args.save_path.startswith('./results/test') and not args.continue_train:
        os.system(f'rm -rf {args.save_path}')
        os.system(f"rm -rf {osp.join(os.environ['TMPDIR'], 'large_numpy_arrays')}")

    os.makedirs(args.save_path, exist_ok=True)

    if args.finetune_pretrained_model:
        args_pretrained = parser.load(osp.join(args.finetune_pretrained_model, 'args.txt'))
        cfg = load_config_from_args(args_pretrained)
        cmd = f'cp {cfg.mean_var_path} {osp.join(args.save_path, "mean_var.pt")}'
        os.system(cmd)
    else:
        cfg = load_config_from_args(args)

    print('Time used for preparing data', time.time() - s_time)
    t_model = create_transformer_model_from_args(args, cfg)

    if args.finetune_pretrained_model:
        t_model.load_from_prefix(args.finetune_pretrained_model, load_optimizer=True)

    world_size = torch.cuda.device_count() if args.ddp else 1
    if world_size > 1:
        assert args.batch_size % world_size == 0
        args.batch_size = args.batch_size // world_size
        port = find_free_port()

        print(f'Found {world_size} GPUs, using DDP at port {port}')

        mp.spawn(train, args=(world_size, args, t_model, port), nprocs=world_size, start_method='spawn')
    else:
        with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
            train(0, world_size, args, t_model)


if __name__ == '__main__':
    main()
