import math

import torch
import torch.nn as nn
from torch.nn.modules import TransformerEncoderLayer, TransformerEncoder
from models.fast_geodesic_transformer import GeoTransformerEncoderLayer, set_geodesic
from utils import get_noise_level_from_args

import os.path as osp
import os
from utils import checkpoint_sort_key


def create_transformer_model_from_args(args, dataset):
    noise_level = get_noise_level_from_args(args, dataset)
    t_model = TransformerModel(n_channels_input=dataset.n_channel_total, n_channels_output=dataset.n_channel_output, d_model=args.t_d_model,
                               d_hid=args.t_dim_feedforward, n_head=args.t_n_head, n_layers=args.t_n_layers, dropout=args.attention_dropout,
                               noise_level=noise_level, n_channels_velo=dataset.n_channel_g_velo, predict_g_velo=dataset.predict_g_velo,
                               activation=args.transformer_activation,
                               num_geodesic=args.n_g_heads, in_place=args.g_in_place)

    return t_model


class TransformerModel(nn.Module):
    def __init__(self, n_channels_input, n_channels_output, d_model, d_hid, n_head, n_layers, dropout=0.5,
                 noise_level=0., n_channels_velo=0, predict_g_velo=0, activation='relu',
                 num_head_mask=-1, num_layer_mask=-1, num_geodesic=2, in_place=0):
        super(TransformerModel, self).__init__()
        encoder_layers = GeoTransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_hid,
                                                    dropout=dropout, batch_first=True, activation=activation,
                                                    geodesic=num_geodesic)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Linear(n_channels_input, d_model)
        self.decoder = nn.Linear(d_model, n_channels_output)
        self.n_head = n_head
        self.num_head_mask = num_head_mask
        self.num_layer_mask = num_layer_mask
        self.num_geodesic = num_geodesic
        self.in_place = in_place

        if predict_g_velo:
            if n_channels_velo:
                self.encoder_g_velo = nn.Linear(n_channels_velo, d_model)
            else:
                self.encoder_g_velo = nn.Parameter(torch.randn(1, d_model))
            self.n_channels_velo = n_channels_velo
            self.decoder_g_velo = nn.Linear(d_model, 3)

        self.d_model = d_model
        self.noise_level = noise_level

    def add_noise(self, x):
        return x + torch.randn_like(x) * self.noise_level if x is not None else None

    def forward(self, in_dict, requires_attn=False):
        """
        :param src: (batch_size, n_faces, n_channels_input)
        :param pe_f: (batch_size, n_faces, n_dim_pe)
        :param g_velo: (batch_size, n_channels_velo)
        :return: output: (batch_size, n_faces, n_channels_output)
        """

        src = in_dict['f']
        g_velo = in_dict['g_velo'] if 'g_velo' in in_dict else None
        geodesic = in_dict['geodesic'] if 'geodesic' in in_dict else None

        if self.training:
            src = self.add_noise(src)
            g_velo = self.add_noise(g_velo)

        src = self.encoder(src.float()) * math.sqrt(self.d_model)
        if g_velo is not None:
            if self.n_channels_velo > 0:
                g_velo = self.encoder_g_velo(g_velo) * math.sqrt(self.d_model)
            else:
                g_velo = self.encoder_g_velo.expand(src.shape[0], -1) * math.sqrt(self.d_model)
            g_velo = g_velo.unsqueeze(1)
            src = torch.cat([src, g_velo], dim=1)

        output, attn = forward_transformer(self.transformer_encoder, src, requires_attn=requires_attn,
                                           geodesic=geodesic, num_geodesic=self.num_geodesic, in_place=self.in_place)

        output_dict = {}

        if g_velo is not None:
            g_velo = output[:, -1, :]
            output = output[:, :-1, :]
            g_velo = self.decoder_g_velo(g_velo)
            output_dict['g_velo'] = g_velo

        output = self.decoder(output)
        output_dict['f'] = output

        if requires_attn:
            return output_dict, attn
        else:
            return output_dict

    def load_from_prefix(self, save_path, epoch=None, load_optimizer=False):
        if epoch is None or epoch == -1:
            checkpoints = [f for f in os.listdir(save_path) if f.endswith('.pth') and 'transformer' in f]
            checkpoints = sorted(checkpoints, key=checkpoint_sort_key)
            checkpoint_id = '_'.join(checkpoints[-1][:-4].split('_')[1:3])
        else:
            checkpoint_id = epoch

        checkpoint_filename = f'transformer_{checkpoint_id}.pth'

        state_dict = torch.load(osp.join(save_path, checkpoint_filename), map_location=self.encoder.weight.device)
        if load_optimizer:
            state_dict_optimizer = torch.load(osp.join(save_path, f'optimizer_{checkpoint_id}.pth'),
                                              map_location=self.encoder.weight.device)
            self.state_dict_optimizer = state_dict_optimizer
        else:
            self.state_dict_optimizer = None

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        # load params
        self.load_state_dict(new_state_dict)
        print(f'Loaded {checkpoint_filename}')
        return checkpoint_filename


def forward_transformer(model: TransformerEncoder, embedding, requires_attn=False, mask=None, num_layer_mask=-1, geodesic=None, num_geodesic=2, in_place=0):
    ## non-geodesic case
    if num_layer_mask == -1:
        num_layer_mask = model.num_layers
    if not requires_attn and num_layer_mask >= model.num_layers:
        if geodesic is not None:
            set_geodesic(model, geodesic)
        return model(src=embedding, mask=mask), None
    else:
        attns = []
        # gt = model(src=embedding, mask=mask, geodesic=geodesic, num_geodesic=num_geodesic, in_place=in_place)
        x = embedding
        for i, layer in enumerate(model.layers):
            mask_to_use = mask if i < num_layer_mask else None
            res, attn = layer.self_attn(x, x, x, attn_mask=mask_to_use, need_weights=requires_attn,
                                        average_attn_weights=False, geodesic=geodesic, num_geodesic=num_geodesic,
                                        in_place=in_place)
            if requires_attn:
                attns.append(attn.detach().cpu())
                print("attn shape", attn.shape)

            # attn is of shape (batch_size, n_head, n_faces, n_faces) if average_attn_weights is False
            res = layer.dropout(res)
            x = layer.norm1(x + res)
            x = layer.norm2(x + layer._ff_block(x))
        # assert torch.allclose(gt, x)
        if requires_attn:
            print(len(attns))
            print(attn[0].shape)
            return x, torch.stack(attns, dim=0)
        else:
            print('no attn')
            print("x shape: ", x.shape)
            return x, None
