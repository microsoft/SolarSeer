# SolarSeerNet consists of AFNO block and Swin Transformer block.
# AFNO block is adopted from FourCastNet. 
# Swin Transformer block is adopted from Swin Transformer V2. 

# FouurCastNet
# Copyright (c) 2022, FourCastNet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The code was authored by the following people:
#
# Jaideep Pathak - NVIDIA Corporation
# Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
# Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
# Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory
# Ashesh Chattopadhyay - Rice University
# Morteza Mardani - NVIDIA Corporation
# Thorsten Kurth - NVIDIA Corporation
# David Hall - NVIDIA Corporation
# Zongyi Li - California Institute of Technology, NVIDIA Corporation
# Kamyar Azizzadenesheli - Purdue University
# Pedram Hassanzadeh - Rice University
# Karthik Kashinath - NVIDIA Corporation
# Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation


# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import numpy as np
from icecream import ic
import argparse
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import torch.fft
from einops import rearrange
from torchvision.transforms import CenterCrop


#----   Cloud Block---------- 
def add(x_list):
    for _x in x_list[1:]:
        x_list[0] += _x
    return x_list[0]


def calculate_original_values(min_orig, max_orig, num_classes):
    interval_width = (max_orig - min_orig) / num_classes
    original_values = torch.linspace(min_orig + interval_width/2, max_orig - interval_width/2, num_classes)
    return original_values


def process_input(inputs, func, params):
    return func(inputs, **params)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01,
                 hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, \
            f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size,
                                                        self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks,
                                                        self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor,
                                                        self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor],
                              device=x.device)
        o1_imag = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor],
                              device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real,
                         self.w1[0]) -
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag,
                         self.w1[1]) +
            self.b1[0]
        )

        o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag,
                         self.w1[0]) +
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real,
                         self.w1[1]) +
            self.b1[1]
        )

        o2_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes],
                         self.w2[0]) -
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes],
                         self.w2[1]) +
            self.b2[0]
        )

        o2_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes],
                         self.w2[0]) +
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes],
                         self.w2[1]) +
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, H, W // 2 + 1, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.type(dtype)

        return x + bias


class Block(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            double_skip=True,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AFNO2D(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size + \
            ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
  
    
class PeriodicPad2d(nn.Module):
    """
        pad longitudinal (left-right) circular
        and pad latitude (top-bottom) with zeros
    """
    def __init__(self, pad_width):
        super(PeriodicPad2d, self).__init__()
        self.pad_width = pad_width

    def forward(self, x):
        # pad left and right circular
        out = F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode="circular")
        # pad top and bottom zeros
        out = F.pad(out, (0, 0, self.pad_width, self.pad_width), mode="constant", value=0)
        return out


def load_backbone_weight(backbone, weight_path, fix_param=True):
    backbone_weight = torch.load(weight_path)
    backbone.load_state_dict(backbone_weight['module'], strict=True)
    if fix_param:
        for param in backbone.parameters():
            param.requires_grad = False
    return backbone


class AFNONet(nn.Module):
    def __init__(
            self,
            params,
            img_size=(720, 1440),
            patch_size=(16, 16),
            in_chans=2,
            out_chans=2,
            input_time_dim=None,
            output_time_dim=None,
            embed_dim=768,
            depth=12,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            num_blocks=16,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
            autoregressive_steps=1,
            use_dilated_conv_blocks=False,
            output_only_last=False,
            target_variable_index=None,
            **kwargs
    ):
        super().__init__()
        self.params = params
        self.img_size = img_size
        self.patch_size = (params.get('patch_size', patch_size[0]), params.get('patch_size', patch_size[1]))
        self.in_chans = params.get('N_in_channels', in_chans)
        self.out_chans = params.get('N_out_channels', out_chans)
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim if output_time_dim is not None else input_time_dim
        self.has_time_dim = input_time_dim is not None
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.use_dilated_conv_blocks = use_dilated_conv_blocks
        self.autoregressive_steps = autoregressive_steps
        self.output_only_last = output_only_last
        self.target_variable_index = target_variable_index
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if self.has_time_dim:
            assert embed_dim % self.input_time_dim == 0, 'embed_dim must be divisible by input_time_dim'
            assert embed_dim % self.output_time_dim == 0, 'embed_dim must be divisible by output_time_dim'
            input_patch_embed_dim = embed_dim // self.input_time_dim
            self.output_patch_embed_dim = embed_dim // self.output_time_dim
        else:
            input_patch_embed_dim = embed_dim
            self.output_patch_embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=self.patch_size,
                                      in_chans=self.in_chans, embed_dim=input_patch_embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, input_patch_embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                  num_blocks=self.num_blocks, sparsity_threshold=sparsity_threshold,
                  hard_thresholding_fraction=hard_thresholding_fraction)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)  
        self.head = nn.Linear(self.output_patch_embed_dim,
                              self.out_chans*self.patch_size[0]*self.patch_size[1], bias=False)

        if self.use_dilated_conv_blocks:
            self.crop_layer = CenterCrop(params['target_size'])

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        if self.has_time_dim:
            x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if self.has_time_dim:
            x = rearrange(x, '(b t) (h w) c -> b h w (c t)', h=self.h, w=self.w, t=self.input_time_dim)
        else:
            x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        if self.has_time_dim:
            x = rearrange(x, 'b h w (c t) -> b t h w c', t=self.output_time_dim)
        return x

    def forward_head(self, x):
        x = self.head(x)
        x = rearrange(
            x,
            "b t h w (p1 p2 c_out) -> b c_out t (h p1) (w p2)" if self.has_time_dim else
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )
        return x

    def forward_step(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def get_next_input(self, inputs, outputs):
        if not self.has_time_dim:
            return outputs[-1]
        elif self.input_time_dim <= self.output_time_dim:
            return outputs[-1][:, :, -self.input_time_dim:]
        else:
            return torch.cat([inputs[:, :, self.output_time_dim:], outputs[-1]], dim=2)

    def forward(self, inputs, decoder_inputs=None):
        output_list = []
        for step in range(self.autoregressive_steps):
            if step > 0:
                inputs = self.get_next_input(inputs, output_list)
            x = self.forward_step(inputs)
            output_list.append(x)
        # For model without time dimension, ignore output_only_last and return the output of the last step
        if self.has_time_dim and not self.output_only_last:
            x = torch.cat(output_list, dim=2)
        else:
            x = output_list[-1]
        if self.use_dilated_conv_blocks:
            x = self.crop_layer(x)
        if self.target_variable_index is not None:
            x = x[:, self.target_variable_index]
        return x


class AFNONetOneStep(AFNONet):
    def __init__(
            self,
            **kwargs
    ):
        super(AFNONetOneStep, self).__init__(**kwargs)

    def forward(self, x):
        x = self.forward_step(x)
        return x


class EncoderAFNONet(AFNONet):
    def __init__(
            self,
            **kwargs
    ):
        super(EncoderAFNONet, self).__init__(**kwargs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class MultiEncoderAFNONet(nn.Module):
    def __init__(
            self,
            multi_params,
            **kwargs
    ):
        super().__init__()
        len_encoder = len(multi_params)
        multi_encoder_params_list = multi_params
        self.encoders = []
        self.output_patch_embed_list = []
        for i in range(len_encoder):
            hard_threshold = multi_encoder_params_list[i]['hard_thresholding_fraction']
            module = EncoderAFNONet(params=multi_encoder_params_list[i],
                                    img_size=multi_encoder_params_list[i]['img_size'],
                                    patch_size=(multi_encoder_params_list[i]['patch_size'],
                                                multi_encoder_params_list[i]['patch_size']),
                                    in_chans=multi_encoder_params_list[i]['N_in_channels'],
                                    out_chans=multi_encoder_params_list[i]['N_out_channels'],
                                    input_time_dim=multi_encoder_params_list[i]['input_time_dim'],
                                    output_time_dim=multi_encoder_params_list[i]['output_time_dim'],
                                    embed_dim=multi_encoder_params_list[i]['embed_dim'],
                                    depth=multi_encoder_params_list[i]['depth'],
                                    mlp_ratio=multi_encoder_params_list[i]['mlp_ratio'],
                                    drop_rate=multi_encoder_params_list[i]['drop_rate'],
                                    drop_path_rate=multi_encoder_params_list[i]['drop_path_rate'],
                                    num_blocks=multi_encoder_params_list[i]['num_blocks'],
                                    sparsity_threshold=multi_encoder_params_list[i]['sparsity_threshold'],
                                    hard_thresholding_fraction=hard_threshold,
                                    autoregressive_steps=multi_encoder_params_list[i]['autoregressive_steps'],
                                    use_dilated_conv_blocks=multi_encoder_params_list[i]['use_dilated_conv_blocks'],
                                    output_only_last=multi_encoder_params_list[i]['output_only_last'],
                                    target_variable_index=multi_encoder_params_list[i]['target_variable_index'])
            self.encoders.append(module)
            self.output_patch_embed_list.append(self.encoders[-1].output_patch_embed_dim)

        self.crop_layer = self.encoders[0].crop_layer if kwargs['use_dilated_conv_blocks'] else None
        self.out_chans = self.encoders[0].out_chans
        self.encoders = nn.ModuleList(self.encoders)
        self.autoregressive_steps = kwargs['autoregressive_steps']
        self.patch_size = (multi_encoder_params_list[0]['patch_size'], multi_encoder_params_list[0]['patch_size'])
        self.embed_dim = multi_encoder_params_list[0]['embed_dim']
        self.input_time_dim = multi_encoder_params_list[0]['input_time_dim']
        output_dim = multi_encoder_params_list[0]['output_time_dim']
        self.output_time_dim = self.input_time_dim if output_dim is None else output_dim
        self.has_time_dim = multi_encoder_params_list[0]['input_time_dim'] is not None
        self.output_only_last = multi_encoder_params_list[0]['output_only_last']
        self.target_variable_index = kwargs['target_variable_index']
        self.img_size = multi_encoder_params_list[0]['img_size']

        # add decoder
        self.action = kwargs['action']
        if self.action == 'concat':
            self.head = nn.Linear(sum(self.output_patch_embed_list),
                                  self.out_chans*self.patch_size[0]*self.patch_size[1], bias=False)
        else:
            self.head = nn.Linear(self.output_patch_embed_list[0],
                                  self.out_chans*self.patch_size[0]*self.patch_size[1], bias=False)

        self.apply(self._init_weights)
        self.act_final = kwargs.get('act_final', None)
        if self.act_final is not None:
            if self.act_final == 'Tanh':
                self.act = torch.nn.Tanh()
            elif self.act_final == 'ReLU':
                self.act = torch.nn.ReLU(inplace=True)
            elif self.act_final == 'LeakyReLU':
                self.act = torch.nn.LeakyReLU(0.2, True)
            elif self.act_final == 'ReLU6':
                self.act = torch.nn.ReLU6(inplace=True)
            elif self.act_final == 'Sigmoid':
                self.act = torch.nn.Sigmoid()
            else:
                raise ValueError(f'No such activation funtion.{self.act_final}')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_step(self, input_list):
        input_encoder_list = []
        for i, module in enumerate(self.encoders):
            x = module(input_list[i])
            input_encoder_list.append(x)
        if self.action == 'add':
            x = process_input(input_encoder_list, add, {})
        elif self.action == 'concat':
            x = process_input(input_encoder_list,  torch.concat, {'dim': -1})
        x = self.head(x)

        x = rearrange(
            x,
            "b t h w (p1 p2 c_out) -> b c_out t (h p1) (w p2)" if self.has_time_dim else
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )
        return x

    def get_next_input(self, inputs, outputs):
        if not self.has_time_dim:
            return outputs[-1]
        elif self.input_time_dim <= self.output_time_dim:
            return outputs[-1][:, :, -self.input_time_dim:]
        else:
            return torch.cat([inputs[:, :, self.output_time_dim:], outputs[-1]], dim=2)

    def forward(self, inputs, decoder_inputs = None):
        output_list = []
        for step in range(self.autoregressive_steps):
            if step > 0:
                inputs = self.get_next_input(inputs, output_list)
            x = self.forward_step([inputs])
            output_list.append(x)

        # For model without time dimension, ignore output_only_last and return the output of the last step
        if self.has_time_dim and not self.output_only_last:
            x = torch.cat(output_list, dim=2)
        else:
            x = output_list[-1]
        if self.crop_layer is not None:
            x = self.crop_layer(x)
        if self.target_variable_index is not None:
            x = x[:, self.target_variable_index]

        # act the output
        if self.act_final is not None:
            x = self.act(x) / 6 if self.act_final == 'ReLU6' else self.act(x)

        return x
    

#----   Irradiance Block---------- 

def backbone_load(ckpt_path, kwargs):
    backbone_weight = torch.load(ckpt_path)
    input_chans = backbone_weight['module']['patch_embed.proj.weight'].shape[1]
    num = kwargs['params']['patch_size'] * kwargs['params']['patch_size']
    output_chans = backbone_weight['module']['head.weight'].shape[0] // num
    kwargs['params']['N_in_channels'] = input_chans
    kwargs['params']['N_out_channels'] = output_chans
    backbone = AFNONetOneStep(**kwargs)
    state = backbone_weight['module'].copy()
    for weight_name in state:
        if weight_name.startswith('backbone'):
            del backbone_weight['module'][weight_name]
    backbone.load_state_dict(backbone_weight['module'], strict=True)

    for param in backbone.parameters():
        param.requires_grad = False
    return backbone


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=(224,224), patch_size=(4,4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # ic(img_size[0], patch_size[0])
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

   
    
class MultiDecoderSwinNet(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    # modified from swin transformer v2
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        out_chans (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(
        self,
        img_size=(512, 1280), 
        patch_size=8, 
        in_chans=3, 
        out_chans=1,
        embed_dim=256, 
        depths=[8], 
        num_heads=[2],
        window_size=16, 
        mlp_ratio=4., 
        qkv_bias=True,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm, 
        ape=False, 
        patch_norm=True,
        use_checkpoint=False, 
        pretrained_window_sizes=[0, 0, 0, 0], 
        **kwargs
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=self.img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, 
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(
            self.num_features, 
            out_chans * patch_size * patch_size)
        # self.head = nn.Linear(self.num_features, out_chans) if out_chans > 0 else nn.Identity()

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward_features(self, x):
        
        x = self.patch_embed(x)
        residual = x

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C

        x = x + residual

        # [B, L, C] -> [B, D, h/2, w/2]
        x = rearrange(x, 'b (h w) d -> b d h w',
                      h=self.patches_resolution[0],
                      w=self.patches_resolution[1]) 
        return x

    def head(self, x):
        # Linear projection
        x = rearrange(x, 'b d h w -> b (h w) d',
                      h=self.img_size[0]//self.patch_size,
                      w=self.img_size[1]//self.patch_size,
                      )

        x = self.fc(x)

        x = rearrange(x, 'b (h w) (ph pw c) -> b c (h ph) (w pw)',
                      h=self.img_size[0]//self.patch_size,
                      w=self.img_size[1]//self.patch_size,
                      ph=self.patch_size,
                      pw=self.patch_size,
                      c=self.out_chans
                      )

        return x

    def forward(self, x):
        clearghi = x[:, 1, :, :]
        x = self.forward_features(x)
        x = torch.squeeze(self.head(x))
        x[clearghi==0] = 0
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.out_chans
        return flops


class SolarSeerNet(nn.Module):
    def __init__(self, multi_params, **kwargs):
        super().__init__()
        self.encoder = MultiEncoderAFNONet(multi_params, **kwargs)
        self.decoder = MultiDecoderSwinNet(
                                    img_size=(512, 1280), 
                                    patch_size=8, 
                                    in_chans=3, 
                                    out_chans=1,
                                    embed_dim=256, 
                                    depths=[8], 
                                    num_heads=[2],
                                    window_size=16, 
                                    mlp_ratio=4., 
                                    qkv_bias=True,
                                    drop_rate=0., 
                                    attn_drop_rate=0., 
                                    drop_path_rate=0.1,
                                    norm_layer=nn.LayerNorm, 
                                    ape=False, 
                                    patch_norm=True,
                                    use_checkpoint=False, 
                                    pretrained_window_sizes=[0, 0, 0, 0]
                                    )
    def forward(self, satellite, clearghi): 
        # satellite and clearghi has been normarlized      
        # encoder
        cloud = self.encoder(satellite)
        cloud = torch.squeeze((100 - 0) / 2 * (cloud + 1) + 0)
        cloud = cloud[:, 26:-6, 40:1190]
        
        # crop and padding
        lead_time = torch.arange(1, 25, device=cloud.device)
        lead_time = lead_time.repeat(1, cloud.shape[1] * cloud.shape[2])
        lead_time = lead_time.reshape(-1, cloud.shape[1], cloud.shape[2])        
        x = torch.cat((cloud[:, None, :, :]/100,
                       clearghi[:, None, :, :]/1000, 
                       lead_time[:, None, :, :] / 24), dim=1)
        x1 = torch.cat((torch.zeros((x.shape[0], 3, 16, 1150), device=x.device), 
                        x, 
                        torch.zeros((x.shape[0], 3, 16, 1150), device=x.device)), dim=2)
        x_padded = torch.cat((torch.zeros((x.shape[0], 3, 512, 65), device=x.device), 
                              x1, 
                              torch.zeros((x.shape[0], 3, 512, 65), device=x.device)), dim=3)
        
        # decoder
        ghi = self.decoder(x_padded)
        
        # crop and ouput        
        y = torch.cat((ghi[:, None, 16:496, 65:1150+65], cloud[:, None, :, :]/100), dim=1)
        return y
    
if __name__ == "__main__":
        multi_params = [{
        "img_size": [512, 1280],
        "embed_dim": 600,
        "depth": 4,
        "mlp_ratio": 4.0,
        "drop_rate": 0.01,
        "drop_path_rate": 0.01,
        "num_blocks": 6,
        "sparsity_threshold": 0.01,
        "hard_thresholding_fraction": 1.0,
        "input_time_dim": 6,
        "output_time_dim": 24,
        "autoregressive_steps": 1,
        "use_dilated_conv_blocks": True,
        "output_only_last": False,
        "patch_size": 4,
        "N_in_channels": 4,
        "N_out_channels": 1,
        "target_size": [512, 1280],
        "target_variable_index": None,
        "topo_index": None,
        "new_index": None
        }]
        model = SolarSeerNet(multi_params=multi_params,
                                act_final="Tanh",
                                use_dilated_conv_blocks=False,
                                autoregressive_steps=1,
                                target_variable_index=[0],
                                action="add")
        satellite = torch.rand(1, 4, 6, 512, 1280).type(torch.float32)
        clearghi = torch.rand(24, 480, 1150).type(torch.float32) 
        y = model(satellite, clearghi)
        y[:, 0, :, :] = y[:, 0, :, :] * 1000 # surface irradiance
        y[:, 1, :, :] = y[:, 1, :, :] * 100  # cloud cover
