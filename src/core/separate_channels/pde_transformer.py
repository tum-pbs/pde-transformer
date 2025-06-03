from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import torch.nn.functional as F
import torch.nn as nn
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings, TimestepEmbedding, Timesteps, LabelEmbedding
from diffusers.utils import BaseOutput
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.models.swinv2.modeling_swinv2 import Swinv2Attention, Swinv2DropPath, Swinv2Intermediate, Swinv2Output
import math
import collections.abc

import numpy as np
from timm.models.layers import trunc_normal_, DropPath
import torch
from transformers.pytorch_utils import meshgrid, find_pruneable_heads_and_indices, prune_linear_layer

from src.core.mixed_channels.udit import FinalLayer, precompute_freqs_cis_2d, apply_rotary_emb

from src.core.mixed_channels.pde_transformer import Mlp


# Copied from transformers.models.swin.modeling_swin.window_partition
def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    batch_size, num_channels, height, width, hidden_dim = input_feature.shape

    input_feature = input_feature.view(
        batch_size, num_channels, height // window_size, window_size, width // window_size, window_size, hidden_dim
    )
    windows = input_feature.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, num_channels, window_size, window_size, hidden_dim)

    return windows


# Copied from transformers.models.swin.modeling_swin.window_reverse
def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    hidden_dim = windows.shape[-1]
    num_channels = windows.shape[-4]
    windows = windows.view(-1, height // window_size, width // window_size, num_channels, window_size, window_size, hidden_dim)
    windows = windows.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(-1, num_channels, height, width, hidden_dim)
    return windows

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class MaskEmbedder(nn.Module):
    """
    Embeds mask tokens into vector representations.
    """

    def __init__(self, hidden_size: int, mask_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(mask_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, mask):
        return self.mlp(mask)

class PdeParameterEmbedder(nn.Module):
    """
    Embeds PDE parameters into vector representations.
    """

    def __init__(self, hidden_size: int, num_pde_parameters_total: int = 1000,
                 frequency_embedding_size: int = 256, dropout_prob: float = 0):
        super().__init__()

        self.dropout_prob = dropout_prob
        self.time_proj = Timesteps(num_channels=frequency_embedding_size, flip_sin_to_cos=True,
                                   downscale_freq_shift=1)
        self.class_embedder = LabelEmbedding(num_pde_parameters_total, hidden_size,
                                             dropout_prob=0)

        self.num_pde_parameters_total = num_pde_parameters_total

        self.mlp_combined = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        self.mlp_timestep = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def token_drop(self, timesteps, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = torch.tensor(force_drop_ids == 1)
        labels = torch.where(drop_ids, self.num_pde_parameters_total, labels)
        timesteps = torch.where(drop_ids, 0.0, timesteps)
        return labels, timesteps

    def forward(self, pde_parameters, pde_parameter_class, train, force_drop_ids=None):

        num_pde_parameters = pde_parameters.shape[1]
        pde_parameters = pde_parameters.view(-1)
        pde_parameter_class = pde_parameter_class.view(-1)

        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            pde_parameters_class, pde_parameters = self.token_drop(pde_parameter_class, pde_parameters,
                                                                   force_drop_ids)

        emb_scalar = self.time_proj(pde_parameters)
        emb_scalar = self.mlp_timestep(emb_scalar)
        emb_label = self.class_embedder(pde_parameter_class)

        emb = self.mlp_combined(emb_scalar + emb_label)

        emb = emb.view(-1, num_pde_parameters, emb.shape[-1])
        emb = emb.sum(dim=1)

        return emb

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class SimplePatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, patch_size=4, bias=True):
        super(SimplePatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])  # -> (B*C, T, H, W)
        x = self.proj(x)
        x = x.view(batch_size, -1, x.shape[1], x.shape[2], x.shape[3])
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c: int = 3, embed_dim: int = 48, patch_size: int = 4, overlap_size: int = 1,
                 bias:bool = False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=(patch_size+2*overlap_size, patch_size+2*overlap_size),
                              stride=patch_size, padding=overlap_size, bias=bias)

        self.patch_size = patch_size

    def forward(self, x, periodic_width: bool = False, periodic_height: bool = False):

        batch_size = x.shape[0]
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4]) # -> (B*C, T, H, W)

        if periodic_width:

            x1 = x[:, :, :, :self.patch_size]
            x2 = x[:, :, :, -self.patch_size:]
            x = torch.cat((x2, x, x1), dim=-1)

        if periodic_height:

            x1 = x[:, :, :self.patch_size, :]
            x2 = x[:, :, -self.patch_size:, :]
            x = torch.cat((x2, x, x1), dim=-2)

        x = self.proj(x)

        if periodic_width:

            x = x[:, :, :, 1:-1]

        if periodic_height:

            x = x[:, :, 1:-1, :]

        x = x.view(batch_size, -1, x.shape[1], x.shape[2], x.shape[3])

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class DownsampleV2(nn.Module):
    def __init__(self, n_feat):
        super(DownsampleV2, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class UpsampleV2(nn.Module):
    def __init__(self, n_feat):
        super(UpsampleV2, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
    def forward(self, x):
        return self.body(x)

class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embedding's dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None, norm_type="layer_norm", bias=True):
        super().__init__()
        if num_embeddings is not None:
            self.emb_impl = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        else:
            self.emb_impl = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 9 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        elif norm_type == "fp32_layer_norm":
            self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb_impl is not None:
            emb = self.emb_impl(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        (msa1_shift, msa1_scale, msa1_gate,
         msa2_shift, msa2_scale, msa2_gate,
         mlp_shift, mlp_scale, mlp_gate) = emb.chunk(9, dim=1)
        return (msa1_shift, msa1_scale, msa1_gate,
                msa2_shift, msa2_scale, msa2_gate,
                mlp_shift, mlp_scale, mlp_gate)

class TokenInitializer(nn.Module):
    """
    Carrier token Initializer based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """
    def __init__(self,
                 dim,
                 # input_resolution,
                 window_size):
                 # ct_size=1):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            window_size: window size.
            ct_size: spatial dimension of carrier token local window
        """
        super().__init__()

        self.window_size = window_size
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        to_global_feature = nn.Sequential()
        to_global_feature.add_module("pos", self.pos_embed)

        # to_global_feature.add_module("pool", nn.AvgPool2d(kernel_size=(window_size, window_size),
        #                                                   stride=(window_size, window_size)))

        self.to_global_feature = to_global_feature
        self.window_size = window_size

    def forward(self, x):

        B, C, H, W, D = x.shape
        x = x.permute(0, 1, 4, 2, 3)
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])

        x = self.to_global_feature(x)

        pad_right = (self.window_size - W % self.window_size) % self.window_size
        pad_bottom = (self.window_size - H % self.window_size) % self.window_size

        x = F.pad(x, (0, pad_right, 0, pad_bottom, 0, 0, 0, 0), mode='constant', value=0)

        x = torch.nn.functional.avg_pool2d(x, kernel_size=(self.window_size, self.window_size), stride=(self.window_size, self.window_size),
                                           divisor_override=(self.window_size - pad_right) * (self.window_size - pad_bottom), padding=0)

        x = x.permute(0, 2, 3, 1)

        x = x.view(B, C, H // self.window_size, W // self.window_size, D)

        return x

class PDEStage(nn.Module):
    def __init__(
        self, dim: int, depth: int,
            num_heads: int, window_size: int, drop_path, mlp_ratio: float = 4.0, apply_shifts: bool = True,
            qkv_bias=True, periodic=False, use_carrier_tokens: bool = True,
    ):
        super().__init__()

        self.dim = dim
        blocks = []
        for i in range(depth):
            block = HATDiTBlock(
                dim=dim,
                num_heads=num_heads,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                window_size=window_size,
                periodic=periodic,
                use_carrier_tokens=use_carrier_tokens,
                mlp_ratio=mlp_ratio,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.periodic = periodic
        self.window_size = window_size

        self.shift_size = window_size // 2

        self.apply_shifts = apply_shifts

        self.carrier_token_active = use_carrier_tokens

        if self.carrier_token_active:
            self.global_tokenizer = TokenInitializer(dim,
                                                     window_size)


    def maybe_pad(self, hidden_states, height, width):
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def get_attn_mask(self, shift_size, height, width, dtype, device):
        if self.shift_size > 0 and not self.periodic:
            # calculate attention mask for shifted window multihead self attention
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype, device=device)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask.unsqueeze(1), self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def forward(self,
                hidden_states: torch.Tensor,
                cond: Optional[torch.Tensor] = None,
                timestep: Optional[torch.LongTensor] = None,
                class_labels: Optional[torch.LongTensor] = None, ):

        B, C, D, H, W = hidden_states.shape

        # precompute attention mask
        attn_mask_precomputed = self.get_attn_mask(self.window_size // 2, H, W, hidden_states.dtype,
                                                   hidden_states.device)

        for n, block in enumerate(self.blocks):

            shift_size = 0 if n % 2 == 0 else self.window_size // 2

            # hidden dim last
            hidden_states = torch.permute(hidden_states, (0, 1, 3, 4, 2))

            if shift_size > 0 and self.apply_shifts:
                attn_mask = attn_mask_precomputed
                shifted_hidden_states = torch.roll(hidden_states, shifts=(-shift_size, -shift_size),
                                                   dims=(2, 3))
            else:
                attn_mask = None
                shifted_hidden_states = hidden_states

            shifted_hidden_states, pad_values = self.maybe_pad(shifted_hidden_states, H, W)
            _, _, height_pad, width_pad, _ = shifted_hidden_states.shape

            if self.carrier_token_active:
                ct = self.global_tokenizer(hidden_states)
            else:
                ct = None

            hidden_states = window_partition(shifted_hidden_states, self.window_size)

            hidden_states2, ct = block(hidden_states, ct, timestep=timestep, class_labels=class_labels, emb=cond,
                                       attn_mask=attn_mask)

            hidden_states = window_reverse(hidden_states2, self.window_size, height_pad, width_pad)

            if height_pad > 0 or width_pad > 0:
                hidden_states = hidden_states[:, : , :H, :W, :].contiguous()

            if shift_size > 0 and self.apply_shifts:
                hidden_states = torch.roll(hidden_states, shifts=(shift_size, shift_size),
                                                   dims=(2, 3))

            hidden_states = torch.permute(hidden_states, (0, 1, 4, 2, 3))

        return hidden_states


class SwinDiTSelfOutput(nn.Module):
    def __init__(self, dim: int, attention_probs_dropout_prob: float = 0.0):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

class SwinDiTIntermediate(nn.Module):
    def __init__(self, dim: int, mlp_ratio = 4.0, hidden_act: str = "gelu"):
        super().__init__()
        self.dense = nn.Linear(dim, int(mlp_ratio * dim))
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class SwinDiTBlockOutput(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, hidden_dropout_prob: float = 0.0):
        super().__init__()
        self.dense = nn.Linear(int(mlp_ratio * dim), dim)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class PosEmbMLPSwinv2D(nn.Module):
    def __init__(self,
                 window_size: list[int],
                 pretrained_window_size: list[int],
                 num_heads: int,
                 no_log=False):
        super().__init__()

        self.window_size = [int(ws) for ws in window_size]
        self.num_heads = num_heads

        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

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

        if not no_log:
            relative_coords_table *= 8  # normalize to -8, 8
            relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).int()

        self.register_buffer("relative_position_index", relative_position_index)

        self.pos_emb = None


    def forward(self, input_tensor, local_window_size):

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
            -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        n_global_feature = input_tensor.shape[2] - local_window_size

        relative_position_bias = torch.nn.functional.pad(relative_position_bias, (n_global_feature,
                                                                                  0,
                                                                                  n_global_feature,
                                                                                  0)).contiguous()

        self.pos_emb = relative_position_bias.unsqueeze(0)

        input_tensor += self.pos_emb
        return input_tensor

class SelfAttention(nn.Module):


    def __init__(self, dim, num_heads,
                 bias=False, posemb_type='rope2d', attn_type='v2', down_shortcut=False, **kwargs):

        super(SelfAttention, self).__init__()
        if kwargs != dict():  # is not empty
            print(f'Kwargs: {kwargs}')

        self.dim = dim
        self.heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=bias)

        # self.to_out = nn.Conv2d(dim, dim, 1)

        # v2
        self.attn_type = attn_type
        if attn_type == 'v2':
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)


        # downsample
        self.dh = 1
        self.dw = 1

        # posemb
        self.posemb_type = posemb_type

        # posemb type
        if self.posemb_type == 'rope2d':
            self.freqs_cis = None

    def forward(self, x):

        # b, h, w, c = x.size()

        # x = rearrange(x, 'b h w c -> b 1 (h w) c')

        b, d, n, c = x.size()
        # x = torch.unsqueeze(x, 1)

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if self.posemb_type == 'rope2d':

            if self.freqs_cis is None or self.freqs_cis.shape[0] != n:

                self.freqs_cis = precompute_freqs_cis_2d(self.dim // self.heads, n).to(x.device)

            # q, k input shape: B N H Hc
            q, k = map(lambda t: rearrange(t, 'b p n (h d) -> (b p) n h d', h=self.heads), qkv[:-1])

            v = rearrange(qkv[2], 'b p n (h d) -> b p h n d', h=self.heads)

            q, k = apply_rotary_emb(q, k, freqs_cis=self.freqs_cis)
            # reshape back

            q = rearrange(q, '(b p) n h d -> b p h n d', b=b)
            k = rearrange(k, '(b p) n h d -> b p h n d', b=b)

        else:

            q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        if self.attn_type is None:  # v1 attention

            attn = (q @ k.transpose(-2, -1))
            attn = attn * self.scale

        elif self.attn_type == 'v2':  # v2 attention

            attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
            logit_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
            attn = attn * logit_scale

        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        # upsample
        # x = rearrange(x, 'b (dh dw) he (h w) d -> b (h dh) (w dw) (he d)',
        #               h=h // self.dh,
        #               w=w // self.dw,
        #               dh=self.dh,
        #               dw=self.dw)

        x = rearrange(x, 'b p h n d -> b p n (h d)')

        # x = self.to_out(x)

        # x = x[:, 0]

        return x

class WindowAttention2DTime(nn.Module):
    """
    Window attention based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        resolution: int = 0,
        attn_type='v2',
    ):
        super().__init__()
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
            resolution: feature resolution.
            seq_length: sequence length.
        """
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

        # attention positional bias
        self.pos_emb_funct = PosEmbMLPSwinv2D(
            window_size=[resolution, resolution],
            pretrained_window_size=[resolution, resolution],
            num_heads=num_heads,
        )
        self.attn_type = attn_type

        if attn_type == 'v2':
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        self.resolution = resolution

    def forward(self, x, attn_mask=None):
        B, D, N, C = x.shape

        x = x.view(B*D, N, C)

        qkv = (
            self.qkv(x)
            .reshape(B*D, -1, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.attn_type == 'v1':
            attn = (q @ k.transpose(-2, -1)) * self.scale

        elif self.attn_type == 'v2':
            attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
            logit_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
            attn = attn * logit_scale

        attn = self.pos_emb_funct(attn, self.resolution ** 2)

        if attn_mask is not None:
            # Apply the attention mask is (precomputed for all layers in PDE forward() function)
            mask_shape = attn_mask.shape[0]
            attn = attn.view(
                (B*D) // mask_shape, mask_shape, self.num_heads, N, N
            ) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B*D, -1, C)

        x = x.view(B, D, N, C)

        return x

def ct_window_simple(ct, W, H, window_size):
    bs = ct.shape[0]
    N = ct.shape[2]
    ct = ct.view(
        bs,
        H,
        W,
        N,
    )
    return ct

def ct_dewindow_simple(ct, W, H, window_size):
    bs = ct.shape[0]
    N = ct.shape[2]
    ct2 = ct.view(
        -1,
        H,
        W,
        N,
    ).permute(0, 3, 1, 2)
    ct2 = ct2.reshape(bs, N, W * H).transpose(1, 2)
    return ct2

def ct_window(ct, W, H, window_size):
    bs = ct.shape[0]
    N = ct.shape[2]
    ct = ct.view(
        bs,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        N,
    )
    ct = ct.permute(0, 1, 3, 2, 4, 5)
    return ct

def ct_dewindow(ct, W, H, window_size):
    bs = ct.shape[0]
    N = ct.shape[2]
    ct2 = ct.view(
        -1,
        W // window_size,
        H // window_size,
        window_size,
        window_size,
        N,
    ).permute(0, 5, 1, 3, 2, 4)
    ct2 = ct2.reshape(bs, N, W * H).transpose(1, 2)
    return ct2

class PosEmbMLPSwinv1D(nn.Module):
    def __init__(self,
                 dim,
                 rank=2,
                 conv=False):
        super().__init__()
        self.rank = rank
        if not conv:
            self.cpb_mlp = nn.Sequential(nn.Linear(self.rank, 512, bias=True),
                                         nn.ReLU(),
                                         nn.Linear(512, dim, bias=False))
        else:
            self.cpb_mlp = nn.Sequential(nn.Conv1d(self.rank, 512, 1,bias=True),
                                         nn.ReLU(),
                                         nn.Conv1d(512, dim, 1,bias=False))
        self.grid_exists = False
        self.pos_emb = None

        self.conv = conv

    def forward(self, input_tensor):

        if self.rank == 1:

            seq_length = input_tensor.shape[1] if not self.conv else input_tensor.shape[2]

            relative_coords_h = torch.arange(0, seq_length, device=input_tensor.device, dtype = input_tensor.dtype)
            relative_coords_h -= seq_length//2
            relative_coords_h /= (seq_length//2)
            relative_coords_table = relative_coords_h
            self.pos_emb = self.cpb_mlp(relative_coords_table.unsqueeze(0).unsqueeze(2))

        else:

            height = input_tensor.shape[1]
            width = input_tensor.shape[2]

            relative_coords_h = torch.arange(0, height, device=input_tensor.device, dtype = input_tensor.dtype)
            relative_coords_w = torch.arange(0, width, device=input_tensor.device, dtype = input_tensor.dtype)
            relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w])).contiguous().unsqueeze(0)
            relative_coords_table[:,0] -= height // 2
            relative_coords_table[:,1] -= width // 2
            relative_coords_table[:,0] /= max((height // 2), 1.0) # special case for 1x1
            relative_coords_table[:,1] /= max((width // 2), 1.0) # special case for 1x1
            if not self.conv:
                self.pos_emb = self.cpb_mlp(relative_coords_table.permute(0, 2, 3, 1))
            else:
                self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2))

        input_tensor = input_tensor + self.pos_emb
        return input_tensor

class HATDiTBlock(nn.Module):
    """
    Hierarchical attention (HAT) based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention

    Modifications:
        - 2D + time
        - AdaIN for diffusion conditioning
        - Conditioning via context tokens
        - Sliding window
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        num_embeds_ada_norm: Optional[int] = 1000,
        window_size=7,
        last=False,
        layer_scale=None,
        ct_size=1,
        do_propagation=False,
        use_carrier_tokens=True,
        shift_size=0,
        periodic=False,
    ):
        super().__init__()
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            mlp_ratio: MLP ratio.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
            act_layer: activation function.
            norm_layer: normalization layer.
            num_embeds_ada_norm: number of embeddings for AdaLayerNormZero.
            sr_ratio: input to window size ratio.
            window_size: window size.
            last: last layer flag.
            layer_scale: layer scale coefficient.
            ct_size: spatial dimension of carrier token local window.
            do_propagation: enable carrier token propagation.
        """

        # positional encoding for windowed attention tokens
        # self.pos_embed = PosEmbMLPSwinv1D(dim, rank=2)

        self.carrier_token_active = use_carrier_tokens

        self.cr_window = 1

        self.attn_spatial = WindowAttention2DTime(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            resolution=window_size,
        )

        self.attn_channel = SelfAttention(
            dim=dim,
            num_heads=num_heads,
            posemb_type='none',
        )

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.window_size = window_size

        self.adain_2 = AdaLayerNormZero(dim, num_embeddings=None, norm_type="layer_norm")

        if self.carrier_token_active:

            self.hat_norm1 = norm_layer(dim)
            self.hat_norm2 = norm_layer(dim)
            self.hat_norm3 = norm_layer(dim)

            self.hat_attn_spatial = SelfAttention(
                dim=dim,
                num_heads=num_heads,
                posemb_type='rope2d',
            )

            self.hat_attn_channel = SelfAttention(
                dim=dim,
                num_heads=num_heads,
                posemb_type='none',
            )

            self.hat_mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )

            self.hat_drop_path = (
                DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            )

            self.hat_pos_embed = PosEmbMLPSwinv1D(
                dim, rank=2,
            )

            self.adain_1 = AdaLayerNormZero(dim, num_embeddings=None, norm_type="layer_norm")

            self.upsampler = nn.Upsample(size=window_size, mode="nearest")

        # keep track for the last block to explicitly add carrier tokens to feature maps
        self.last = last
        self.do_propagation = do_propagation

        # layer_norm_eps = 1e-5
        # self.layernorm_before = nn.LayerNorm(dim, eps=layer_norm_eps)
        # self.intermediate = SwinDiTIntermediate(dim)
        # self.output = SwinDiTBlockOutput(dim)
        # self.layernorm_after = nn.LayerNorm(dim, eps=layer_norm_eps)

    def forward(self, x, carrier_tokens,
                timestep: Optional[torch.LongTensor] = None,
                class_labels: Optional[torch.LongTensor] = None,
                emb: Optional[torch.LongTensor] = None,
                attn_mask: Optional[torch.Tensor] = None):

        B, C, H, W, N = x.shape

        Bc = emb.shape[0]

        emb = emb.view(-1, emb.shape[-1])

        ct = carrier_tokens

        # Do we need pos_embed here?
        # x = self.pos_embed(x)

        x = x.view(B, C, H * W, N)

        if self.carrier_token_active:

            Bc, Dc, Hc, Wc, Nc = ct.shape

            # positional bias for carrier tokens
            ct = ct.view(Bc*Dc, Hc, Wc, Nc)
            ct = self.hat_pos_embed(ct)
            ct = ct.view(Bc, Dc, Hc * Wc, Nc)

            # ct = ct.reshape(Bc, Hc * Wc, Nc)

            ######## DiT block with MSA, MLP, and AdaIN ########

            (msa1_shift, msa1_scale, msa1_gate,
             msa2_shift, msa2_scale, msa2_gate,
             mlp_shift, mlp_scale, mlp_gate) = self.adain_1(timestep=timestep,
                                                            class_labels=class_labels,
                                                            emb=emb)

            ##################### Spatial Attention ################

            msa1_shift = msa1_shift.view(Bc, Dc, 1, -1)
            msa1_scale = msa1_scale.view(Bc, Dc, 1, -1)
            msa1_gate = msa1_gate.view(Bc, Dc, 1, -1)

            # attention plus mlp
            ct_msa = self.hat_norm1(ct)
            ct_msa = ct_msa * (1 + msa1_scale) + msa1_shift

            ct_msa = self.hat_attn_spatial(ct_msa)

            ct_msa = ct_msa * (1 + msa1_gate)
            ct = ct + self.hat_drop_path(ct_msa)

            #########################################################

            ###################### Channel Attention ################

            msa2_shift = msa2_shift.view(Bc, Dc, 1, -1)
            msa2_scale = msa2_scale.view(Bc, Dc, 1, -1)
            msa2_gate = msa2_gate.view(Bc, Dc, 1, -1)

            # attention plus mlp
            ct_msa = self.hat_norm2(ct)
            ct_msa = ct_msa * (1 + msa2_scale) + msa2_shift

            ct_msa = torch.permute(ct_msa, (0, 2, 1, 3))
            ct_msa = self.hat_attn_channel(ct_msa)
            ct_msa = torch.permute(ct_msa, (0, 2, 1, 3))

            ct_msa = ct_msa * (1 + msa2_gate)
            ct = ct + self.hat_drop_path(ct_msa)

            ###########################################################

            ####################### MLP ###############################

            mlp_shift = mlp_shift.view(Bc, Dc, 1, -1)
            mlp_scale = mlp_scale.view(Bc, Dc, 1, -1)
            mlp_gate = mlp_gate.view(Bc, Dc, 1, -1)

            ct_mlp = self.hat_norm3(ct)
            ct_mlp = ct_mlp * (1 + mlp_scale) + mlp_shift
            ct_mlp = self.hat_mlp(ct_mlp)
            ct_mlp = ct_mlp * (1 + mlp_gate)

            ct = ct + self.hat_drop_path(ct_mlp)

            ###########################################################

            ct = ct.permute(0, 2, 1, 3).reshape(Bc * Hc * Wc, Dc, 1, Nc)

            # ct = ct.reshape(x.shape[0], -1, N)
            # concatenate carrier_tokens to the windowed tokens
            x = torch.cat((ct, x), dim=2)


        ########### DiT block with MSA, MLP, and AdaIN ############
        (msa1_shift, msa1_scale, msa1_gate,
         msa2_shift, msa2_scale, msa2_gate,
         mlp_shift, mlp_scale, mlp_gate) = self.adain_2(timestep=timestep,
                                                        class_labels=class_labels,
                                                        emb=emb)

        num_windows_total = int(B // Bc)

        msa1_shift = msa1_shift.view(Bc, C, 1, -1)
        msa1_scale = msa1_scale.view(Bc, C, 1, -1)
        msa1_gate = msa1_gate.view(Bc, C, 1, -1)
        msa2_shift = msa2_shift.view(Bc, C, 1, -1)
        msa2_scale = msa2_scale.view(Bc, C, 1, -1)
        msa2_gate = msa2_gate.view(Bc, C, 1, -1)
        mlp_shift = mlp_shift.view(Bc, C, 1, -1)
        mlp_scale = mlp_scale.view(Bc, C, 1, -1)
        mlp_gate = mlp_gate.view(Bc, C, 1, -1)

        msa1_shift = msa1_shift.repeat_interleave(num_windows_total, dim=0)
        msa1_scale = msa1_scale.repeat_interleave(num_windows_total, dim=0)
        msa1_gate = msa1_gate.repeat_interleave(num_windows_total, dim=0)

        msa2_shift = msa2_shift.repeat_interleave(num_windows_total, dim=0)
        msa2_scale = msa2_scale.repeat_interleave(num_windows_total, dim=0)
        msa2_gate = msa2_gate.repeat_interleave(num_windows_total, dim=0)

        mlp_shift = mlp_shift.repeat_interleave(num_windows_total, dim=0)
        mlp_scale = mlp_scale.repeat_interleave(num_windows_total, dim=0)
        mlp_gate = mlp_gate.repeat_interleave(num_windows_total, dim=0)

        ######################## Spatial Attention ####################

        x_msa = self.norm1(x)
        x_msa = x_msa * (1 + msa1_scale) + msa1_shift
        x_msa = self.attn_spatial(x_msa, attn_mask=attn_mask)
        x_msa = x_msa * (1 + msa1_gate)

        x = x + self.drop_path(x_msa)

        #############################################################

        ###################### Channel Attention ####################

        x_msa = self.norm2(x)
        x_msa = x_msa * (1 + msa2_scale) + msa2_shift

        # Permute axes for channel attention
        x_msa = torch.permute(x_msa, (0, 2, 1, 3))
        x_msa = self.attn_channel(x_msa)
        x_msa = torch.permute(x_msa, (0, 2, 1, 3))

        x_msa = x_msa * (1 + msa2_gate)

        x = x + self.drop_path(x_msa)

        #############################################################

        ############################ MLP ############################

        x_mlp = self.norm3(x)

        x_mlp = x_mlp * (1 + mlp_scale) + mlp_shift
        x_mlp = self.mlp(x_mlp)
        x_mlp = x_mlp * (1 + mlp_gate)
        x = x + self.drop_path(x_mlp)

        #############################################################
        ##########  Split carrier tokens and normal tokens ##########

        if self.carrier_token_active:

            # for hierarchical attention we need to split carrier tokens and window tokens back
            ctr, x = x.split(
                [
                    x.shape[2] - self.window_size * self.window_size,
                    self.window_size * self.window_size,
                ],
                dim=2,
            )

            # reshape carrier tokens
            ct = ctr.reshape(Bc, Dc, Hc * Wc, Nc)  # noqa

            if self.last and self.do_propagation:
                # propagate carrier token information into the image
                ctr_image_space = ctr.transpose(1, 2).reshape(
                    B, N, self.cr_window, self.cr_window
                )
                x = x + self.gamma1 * self.upsampler(
                    ctr_image_space.to(dtype=torch.float32)
                ).flatten(2).transpose(1, 2).to(dtype=x.dtype)

        x = x.view(B, C, H, W, N)

        return x, ct


class PDEImpl(nn.Module):
    """
    Diffusion UNet model with a Transformer backbone.
    """

    def __init__(
            self,
            num_timesteps: int = 6,
            window_size: int = 8,
            patch_size: int = 4,
            overlap_size: int = 0,
            hidden_size: int = 1152,
            depth: Tuple[int] =(2, 4, 4, 6, 4, 4, 2),
            num_heads: int = 16,
            mlp_ratio: int = 4,
            class_dropout_prob = 0.1,
            num_classes: int = 1000,
            learn_sigma: bool = True,
            periodic: bool = False,
            use_carrier_tokens: bool = True,
            apply_shifts: bool = True,
            **kwargs
    ):
        super().__init__()

        assert len(depth) % 2 == 1, "Encoder and decoder depths must be equal."
        self.num_encoder_layers = len(depth) // 2

        self.learn_sigma = learn_sigma
        self.in_channels = num_timesteps
        self.out_channels = num_timesteps

        self.SIMULATION_TIME_SCALING = 0.0
        self.SIMULATION_DT_SCALING = 0.0
        self.PDE_PARAMETER_SCALING = 0.0

        self.num_heads = num_heads
        self.periodic = periodic

        dit_stage_args = {
            "drop_path": None,
            "periodic": periodic,
            'use_carrier_tokens': use_carrier_tokens,
            'mlp_ratio': mlp_ratio,
            'apply_shifts': apply_shifts,
        }

        self.use_carrier_tokens = use_carrier_tokens

        if patch_size is not None:
            self.x_embedder = SimplePatchEmbed(in_c=num_timesteps, embed_dim=hidden_size,
                                         patch_size=patch_size,
                                         bias=True)
            self.patch_size = patch_size
        else:
            self.x_embedder = OverlapPatchEmbed(in_c=num_timesteps, embed_dim=hidden_size,
                                        patch_size=1, overlap_size=overlap_size,
                                        bias=True)
            self.patch_size = 1

        # timestep and label embedders
        for i in range(self.num_encoder_layers + 1):
            self.__setattr__(f"t_embedder_{i}",
                             TimestepEmbedder(hidden_size * 2 ** i))
            self.__setattr__(f"simulation_time_embedder_{i}",
                             TimestepEmbedder(hidden_size * 2 ** i))
            self.__setattr__(f"simulation_dt_embedder_{i}",
                             TimestepEmbedder(hidden_size * 2 ** i))
            self.__setattr__(f"channel_type_embedder_{i}",
                             LabelEmbedder(num_classes, hidden_size * 2 ** i, class_dropout_prob))
            self.__setattr__(f"pde_type_embedder_{i}",
                             LabelEmbedder(num_classes, hidden_size * 2 ** i, class_dropout_prob))
            self.__setattr__(f"pde_parameter_embedder_{i}",
                             PdeParameterEmbedder(hidden_size * 2 ** i, num_pde_parameters_total=num_classes,
                                                                        dropout_prob=class_dropout_prob))
            self.__setattr__(f"task_embedder_{i}",
                             LabelEmbedder(num_classes, hidden_size * 2 ** i, dropout_prob=0.0))

        # encoder
        for i in range(self.num_encoder_layers):
            self.__setattr__(f"encoder_level_{i}", PDEStage(dim=hidden_size * 2 ** i, num_heads=num_heads,
                                            window_size=window_size, depth=depth[i], **dit_stage_args))
            self.__setattr__(f"down{i}_{i+1}", Downsample(hidden_size * 2 ** i))

        # latent
        self.latent = PDEStage(dim=hidden_size * 2 ** self.num_encoder_layers, num_heads=num_heads,
                                            window_size=window_size, depth=depth[self.num_encoder_layers], **dit_stage_args)


        # double hidden size for last decoder layer 0
        self.__setattr__("up1_0", Upsample(hidden_size * 2))
        self.__setattr__("reduce_chan_level0", nn.Conv2d(hidden_size * 2, hidden_size * 2, kernel_size=1, bias=True))
        self.__setattr__("decoder_level_0", PDEStage(dim=hidden_size * 2, num_heads=num_heads,
                                        window_size=window_size, depth=depth[self.num_encoder_layers + 1], **dit_stage_args))

        # decoder layers 1 - num_encoder_layers
        for i in range(1, self.num_encoder_layers):
            self.__setattr__(f"up{i+1}_{i}", Upsample(hidden_size * 2 ** (i+1)))
            self.__setattr__(f"reduce_chan_level{i}", nn.Conv2d(hidden_size * 2 ** (i+1), hidden_size * 2 ** i, kernel_size=1, bias=True))
            self.__setattr__(f"decoder_level_{i}", PDEStage(dim=hidden_size * 2 ** i, num_heads=num_heads,
                                            window_size=window_size, depth=depth[self.num_encoder_layers + i + 1], **dit_stage_args))

        self.output = nn.Conv2d(int(hidden_size * 2), int(hidden_size * 2), kernel_size=3, stride=1, padding=1,
                                bias=True)

        self.final_layer = FinalLayer(hidden_size * 2, self.out_channels * self.patch_size * self.patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        for i in range(self.num_encoder_layers):
            # Initialize label embedding table:
            nn.init.normal_(self.__getattr__(f"channel_type_embedder_{i}").embedding_table.weight, std=0.02)
            nn.init.normal_(self.__getattr__(f"pde_type_embedder_{i}").embedding_table.weight, std=0.02)
            nn.init.normal_(self.__getattr__(f"task_embedder_{i}").embedding_table.weight, std=0.02)

            # Initialize timestep embedding MLP:
            nn.init.normal_(self.__getattr__(f"t_embedder_{i}").mlp[0].weight, std=0.02)
            nn.init.normal_(self.__getattr__(f"t_embedder_{i}").mlp[2].weight, std=0.02)
            nn.init.normal_(self.__getattr__(f"simulation_time_embedder_{i}").mlp[0].weight, std=0.02)
            nn.init.normal_(self.__getattr__(f"simulation_time_embedder_{i}").mlp[2].weight, std=0.02)
            nn.init.normal_(self.__getattr__(f"simulation_dt_embedder_{i}").mlp[0].weight, std=0.02)
            nn.init.normal_(self.__getattr__(f"simulation_dt_embedder_{i}").mlp[2].weight, std=0.02)

            # Initialize PDE parameter embedding MLP:
            nn.init.normal_(self.__getattr__(f"pde_parameter_embedder_{i}").mlp_combined[0].weight, std=0.02)
            nn.init.normal_(self.__getattr__(f"pde_parameter_embedder_{i}").mlp_combined[2].weight, std=0.02)
            nn.init.normal_(self.__getattr__(f"pde_parameter_embedder_{i}").mlp_timestep[0].weight, std=0.02)
            nn.init.normal_(self.__getattr__(f"pde_parameter_embedder_{i}").mlp_timestep[2].weight, std=0.02)

        blocks = [self.__getattr__(f"encoder_level_{i}") for i in range(self.num_encoder_layers)]
        blocks += [self.latent]
        blocks += [self.__getattr__(f"decoder_level_{i}") for i in range(self.num_encoder_layers)]

        for block in blocks:

            for blc in block.blocks:

                nn.init.constant_(blc.adain_2.linear.weight, 0)
                nn.init.constant_(blc.adain_2.linear.bias, 0)

                if self.use_carrier_tokens:

                    nn.init.constant_(blc.adain_1.linear.weight, 0)
                    nn.init.constant_(blc.adain_1.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.out_proj.weight, 0)
        nn.init.constant_(self.final_layer.out_proj.bias, 0)

    def forward(self,
                x: list[torch.Tensor],  # list of tensors with shape (B, T, H, W)
                simulation_time: list[torch.Tensor], # list of tensor with shape B
                channel_type: list[torch.Tensor], # list of tensor with shape B
                pde_type: list[torch.Tensor], # list of tensor with shape B
                pde_parameters: list[torch.Tensor], # list of tensor with shape B, self.num_pde_params
                pde_parameters_class: list[torch.Tensor], # list of tensor with shape B
                simulation_dt: list[torch.Tensor], # list of tensor with shape B
                task: list[torch.Tensor], # list of tensors with shape B
                t: list[torch.Tensor],  # list of tensors with shape B
               ):
        """
        Forward pass of SwinDiT.
        x (torch.Tensor): input tensor of shape (B, C, T, H, W).
        simulation_time (torch.Tensor): simulation time tensor of shape (B, C).
        channel_type (torch.Tensor): channel type tensor of shape (B, C).
        diffusion_time (torch.Tensor): diffusion time tensor of shape (B,).
        pde_type (torch.Tensor): PDE type tensor of shape (B,).
        pde_parameters (torch.Tensor): PDE parameters tensor of shape (B, num_pde_params).
        pde_parameters_class (torch.Tensor): PDE parameters class tensor of shape (B,).
        simulation_dt (torch.Tensor): simulation time tensor of shape (B,).
        diffusion_mask (torch.Tensor): diffusion mask tensor of shape (B, C, D).
        """

        x = torch.stack(x, dim=1)
        t = torch.stack(t, dim=1)
        simulation_time = torch.stack(simulation_time, dim=1)
        channel_type = torch.stack(channel_type, dim=1)
        pde_type = torch.stack(pde_type, dim=1)
        pde_parameters = torch.stack(pde_parameters, dim=1)
        pde_parameters_class = torch.stack(pde_parameters_class, dim=1)
        simulation_dt = torch.stack(simulation_dt, dim=1)
        task = torch.stack(task, dim=1)

        x = self.x_embedder(x)
        # x = self.x_embedder(x, periodic_height=self.periodic, periodic_width=self.periodic)
        # -> (B, C, D, H // patch_size, W // patch_size)
        B, C, D, H, W = x.shape

        num_pde_parameters = pde_parameters.shape[2]

        emb_list = []
        for i in range(self.num_encoder_layers + 1):

            t = t.view(-1)
            simulation_dt = simulation_dt.view(-1)
            simulation_time = simulation_time.view(-1)

            t_emb = self.__getattr__(f"t_embedder_{i}")(t)
            simulation_time_emb = self.__getattr__(f"simulation_time_embedder_{i}")(simulation_time)
            simulation_dt_emb = self.__getattr__(f"simulation_dt_embedder_{i}")(simulation_dt)

            t_emb = t_emb.view((B, C) + t_emb.shape[1:])
            simulation_time_emb = simulation_time_emb.view((B, C) + simulation_time_emb.shape[1:])
            simulation_dt_emb = simulation_dt_emb.view((B, C) + simulation_dt_emb.shape[1:])

            pde_parameters = pde_parameters.view( (-1, num_pde_parameters))
            pde_parameters_class = pde_parameters_class.view( (-1, num_pde_parameters))
            pde_parameter_emb = self.__getattr__(f"pde_parameter_embedder_{i}")(pde_parameters,
                                                                                pde_parameters_class,
                                                                                self.training)

            pde_parameter_emb = pde_parameter_emb.view((B, C) + pde_parameter_emb.shape[1:])

            channel_type = channel_type.view(-1)
            channel_type_emb = self.__getattr__(f"channel_type_embedder_{i}")(channel_type, self.training)
            channel_type_emb = channel_type_emb.view((B, C) + channel_type_emb.shape[1:])

            pde_type = pde_type.view(-1)
            pde_type_emb = self.__getattr__(f"pde_type_embedder_{i}")(pde_type, self.training)
            pde_type_emb = pde_type_emb.view((B, C) + pde_type_emb.shape[1:])

            task = task.view(-1)
            task_emb = self.__getattr__(f"task_embedder_{i}")(task, self.training)
            task_emb = task_emb.view((B, C) + task_emb.shape[1:])

            c = (t_emb + self.SIMULATION_TIME_SCALING * simulation_time_emb + channel_type_emb +
                 pde_type_emb + self.SIMULATION_DT_SCALING * simulation_dt_emb + task_emb + self.PDE_PARAMETER_SCALING * pde_parameter_emb)

            emb_list.append(c)

        residuals_list = []

        for i, c in enumerate(emb_list[:-1]):
            # encoder
            out_enc_level = self.__getattr__(f"encoder_level_{i}")(x, c)
            residuals_list.append(out_enc_level)

            out_enc_level = out_enc_level.view((B*C,) + out_enc_level.shape[2:])
            out_enc_level = self.__getattr__(f"down{i}_{i+1}")(out_enc_level)
            x = out_enc_level.view((B, C) + out_enc_level.shape[1:])

        c = emb_list[-1]
        x = self.latent(x, c)

        for i, (residual, emb) in enumerate(zip(residuals_list[1:][::-1], emb_list[1:-1][::-1])):
            # decoder
            x = x.view((B*C,) + x.shape[2:])
            x = self.__getattr__(f"up{self.num_encoder_layers - i}_{self.num_encoder_layers - i - 1}")(x)
            x = x.view((B, C) + x.shape[1:])

            x = torch.cat([x, residual], 2)

            x = x.view((B*C,) + x.shape[2:])
            x = self.__getattr__(f"reduce_chan_level{self.num_encoder_layers - i - 1}")(x)
            x = x.view((B, C) + x.shape[1:])
            x = self.__getattr__(f"decoder_level_{self.num_encoder_layers - i - 1}")(x, emb)

        x = x.view((B*C,) + x.shape[2:])
        x = self.__getattr__(f"up1_0")(x)
        x = x.view((B, C) + x.shape[1:])

        x = torch.cat([x, residuals_list[0]], 2)

        x = x.view((B*C,) + x.shape[2:])
        x = self.__getattr__(f"reduce_chan_level0")(x)
        x = x.view((B, C) + x.shape[1:])

        x = self.__getattr__(f"decoder_level_0")(x, emb_list[1])
        x = x.view((B*C,) + x.shape[2:])

        # output
        x = self.output(x)

        emb_final = emb_list[1]
        emb_final = emb_final.view((B * C,) + emb_final.shape[2:])
        x = self.final_layer(x, emb_final)  # (B*C, T, patch_size ** 2 * out_channels)

        # unpatchify
        x = x.permute(0, 2, 3, 1)

        x = x.reshape(
            shape=x.shape[:3] + (self.patch_size, self.patch_size, self.out_channels)
        )

        height = x.shape[1]
        width = x.shape[2]

        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(
            shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        )

        x = x.view((B, C) + x.shape[1:])

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of IPT, but also batches the unconIPTional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

@dataclass
class SwinDiTOutput(BaseOutput):
    """
    The output of [`SwinDiT`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.Tensor

class PDETransformer(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
            self,
            sample_size: int,
            num_timesteps: int,
            type: str,
            periodic: bool = False,
            carrier_token_active: bool = True,
            patch_size: Optional[int] = None,
    ):
        super(PDETransformer, self).__init__()
        args = { 'num_timesteps': num_timesteps, 'patch_size': patch_size, 'learn_sigma': False,
                 'periodic': periodic, 'use_carrier_tokens': carrier_token_active}

        self.model: PDEImpl = PDE_models[type](**args)
        self.sample_size = sample_size
        self.in_channels = num_timesteps
        self.out_channels = num_timesteps
        self.patch_size = patch_size


    def forward(
            self,
            x: list[torch.Tensor],  # list of tensors with shape (B, T, H, W)
            simulation_time: list[torch.Tensor],  # list of tensor with shape B
            channel_type: list[torch.Tensor],  # list of tensor with shape B
            pde_type: list[torch.Tensor],  # list of tensor with shape B
            pde_parameters: list[torch.Tensor],  # list of tensor with shape B, num_pde_parameters
            pde_parameters_class: list[torch.Tensor],  # list of tensor with shape B
            simulation_dt: list[torch.Tensor],  # list of tensor with shape B
            task: list[torch.Tensor],  # list of tensors with shape B C
            t: list[torch.Tensor],  # list of tensors with shape B
            return_dict: bool = True,
    ):

        # SCALE PARAMETERS

        # timestep scaling (from [0,1] to [0,1000])
        t = [t_i * 1000 for t_i in t]

        # pred_parameters scaling
        pde_parameters = [pde_parameters_i * 1000 for pde_parameters_i in pde_parameters]

        # simulation_time scaling
        simulation_time = [simulation_time_i * 1000 for simulation_time_i in simulation_time]

        # simulation_dt scaling
        simulation_dt = [simulation_dt_i * 1000 for simulation_dt_i in simulation_dt]

        output = self.model.forward(
            x=x,
            simulation_time=simulation_time,
            channel_type=channel_type,
            pde_type=pde_type,
            pde_parameters=pde_parameters,
            pde_parameters_class=pde_parameters_class,
            simulation_dt=simulation_dt,
            task=task,
            t=t,
        )

        output = list(torch.transpose(output, 0, 1))

        if not return_dict:
            return (output,)

        return SwinDiTOutput(sample=output)


#################################################################################
#                                   FasterDITs Configs                          #
#################################################################################


def PDE_custom(**kwargs):
    return PDEImpl(**kwargs)


def PDE_S(**kwargs):
    return PDEImpl(down_factor=2, hidden_size=96, num_heads=4, depth=[2, 5, 8, 5, 2], ffn_type='rep', rep=1, mlp_ratio=4,
                 attn_type='v2', posemb_type='rope2d', downsampler='dwconv5', down_shortcut=1, **kwargs)


def PDE_B(**kwargs):
    return PDEImpl(down_factor=2, hidden_size=192, num_heads=8, depth=[2, 5, 8, 5, 2], ffn_type='rep', rep=1, mlp_ratio=4,
                 attn_type='v2', posemb_type='rope2d', downsampler='dwconv5', down_shortcut=1, **kwargs)


def PDE_L(**kwargs):
    return PDEImpl(down_factor=2, hidden_size=384, num_heads=16, depth=[2, 5, 8, 5, 2], ffn_type='rep', rep=1,
                 mlp_ratio=4, attn_type='v2', posemb_type='rope2d', downsampler='dwconv5', down_shortcut=1, **kwargs)

PDE_models = {
    'PDE-custom': PDE_custom,
    'PDE-S': PDE_S,  # U-DiT-S
    'PDE-B': PDE_B,  # U-DiT-B
    'PDE-L': PDE_L,  # U-DiT-L
}