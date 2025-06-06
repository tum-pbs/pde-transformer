from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import torch.nn.functional as F
import torch.nn as nn
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings
from diffusers.utils import BaseOutput
from einops import rearrange
from torch.nn import PixelShuffle
import math

import numpy as np
from timm.models.layers import DropPath
import torch

from pdetransformer.core.mixed_channels.udit import FinalLayer, precompute_freqs_cis_2d, apply_rotary_emb

###############################
# We need to create subclass of Swinv2PreTrainedModel because it sets use_mask_token=True
# This will create a parameter swinv2.embeddings.mask_token that will receive no gradient if bool_masked_pos is None
# This then triggers https://github.com/Lightning-AI/pytorch-lightning/issues/17212
###############################

class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        """
        Args:
            in_features: input features dimension.
            hidden_features: hidden features dimension.
            out_features: output features dimension.
            act_layer: activation function.
            drop: dropout rate.
        """

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_size = x.size()
        x = x.view(-1, x_size[-1])
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.view(x_size)
        return x

# Copied from transformers.models.swin.modeling_swin.window_partition
def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape

    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows


# Copied from transformers.models.swin.modeling_swin.window_reverse
def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    num_channels = windows.shape[-1]
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, num_channels)
    return windows

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
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
        x = self.proj(x)

        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c: int = 3, embed_dim: int = 48, patch_size: int = 4, overlap_size: int = 1,
                 bias:bool = False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=(patch_size+2*overlap_size, patch_size+2*overlap_size),
                              stride=patch_size, padding=overlap_size, bias=bias)

        self.patch_size = patch_size

    def forward(self, x, periodic_x: bool = False, periodic_y: bool = False):

        # x shape = (B, C, H, W)
        if periodic_x:

            x1 = x[:, :, :, :self.patch_size]
            x2 = x[:, :, :, -self.patch_size:]
            x = torch.cat((x2, x, x1), dim=-1)

        if periodic_y:

            x1 = x[:, :, :self.patch_size, :]
            x2 = x[:, :, -self.patch_size:, :]
            x = torch.cat((x2, x, x1), dim=-2)

        x = self.proj(x)

        if periodic_x:

            x = x[:, :, :, 1:-1]

        if periodic_y:

            x = x[:, :, 1:-1, :]

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat, keep_dim=False):
        super(Downsample, self).__init__()

        if keep_dim:
            n_feat_out = n_feat // 4
        else:
            n_feat_out = n_feat // 2

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat_out, kernel_size=3, stride=1, padding=1, bias=False),
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
    def __init__(self, n_feat, keep_dim=False):
        super(Upsample, self).__init__()

        if keep_dim:
            n_feat_out = n_feat * 4
        else:
            n_feat_out = n_feat * 2

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat_out, kernel_size=3, stride=1, padding=1, bias=False),
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

class TokenInitializer(nn.Module):
    """
    Carrier token Initializer based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """
    def __init__(self,
                 dim,
                 window_size):
        """
        Args:
            dim: feature size dimension.
            window_size: window size.
        """
        super().__init__()

        self.window_size = window_size
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        to_global_feature = nn.Sequential()
        to_global_feature.add_module("pos", self.pos_embed)

        self.to_global_feature = to_global_feature
        self.window_size = window_size

    def forward(self, x):

        x = x.permute(0, 3, 1, 2)

        x = self.to_global_feature(x)

        B, C, H, W = x.shape

        pad_right = (self.window_size - W % self.window_size) % self.window_size
        pad_bottom = (self.window_size - H % self.window_size) % self.window_size

        x = F.pad(x, (0, pad_right, 0, pad_bottom, 0, 0, 0, 0), mode='constant', value=0)

        x = torch.nn.functional.avg_pool2d(x, kernel_size=(self.window_size, self.window_size), stride=(self.window_size, self.window_size),
                                           divisor_override=(self.window_size - pad_right) * (self.window_size - pad_bottom), padding=0)

        x = x.permute(0, 2, 3, 1)

        return x

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
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb_impl is not None:
            emb = self.emb_impl(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate = emb.chunk(6, dim=1)
        return msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate

class PDEStage(nn.Module):
    def __init__(
        self, dim: int, depth: int,
            num_heads: int, window_size: int,
            periodic=False, carrier_token_active: bool = True,
            mlp_ratio: float = 4.0,
            drop_path: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        blocks = []
        for i in range(depth):

            block = PDEBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                carrier_token_active=carrier_token_active,
                drop_path=drop_path,
            )
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.periodic = periodic
        self.window_size = window_size

        self.shift_size = window_size // 2

        self.carrier_token_active = carrier_token_active

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

        if height < self.window_size or width < self.window_size:
            return None

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

            mask_windows = window_partition(img_mask, self.window_size)
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

        B, C, H, W = hidden_states.shape

        # precompute attention mask
        attn_mask_precomputed = self.get_attn_mask(self.window_size // 2, H, W, hidden_states.dtype,
                                                   hidden_states.device)

        for n, block in enumerate(self.blocks):

            shift_size = 0 if n % 2 == 0 else self.window_size // 2

            # channels last
            hidden_states = torch.permute(hidden_states, (0, 2, 3, 1))

            if shift_size > 0:
                attn_mask = attn_mask_precomputed
                shifted_hidden_states = torch.roll(hidden_states, shifts=(-shift_size, -shift_size),
                                                   dims=(1, 2))
            else:
                attn_mask = None
                shifted_hidden_states = hidden_states

            shifted_hidden_states, pad_values = self.maybe_pad(shifted_hidden_states, H, W)
            _, height_pad, width_pad, _ = shifted_hidden_states.shape

            if self.carrier_token_active:
                ct = self.global_tokenizer(hidden_states)
            else:
                ct = None

            hidden_states = window_partition(shifted_hidden_states, self.window_size)

            hidden_states, ct = block(hidden_states, ct, timestep=timestep, class_labels=class_labels, emb=cond,
                                      attn_mask=attn_mask)

            hidden_states = window_reverse(hidden_states, self.window_size, height_pad, width_pad)

            if height_pad > 0 or width_pad > 0:
                hidden_states = hidden_states[:, :H, :W, :].contiguous()

            if shift_size > 0:
                hidden_states = torch.roll(hidden_states, shifts=(shift_size, shift_size),
                                                   dims=(1, 2))

            hidden_states = torch.permute(hidden_states, (0, 3, 1, 2))

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

        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

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

        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

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

class CarrierTokenAttention2DTimestep(nn.Module):


    def __init__(self, dim, num_heads,
                 bias=False, posemb_type='rope2d', attn_type='v2', **kwargs):

        super(CarrierTokenAttention2DTimestep, self).__init__()
        if kwargs != dict():  # is not empty
            print(f'Kwargs: {kwargs}')

        self.dim = dim
        self.heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=bias)

        # v2
        self.attn_type = attn_type
        if attn_type == 'v2':
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        self.dh = 1
        self.dw = 1

        # posemb
        self.posemb_type = posemb_type

        # posemb type
        if self.posemb_type == 'rope2d':
            self.freqs_cis = None

    def forward(self, x):

        b, n, c = x.size()
        x = torch.unsqueeze(x, 1)

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if self.posemb_type == 'rope2d':

            if self.freqs_cis is None or self.freqs_cis.shape[0] != n:

                self.freqs_cis = precompute_freqs_cis_2d(self.dim // self.heads, n).to(x.device)

            # q, k input shape: B N H Hc
            q, k = map(lambda t: rearrange(t, 'b p n (h d) -> (b p) n h d', h=self.heads), qkv[:-1])

            v = rearrange(qkv[2], 'b p n (h d) -> b p h n d', h=self.heads)

            q, k = apply_rotary_emb(q, k, freqs_cis=self.freqs_cis)

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

        x = rearrange(x, 'b p h n d -> b p n (h d)')

        x = x[:, 0]

        return x

class WindowAttention2DTime(nn.Module):

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
        self.attn_drop = nn.Dropout(attn_drop)

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
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, -1, 3, self.num_heads, C // self.num_heads)
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
                B // mask_shape, mask_shape, self.num_heads, N, N
            ) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        return x

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
                # self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2).transpose(1,2))
                self.pos_emb = self.cpb_mlp(relative_coords_table.permute(0, 2, 3, 1))
            else:
                self.pos_emb = self.cpb_mlp(relative_coords_table.flatten(2))

        input_tensor = input_tensor + self.pos_emb
        return input_tensor

class PDEBlock(nn.Module):

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
        window_size=7,
        last=False,
        do_propagation=False,
        carrier_token_active=True,
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
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            window_size: window size for sliding window attention.
            last: bool argument to indicate if this is the last block in the PDE.
            do_propagation: bool argument to indicate if this block is used for propagation.
            carrier_token_active: bool argument to indicate if carrier tokens are used.
        """

        # positional encoding for windowed attention tokens
        self.norm1 = norm_layer(dim)

        self.carrier_token_active = carrier_token_active

        self.cr_window = 1
        self.attn = WindowAttention2DTime(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            resolution=window_size,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
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

            # if do hierarchical attention, this part is for carrier tokens
            self.hat_norm1 = norm_layer(dim)
            self.hat_norm2 = norm_layer(dim)
            self.hat_attn = CarrierTokenAttention2DTimestep(
                dim=dim,
                num_heads=num_heads,
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

    def forward(self, x, carrier_tokens,
                timestep: Optional[torch.LongTensor] = None,
                class_labels: Optional[torch.LongTensor] = None,
                emb: Optional[torch.LongTensor] = None,
                attn_mask: Optional[torch.Tensor] = None):

        B, H, W, N = x.shape
        ct = carrier_tokens

        x = x.view(B, H * W, N)

        Bc = emb.shape[0]

        if self.carrier_token_active:

            Bc, Hc, Wc, Nc = ct.shape

            # positional bias for carrier tokens
            ct = self.hat_pos_embed(ct)
            ct = ct.reshape(Bc, Hc * Wc, Nc)


            ######## DiT block with MSA, MLP, and AdaIN ########
            msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate = self.adain_1(timestep=timestep,
                                                                                          class_labels=class_labels,                                                                      emb=emb)
            ct_msa = self.hat_norm1(ct)
            ct_msa = ct_msa * (1 + msa_scale[:, None]) + msa_shift[:, None]

            # attention plus mlp

            ct_msa = self.hat_attn(ct_msa, attn_mask=attn_mask)

            ct_msa = ct_msa * (1 + msa_gate[:, None])
            ct = ct + self.hat_drop_path(ct_msa)

            ct_mlp = self.hat_norm2(ct)
            ct_mlp = ct_mlp * (1 + mlp_scale[:, None]) + mlp_shift[:, None]
            ct_mlp = self.hat_mlp(ct_mlp)
            ct_mlp = ct_mlp * (1 + mlp_gate[:, None])

            ct = ct + self.hat_drop_path(ct_mlp)
            ct = ct.reshape(x.shape[0], -1, N)

            # concatenate carrier_tokens to the windowed tokens
            x = torch.cat((ct, x), dim=1)


        ########### DiT block with MSA, MLP, and AdaIN ############
        msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate = self.adain_2(timestep=timestep,
                                                                                      class_labels=class_labels,
                                                                                      emb=emb)

        num_windows_total = int(B // Bc)

        msa_shift = msa_shift.repeat_interleave(num_windows_total, dim=0)
        msa_scale = msa_scale.repeat_interleave(num_windows_total, dim=0)
        msa_gate = msa_gate.repeat_interleave(num_windows_total, dim=0)
        mlp_shift = mlp_shift.repeat_interleave(num_windows_total, dim=0)
        mlp_scale = mlp_scale.repeat_interleave(num_windows_total, dim=0)
        mlp_gate = mlp_gate.repeat_interleave(num_windows_total, dim=0)

        x_msa = self.norm1(x)

        x_msa = x_msa * (1 + msa_scale[:, None]) + msa_shift[:, None]

        x_msa = self.attn(x_msa, attn_mask=attn_mask)
        x_msa = x_msa * (1 + msa_gate[:, None])

        x = x + self.drop_path(x_msa)

        x_mlp = self.norm2(x)

        x_mlp = x_mlp * (1 + mlp_scale[:, None]) + mlp_shift[:, None]
        x_mlp = self.mlp(x_mlp)
        x_mlp = x_mlp * (1 + mlp_gate[:, None])
        x = x + self.drop_path(x_mlp)

        ##########################################################

        if self.carrier_token_active:

            # for hierarchical attention we need to split carrier tokens and window tokens back
            ctr, x = x.split(
                [
                    x.shape[1] - self.window_size * self.window_size,
                    self.window_size * self.window_size,
                ],
                dim=1,
            )

            ct = ctr.reshape(Bc, Hc * Wc, Nc)  # reshape carrier tokens.

            if self.last and self.do_propagation:
                # propagate carrier token information into the image
                ctr_image_space = ctr.transpose(1, 2).reshape(
                    B, N, self.cr_window, self.cr_window
                )
                x = x + self.gamma1 * self.upsampler(
                    ctr_image_space.to(dtype=torch.float32)
                ).flatten(2).transpose(1, 2).to(dtype=x.dtype)

        return x, ct

class ConditionedEncoder2DBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 num_groups: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.gn_1 = nn.GroupNorm(num_groups, in_channels)
        self.activation_1 = nn.GELU()
        self.conv_1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

        self.mlp_scale_bias = nn.Linear(embed_dim, 2 * in_channels)
        self.gn_2 = nn.GroupNorm(num_groups, in_channels)
        self.activation_2 = nn.GELU()
        self.conv_2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x, embedding):

        scale_and_shift = self.mlp_scale_bias(embedding)
        scale, shift = scale_and_shift.chunk(2, dim=-1)

        x_res = x

        x = self.gn_1(x)
        x = self.activation_1(x)
        x = self.conv_1(x)
        x = self.gn_2(x)
        x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        x = self.activation_2(x)
        x = self.conv_2(x)

        x = x + x_res

        return x

class ConditionedEncoder2D(nn.Module):

    def __init__(self,
                 in_channels: int,
                 feature_embedding_dim: int,
                 num_downsampling_layers: int,
                 embedding_dim: int,
                 num_groups: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.feature_embedding_dim = feature_embedding_dim
        self.num_downsampling_layers = num_downsampling_layers
        self.embedding_dim = embedding_dim

        self.feature_embed = nn.Conv2d(in_channels, feature_embedding_dim, 3, 1, 1)
        self.downsampling_layers = nn.ModuleList()
        for i in range(num_downsampling_layers):
            self.downsampling_layers.append(
                nn.Conv2d(feature_embedding_dim * 2 ** i, feature_embedding_dim * 2 ** (i+1), 3, 2, 1)
            )
        self.blocks = nn.ModuleList()
        for i in range(num_downsampling_layers - 1):
            self.blocks.append(
                ConditionedEncoder2DBlock(feature_embedding_dim * 2 ** (i+1), embedding_dim,
                                          num_groups=num_groups)
            )


    def forward(self, x, embedding):

        x = self.feature_embed(x)

        res_list = [x]

        x = self.downsampling_layers[0](x)

        for i in range(self.num_downsampling_layers - 1):
            x = self.blocks[i](x, embedding)
            res_list.append(x)
            x = self.downsampling_layers[i+1](x)

        res_list.append(x)

        return res_list

ConditionedDecoder2DBlock = ConditionedEncoder2DBlock

class DecoderUpsamplingBlock(nn.Module):

        def __init__(self,
                    in_channels: int,
                    out_channels: int):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels

            self.linear_conv = nn.Conv2d(in_channels, out_channels * 2, 1)
            self.shuffle = PixelShuffle(2)

        def forward(self, x):
            x = self.linear_conv(x)
            x = self.shuffle(x)
            return x

class ConditionedDecoder2D(nn.Module):

        def __init__(self,
                    out_channels: int,
                    feature_embedding_dim: int,
                    num_upsampling_layers: int,
                    embedding_dim: int,
                    features_first_layer: int = None,
                    num_groups: int = 32):
            super().__init__()
            self.out_channels = out_channels
            self.feature_embedding_dim = feature_embedding_dim
            self.num_upsampling_layers = num_upsampling_layers
            self.embedding_dim = embedding_dim

            self.decompress = nn.Conv2d(feature_embedding_dim, out_channels, 3, 1, 1)

            self.blocks = nn.ModuleList()
            for i in range(num_upsampling_layers - 1):
                self.blocks.append(
                    ConditionedDecoder2DBlock(feature_embedding_dim * 2 ** (num_upsampling_layers - i - 1), embedding_dim,
                                              num_groups=num_groups)
                )


            if features_first_layer is None:
                features_first_layer = feature_embedding_dim

            self.upsampling_layers = nn.ModuleList()

            local_feature_dim = feature_embedding_dim * 2 ** num_upsampling_layers
            self.upsampling_layers.append(
                DecoderUpsamplingBlock(features_first_layer, local_feature_dim)
            )
            for i in range(num_upsampling_layers-1):
                local_feature_dim = feature_embedding_dim * 2 ** (num_upsampling_layers - i - 1)
                self.upsampling_layers.append(
                    DecoderUpsamplingBlock(local_feature_dim, local_feature_dim)
                )

        def forward(self, x, embedding, encoder_outputs):

            x = self.upsampling_layers[0](x)
            x += encoder_outputs[::-1][1]

            for i in range(self.num_upsampling_layers - 1):
                x = self.blocks[i](x, embedding)
                x = self.upsampling_layers[i+1](x)
                x += encoder_outputs[::-1][i+2]

            x = self.decompress(x)

            return x

class PDEImpl(nn.Module):
    """
    Diffusion UNet model with a Transformer backbone.
    """

    def __init__(
            self,
            in_channels: int = 4,
            out_channels: int = 4,
            window_size: int = 8,
            patch_size: Optional[int] = None,
            hidden_size: int = 96,
            max_hidden_size: int = 2048,
            depth= [2, 4, 4, 6, 4, 4, 2],
            num_heads: int = 16,
            mlp_ratio: float = 4.0,
            class_dropout_prob: float = 0.1,
            num_classes=1000,
            periodic=False,
            carrier_token_active: bool = True,
            **kwargs
    ):
        super().__init__()

        assert len(depth) % 2 == 1, "Encoder and decoder depths must be equal."
        self.num_encoder_layers = len(depth) // 2

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_heads = num_heads
        self.periodic = periodic

        self.use_carrier_tokens = carrier_token_active
        self.max_hidden_size = max_hidden_size

        assert self.max_hidden_size >= hidden_size, f"max_hidden_size {max_hidden_size} must be greater than or equal to hidden_size {hidden_size}."

        dit_stage_args = {
            "drop_path": 0.0,
            "periodic": periodic,
            'carrier_token_active': carrier_token_active,
            'mlp_ratio': mlp_ratio,
        }

        if patch_size is not None:
            self.x_embedder = SimplePatchEmbed(in_channels, hidden_size, patch_size, bias=True)
            self.patch_size = patch_size
        else:
            self.x_embedder = OverlapPatchEmbed(in_channels, hidden_size, bias=True)
            self.patch_size = 1

        # timestep and label embedders
        for i in range(self.num_encoder_layers + 1):

            hidden_size_layer = min(hidden_size * 2 ** i, max_hidden_size)
            self.__setattr__(f"t_embedder_{i}", TimestepEmbedder(hidden_size_layer))
            self.__setattr__(f"y_embedder_{i}", LabelEmbedder(num_classes, hidden_size_layer, class_dropout_prob))

        # encoder
        for i in range(self.num_encoder_layers):
            hidden_size_layer = min(hidden_size * 2 ** i, max_hidden_size)
            self.__setattr__(f"encoder_level_{i}", PDEStage(dim=hidden_size_layer, num_heads=num_heads,
                                            window_size=window_size, depth=depth[i], **dit_stage_args))
            if hidden_size_layer == max_hidden_size:
                keep_dim = True
            else:
                keep_dim = False
            self.__setattr__(f"down{i}_{i+1}", Downsample(hidden_size_layer, keep_dim=keep_dim))

        # latent
        hidden_size_latent = min(hidden_size * 2 ** self.num_encoder_layers, max_hidden_size)
        self.latent = PDEStage(dim=hidden_size_latent, num_heads=num_heads,
                                            window_size=window_size, depth=depth[self.num_encoder_layers], **dit_stage_args)

        hidden_size_layer0 = min(hidden_size * 2, max_hidden_size)
        if hidden_size_layer0 >= max_hidden_size:
            keep_dim = True
        else:
            keep_dim = False

        # double hidden size for last decoder layer 0
        self.__setattr__("up1_0", Upsample(hidden_size_layer0, keep_dim=keep_dim))
        self.__setattr__("reduce_chan_level0", nn.Conv2d(2 * min(hidden_size, max_hidden_size), hidden_size_layer0, kernel_size=1, bias=True))
        self.__setattr__("decoder_level_0", PDEStage(dim=hidden_size_layer0, num_heads=num_heads,
                                        window_size=window_size, depth=depth[self.num_encoder_layers + 1], **dit_stage_args))

        # decoder layers 1 - num_encoder_layers
        for i in range(1, self.num_encoder_layers):

            hidden_size_layer = min(hidden_size * 2 ** i, max_hidden_size)
            if 2 * hidden_size_layer >= max_hidden_size:
                keep_dim = True
                hidden_size_upsample = max_hidden_size
            else:
                keep_dim = False
                hidden_size_upsample = 2 * hidden_size_layer

            self.__setattr__(f"up{i+1}_{i}", Upsample(hidden_size_upsample, keep_dim=keep_dim))
            self.__setattr__(f"reduce_chan_level{i}", nn.Conv2d(hidden_size_layer * 2, hidden_size_layer, kernel_size=1, bias=True))
            self.__setattr__(f"decoder_level_{i}", PDEStage(dim=hidden_size_layer, num_heads=num_heads,
                                            window_size=window_size, depth=depth[self.num_encoder_layers + i + 1], **dit_stage_args))

        hidden_size_out = min(2 * hidden_size, max_hidden_size)
        self.output = nn.Conv2d(hidden_size_out, hidden_size_out, kernel_size=3, stride=1, padding=1,
                                bias=True)

        self.final_layer = FinalLayer(hidden_size_out, self.out_channels * self.patch_size * self.patch_size)
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
            nn.init.normal_(self.__getattr__(f"y_embedder_{i}").embedding_table.weight, std=0.02)

            # Initialize timestep embedding MLP:
            nn.init.normal_(self.__getattr__(f"t_embedder_{i}").mlp[0].weight, std=0.02)
            nn.init.normal_(self.__getattr__(f"t_embedder_{i}").mlp[2].weight, std=0.02)


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

    def forward(self, x, t, y):
        """
        Forward pass of PDE transformer.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N, ) tensor of diffusion timesteps
        y: (N, ) tensor of class labels
        """
        x = self.x_embedder(x)  # (N, C, H, W)

        emb_list = []
        for i in range(self.num_encoder_layers + 1):
            t_emb = self.__getattr__(f"t_embedder_{i}")(t)
            y_emb = self.__getattr__(f"y_embedder_{i}")(y, self.training)
            c = t_emb + y_emb
            emb_list.append(c)

        residuals_list = []
        for i, c in enumerate(emb_list[:-1]):
            # encoder
            out_enc_level = self.__getattr__(f"encoder_level_{i}")(x, c)
            residuals_list.append(out_enc_level)
            x = self.__getattr__(f"down{i}_{i+1}")(out_enc_level)

        c = emb_list[-1]
        x = self.latent(x, c)

        for i, (residual, emb) in enumerate(zip(residuals_list[1:][::-1], emb_list[1:-1][::-1])):
            # decoder
            x = self.__getattr__(f"up{self.num_encoder_layers - i}_{self.num_encoder_layers - i - 1}")(x)
            x = torch.cat([x, residual], 1)
            x = self.__getattr__(f"reduce_chan_level{self.num_encoder_layers - i - 1}")(x)
            x = self.__getattr__(f"decoder_level_{self.num_encoder_layers - i - 1}")(x, emb)

        x = self.__getattr__(f"up1_0")(x)
        x = torch.cat([x, residuals_list[0]], 1)
        x = self.__getattr__(f"reduce_chan_level0")(x)
        x = self.__getattr__(f"decoder_level_0")(x, emb_list[1])

        # output
        x = self.output(x)
        x = self.final_layer(x, emb_list[1])  # (N, T, patch_size ** 2 * out_channels)

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

        return x

@dataclass
class PDEOutput(BaseOutput):
    """
    The output of [`PDEOutput`].

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
            in_channels: int,
            out_channels: int,
            type: str,
            periodic: bool = False,
            carrier_token_active: bool = True,
            window_size: int = 8,
            patch_size: Optional[int] = None,
            **kwargs
    ):
        super(PDETransformer, self).__init__()
        args = {'in_channels': in_channels, 'out_channels': out_channels, 'patch_size': patch_size,
                'periodic': periodic, 'carrier_token_active': carrier_token_active, 'window_size': window_size}

        args.update(kwargs)

        self.model: PDEImpl = PDE_models[type](**args)
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            return_dict: bool = True,
    ):

        if timestep is None:
            timestep = torch.Tensor([0]).to(hidden_states.device)

        if len(timestep.shape) == 0:
            timestep = timestep.unsqueeze(0)
            timestep = timestep.repeat(hidden_states.shape[0])
            timestep = timestep.to(hidden_states.device)

        # timestep scaling (from 0 - 1 to 0 - 1000)
        timestep = timestep * 1000.0

        output = self.model.forward(hidden_states, timestep, class_labels)

        if not return_dict:
            return (output,)

        return PDEOutput(sample=output)

#################################################################################
#                            PDE Transformer Configs                            #
#################################################################################


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
    'PDE-S': PDE_S,
    'PDE-B': PDE_B,
    'PDE-L': PDE_L,
}