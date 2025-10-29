###############################################################################

from typing import Union, Sequence

import math

import torch.hub
import torch.utils.checkpoint as checkpoint
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from einops import rearrange

from ....utils import instantiate_from_config
from .output3d import Output3D

default_decoder_filters = [48, 96, 176, 256]
default_last = 48

from functools import lru_cache

import numpy as np

from functools import partial
from typing import Optional

import torch

try:
    from mamba_ssm import Mamba
    from mamba_ssm.ops.triton.layer_norm import RMSNorm
except ImportError:
    pass

from timm.layers.drop import DropPath

from torch import Tensor, nn
from torch.nn import Dropout

from dataclasses import dataclass


@dataclass
class ContextEncoderConfig:
    enabled: bool = True

    use_checkpoint: bool = False

    n_layer: int = 12

    hidden_size: Optional[int] = None

    num_heads: int = 8

    mlp_ratio: int = 4

    skip_connection: bool = True

    grad_ratio: float = 1.0
    """Ratio of sequence to maintain gradients for better memory usage."""

    attention_method: str = "hyper"

    in_context_patches: int = -1
    """Number of in-context patches. Default -1 is infinity."""

    init_zero_proj: bool = True


class Block(nn.Module):
    def __init__(
            self,
            dim,
            mixer_cls,
            norm_cls=nn.LayerNorm,
            residual_in_fp32=False,
            reverse=False,
            transpose=False,
            split_head=False,
            drop_path_rate=0.0,
            drop_rate=0.0,
            use_mlp=False,
            downsample=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.split_head = split_head
        self.reverse = reverse
        self.transpose = transpose
        self.drop_path = DropPath(drop_prob=drop_path_rate)
        self.dropout = Dropout(p=drop_rate)

    def forward(
            self,
            hidden_states: Tensor,
            residual: Optional[Tensor] = None,
            inference_params=None,
            **kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if self.reverse:
            hidden_states = hidden_states.flip(1)

        hidden_states = self.norm(hidden_states)

        hidden_states = hidden_states + self.drop_path(
            self.mixer(hidden_states, inference_params=inference_params, **kwargs)
        )

        if self.reverse:
            hidden_states = hidden_states.flip(1)

        return hidden_states, None

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        layer_idx=None,
        device=None,
        dtype=None,
        reverse=None,
        is_2d=False,
        drop_rate=0.1,
        drop_path_rate=0.1,
        use_mlp=False,
        transpose=False,
        split_head=False,
        use_nd=False,
        downsample=False,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        residual_in_fp32=residual_in_fp32,
        reverse=reverse,
        transpose=transpose,
        drop_rate=drop_rate,
        use_mlp=use_mlp,
        drop_path_rate=drop_path_rate,
        split_head=split_head,
        downsample=downsample,
    )
    block.layer_idx = layer_idx
    return block


if __name__ == "__main__":
    ssm_cfg = {"d_state": 16}
    blk = create_block(
        d_model=768,
        ssm_cfg=ssm_cfg,
        residual_in_fp32=True,
        drop_rate=0.1,
        drop_path_rate=0.1,
        reverse=False,
        transpose=False,
        use_mlp=False,
        is_2d=False,
        rms_norm=False,
        split_head=False,
        use_nd=False,
        downsample=False,
    ).cuda()
    x = torch.rand(4, 322, 768).cuda()
    y, _ = blk(x)
    assert x.shape == y.shape

import torch
from torch import nn


class LLMAttention(nn.Module):
    def __init__(
            self,
            dim,
            inner_dim,
            num_heads,
            causal=False,
    ):
        super().__init__()
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.proj = nn.Linear(inner_dim, dim)
        assert inner_dim % num_heads == 0, (inner_dim, num_heads)
        self.num_heads = num_heads

        from .hyper_attn import HyperAttention

        self.attn = HyperAttention(
            input_dim=inner_dim // num_heads,
            lsh_num_projs=7,
            block_size=256,
            sample_size=256,
            min_seq_len=4096,
        )
        self.causal = causal

    def forward(self, x):
        """
        X: N L H
        """
        B, L, D = x.shape
        q, k, v = (
            self.qkv(x).reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )  # B H L D // num_heads
        attn_out = self.attn(q, k, v, causal=self.causal).permute(
            0, 2, 1, 3
        )  # B H L D // num_heads
        attn_out = attn_out.reshape(B, L, -1).contiguous()
        attn_out = self.proj(attn_out)

        return attn_out


class ViTAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            use_mask=False,
            **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_mask = use_mask
        if use_mask:
            self.att_mask = nn.Parameter(torch.Tensor(self.num_heads, 196, 196))

    def forward(self, x):

        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.use_mask:
            attn = attn * torch.sigmoid(self.att_mask).expand(B, -1, -1, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


@lru_cache(maxsize=32)
def get_3d_sincos_pos_embed(
        embed_dim,
        grid_size: Union[int, Sequence[int]],
        num_region_tokens: int = 0,
        i: int = 0,
        j: int = 0,
        k: int = 0,
        temperature=10,
):
    """
    Generate a 3D sine-cosine positional embedding.
    Args:
        embed_dim (int): Total embedding dimension.
        grid_size (int): The size of each spatial dimension of the grid.
        num_region_tokens (int): Number of region tokens to include in the embedding.
        i (int):
        j (int):
        k (int): Offsets for the grid along each dimension.
        temperature (int): The temperature parameter.
    Returns:
        numpy.ndarray: A 3D sine-cosine positional embedding of shape
                       [(1 + grid_size^3) if cls_token else grid_size^3, embed_dim].
    """

    # check if grid_size is a sequence
    if isinstance(grid_size, Sequence):
        assert len(grid_size) == 3, "grid_size must be a sequence of length 3"
        grid_size_w = grid_size[0]
        grid_size_h = grid_size[1]
        grid_size_d = grid_size[2]
    else:
        grid_size_w = grid_size_h = grid_size_d = grid_size

    grid_d = torch.arange(grid_size_d, dtype=torch.float32) + i * grid_size_d
    grid_h = torch.arange(grid_size_h, dtype=torch.float32) + j * grid_size_h
    grid_w = torch.arange(grid_size_w, dtype=torch.float32) + k * grid_size_w
    grid = torch.meshgrid(grid_w, grid_h, grid_d, indexing='ij')  # Shape: [3, W, H, D]
    grid = torch.stack(grid, dim=0).reshape([3, 1, grid_size_w, grid_size_h, grid_size_d])

    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid, temperature=temperature)
    if num_region_tokens > 0:
        region_pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim,
                                                             torch.arange(num_region_tokens),
                                                             temperature=3)

        # pos_embed = np.concatenate([np.zeros([num_region_tokens, embed_dim]), pos_embed], axis=0)
        pos_embed = np.concatenate([pos_embed, region_pos_embed], axis=0)

    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, temperature=10):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    # assert embed_dim % 2 == 0
    embed_dim_ = int(math.ceil(embed_dim / 2))
    omega = np.arange(embed_dim_, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / temperature ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb[:, :embed_dim]  # (M, D), truncate to the desired embedding dimension


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid, temperature=10):
    """
    Generate 3D sine-cosine embeddings from a grid.
    Args:
        embed_dim (int): Total embedding dimension.
        grid (numpy.ndarray): Grid positions of shape [3, 1, W, H, D].
    Returns:
        numpy.ndarray: A 3D positional embedding of shape [W*H*D, embed_dim].
    """

    # assert embed_dim % 3 == 0, "Embedding dimension must be divisible by 3 for 3D embedding."
    embed_dim_per_axis = int(math.ceil(embed_dim / 3))

    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim_per_axis, grid[0].flatten(), temperature=temperature)  # Depth
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim_per_axis, grid[1].flatten(), temperature=temperature)  # Height
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim_per_axis, grid[2].flatten(), temperature=temperature)  # Width

    emb = np.concatenate([emb_d, emb_h, emb_w], axis=1)  # combine along embedding dimensions
    return emb[:, :embed_dim]  # truncate to the desired embedding dimension


class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.pretraining_tp = 1
        self.hidden_size: int = hidden_size
        self.intermediate_size: int = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LLMLayer(nn.Module):
    def __init__(
            self, dim, inner_dim, num_heads, causal=False, attention_method="hyper"
    ):
        super().__init__()

        # num_heads = inner_dim // dim

        if attention_method == "hyper":
            self.attn = LLMAttention(dim, dim, num_heads, causal=causal)
        else:
            self.attn = ViTAttention(dim, dim, num_heads, causal=causal)

        self.input_layernorm = LlamaRMSNorm(dim, eps=1e-05)
        self.post_attention_layernorm = LlamaRMSNorm(dim, eps=1e-05)
        self.mlp = LlamaMLP(dim, inner_dim)
        self.causal = causal

    def forward(self, hidden_states, residual_in=-1):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attn(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        if residual_in != -1:
            return hidden_states, 0.0
        return hidden_states


class ContextModel(nn.Module):
    def __init__(
            self,
            in_dim,
            mlp_ratio=4,
            use_checkpoint=False,
            hidden_size=768,
            num_heads=8,
            n_layers=2,
            attention_method=None,
            num_regions=1,
            in_context_patches=-1,
            region_token_active: bool = True,
            init_zero_proj=True,
            emb_dims: Optional[Sequence[int]] = None,
            region_token_only=False,
    ):
        super().__init__()

        self.emb_dims = emb_dims
        self.use_checkpoint = use_checkpoint
        self.in_context_patches = in_context_patches
        self.input_proj = nn.Linear(in_dim, hidden_size)
        self.out_proj = nn.Linear(hidden_size, in_dim)

        self.region_token_active = region_token_active
        self.region_token_only = region_token_only

        self.num_regions = num_regions
        self.init_zero_proj = init_zero_proj

        assert attention_method in ["hyper", "naive", "mamba"]

        if attention_method == "mamba":

            ssm_cfg = {"d_state": 16}
            self.layers = nn.Sequential(
                *[
                    create_block(
                        d_model=hidden_size,
                        ssm_cfg=ssm_cfg,
                        residual_in_fp32=True,
                        drop_rate=0.0,
                        drop_path_rate=0.0,
                        reverse=i % 2 == 0,
                        transpose=False,
                        use_mlp=False,
                        is_2d=False,
                        rms_norm=False,
                        split_head=False,
                        use_nd=False,
                        downsample=False,
                    )
                    for i in range(n_layers)
                ]
            )
        else:
            self.layers = nn.Sequential(
                *[
                    LLMLayer(
                        hidden_size,
                        hidden_size * mlp_ratio,
                        num_heads,
                        causal=False,
                        attention_method=attention_method,
                    )
                    for _ in range(n_layers)
                ]
            )

        self.region_tokens = nn.Parameter(torch.zeros(1, num_regions, hidden_size))
        self.hidden_size = hidden_size

        self.emb_encoder_1 = nn.Linear(hidden_size, self.emb_dims[0])
        self.emb_encoder_2 = nn.Linear(hidden_size, self.emb_dims[1])

        nn.init.zeros_(self.emb_encoder_1.weight)
        nn.init.zeros_(self.emb_encoder_1.bias)

        nn.init.zeros_(self.emb_encoder_2.weight)
        nn.init.zeros_(self.emb_encoder_2.bias)

        self.init_weights()

    def init_weights(self):

        if self.region_tokens is not None:
            nn.init.normal_(self.region_tokens, std=1e-6)

        if self.init_zero_proj:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):

        n, _, h, w, d = x.shape

        pos_embed = get_3d_sincos_pos_embed(self.hidden_size, (h, w, d),
                                            num_region_tokens=self.num_regions)

        x = rearrange(x, "n c h w d -> n (h w d) c")

        x = self.input_proj(x)
        x = torch.cat([x, self.region_tokens.repeat(n, 1, 1)], dim=1)
        x = x + torch.tensor(pos_embed).to(x)
        residual = None

        if self.use_checkpoint:
            for i, blk in enumerate(self.layers):
                x, residual = checkpoint.checkpoint(blk, x, residual)
                if i == len(self.layers) - 1:
                    x = (x + residual) if residual is not None else x
        else:
            for i, blk in enumerate(self.layers):
                x, residual = blk(x, residual)
                if i == len(self.layers) - 1:
                    x = (x + residual) if residual is not None else x

        region_tokens_out = x[:, -self.num_regions:]
        x = x[:, :-self.num_regions]

        x = self.out_proj(x)

        x = rearrange(x, "n (h w d) c -> n c h w d", h=h, w=w, d=d)

        if self.region_token_only:
            x = torch.zeros_like(x) # x = x * 0.0

        region_tokens_out = rearrange(region_tokens_out,
                                      "n r d -> (n r) d")

        enc1 = self.emb_encoder_1(region_tokens_out)
        enc2 = self.emb_encoder_2(region_tokens_out)

        if not self.region_token_active:
            enc1 = torch.zeros_like(enc1)
            enc2 = torch.zeros_like(enc2)

        return x, [enc1, enc2]


class NestedTokenization(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self,
                 model_dict,
                 context_model_dict,
                 backbone_weights_file: str = None,
                 grad_ratio_encoder: float = 0.1,
                 grad_ratio_decoder: float = 0.1,
                 crop_size: int = 128,
                 num_regions: int = 1,
                 emb_dims: Optional[Sequence[int]] = None,
                 region_token_only: bool = False,
                 region_token_active: bool = True,
                 ):

        super().__init__()

        self.model_impl = instantiate_from_config(model_dict)
        self.grad_ratio_encoder = grad_ratio_encoder
        self.grad_ratio_decoder = grad_ratio_decoder

        self.num_regions = num_regions
        self.emb_dims = emb_dims
        self.region_token_only = region_token_only
        self.region_token_active = region_token_active

        region_config = ContextEncoderConfig(**context_model_dict)

        self.region_model = RegionModel(self.model_impl, region_config, crop_size=crop_size,
                                        num_regions=self.num_regions, emb_dims=self.emb_dims,
                                        region_token_only=self.region_token_only,
                                        region_token_active=self.region_token_active)

        try:

            # load weights from pretrained backbone
            if backbone_weights_file:
                state_dict = torch.load(backbone_weights_file, map_location="cpu")["state_dict"]
                state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items() if "model." in k}
                self.model_impl.load_state_dict(state_dict, strict=True)

                print("Backbone weights loaded from: ", backbone_weights_file)
            else:
                print("No backbone weights file provided.")

        except Exception as e:
            print("Error loading backbone weights: ", e)
            raise e

        self.backbone = self.model_impl

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: Optional[torch.Tensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            pde_parameters: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_states: input tensor of shape `(batch_size, num_channels, height, width, depth)`
            timestep: timestep tensor of shape `(batch_size, 1)`
            class_labels: class label tensor of shape `(batch_size, 1)`
            pde_parameters: PDE parameters tensor of shape `(batch_size, 1)`
        """

        out = self.region_model(
            hidden_states,
            timestep=timestep,
            class_labels=class_labels,
            pde_parameters=pde_parameters,
            grad_ratio_encoder=self.grad_ratio_encoder,
            grad_ratio_decoder=self.grad_ratio_decoder,
        )

        return out


class RegionModel(AbstractModel):
    def __init__(
            self,
            backbone: nn.Module,
            context_config: ContextEncoderConfig,
            channels_last: bool = False,
            crop_size: int = 256,
            num_regions: int = 8,
            emb_dims: Optional[Sequence[int]] = None,
            region_token_only: bool = False,
            region_token_active: bool = True,
    ):

        self.channels_last = channels_last
        self.crop_size = crop_size
        self.num_regions = num_regions
        self.emb_dims = emb_dims

        self.region_token_only = region_token_only

        self.grad_ratio_encoder = 0.1
        self.grad_ratio_decoder = 0.1
        self.context_config = context_config
        self.region_token_active = region_token_active

        super().__init__()

        self.backbone = backbone

        if self.context_config.enabled:

            if self.context_config.hidden_size is None:
                hidden_size = self.backbone.latent_size
            else:
                hidden_size = self.context_config.hidden_size

            self.seq2seq = ContextModel(
                in_dim=self.backbone.latent_size,
                mlp_ratio=self.context_config.mlp_ratio,
                hidden_size=hidden_size,
                num_regions=self.num_regions,
                n_layers=self.context_config.n_layer,
                num_heads=self.context_config.num_heads,
                use_checkpoint=self.context_config.use_checkpoint,
                attention_method=self.context_config.attention_method,
                in_context_patches=self.context_config.in_context_patches,
                init_zero_proj=self.context_config.init_zero_proj,
                emb_dims=self.emb_dims,
                region_token_only=self.region_token_only,
                region_token_active=self.region_token_active
            )
        else:
            self.seq2seq = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                timestep: Optional[torch.LongTensor] = None,
                class_labels: Optional[torch.LongTensor] = None,
                pde_parameters: Optional[torch.Tensor] = None,
                grad_ratio_encoder: Optional[float] = None,
                grad_ratio_decoder: Optional[float] = None,
                ):

        if grad_ratio_encoder is None:
            grad_ratio_encoder = self.grad_ratio_encoder
        if grad_ratio_decoder is None:
            grad_ratio_decoder = self.grad_ratio_decoder

        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        n_regions_x = x.shape[2] // self.crop_size
        n_regions_y = x.shape[3] // self.crop_size
        n_regions_z = x.shape[4] // self.crop_size

        n_regions = n_regions_x * n_regions_y * n_regions_z

        assert n_regions == self.num_regions, f"Number of regions {n_regions} does not match the value inside the config {self.num_regions}"

        #############################################
        ################## Encoder ##################
        #############################################

        if n_regions > 1:  # on gradient chipping

            x, timestep, class_labels, pde_parameters = (
                self.nested_tokenization(x, timestep, class_labels, pde_parameters))

            n = x.shape[0]
            n_grad = math.ceil(n * grad_ratio_encoder)

            if grad_ratio_encoder >= 1.0:
                output3d: Output3D = self.backbone.encode(x,
                                                          timestep=timestep,
                                                          class_labels=class_labels,
                                                          pde_parameters=pde_parameters)

                hidden_states = output3d.hidden_states
                emb = output3d.embedding

            elif n_grad == 0:

                with torch.no_grad():
                    output3d = self.backbone.encode(x,
                                                    timestep=timestep,
                                                    class_labels=class_labels,
                                                    pde_parameters=pde_parameters)

                    hidden_states = output3d.hidden_states
                    emb = output3d.embedding

            else:

                idx = torch.randperm(n)
                idx_inv = torch.argsort(idx)

                x = x[idx]

                if not (timestep is None):
                    timestep = timestep[idx]
                    timestep_grad = timestep[:n_grad]
                    timestep_no_grad = timestep[n_grad:]
                else:
                    timestep_grad = None
                    timestep_no_grad = None

                if not (class_labels is None):
                    class_labels = class_labels[idx]
                    class_labels_grad = class_labels[:n_grad]
                    class_labels_no_grad = class_labels[n_grad:]
                else:
                    class_labels_grad = None
                    class_labels_no_grad = None

                if not (pde_parameters is None):
                    pde_parameters = pde_parameters[idx]
                    pde_parameters_grad = pde_parameters[:n_grad]
                    pde_parameters_no_grad = pde_parameters[n_grad:]
                else:
                    pde_parameters_grad = None
                    pde_parameters_no_grad = None

                output3d_grad = self.backbone.encode(x[:n_grad],
                                                     timestep=timestep_grad,
                                                     class_labels=class_labels_grad,
                                                     pde_parameters=pde_parameters_grad)

                with torch.no_grad():
                    output3d_stopgrad = self.backbone.encode(x[n_grad:],
                                                             timestep=timestep_no_grad,
                                                             class_labels=class_labels_no_grad,
                                                             pde_parameters=pde_parameters_no_grad)

                # merge the results
                hidden_states_grad = list(output3d_grad.hidden_states)
                hidden_states_stopgrad = list(output3d_stopgrad.hidden_states)

                hidden_states = list(
                    [
                        torch.cat([a, b], dim=0)[idx_inv]
                        for a, b in zip(hidden_states_grad, hidden_states_stopgrad)
                    ]
                )

                if isinstance(output3d_grad.embedding, list):
                    emb = list(
                        [
                            torch.cat([a, b], dim=0)[idx_inv]
                            for a, b in zip(output3d_grad.embedding, output3d_stopgrad.embedding)
                        ]
                    )
                else:
                    emb = torch.cat([output3d_grad.embedding, output3d_stopgrad.embedding],
                                    dim=0)[idx_inv]

                del output3d_grad
                del output3d_stopgrad

        else:

            output3d = self.backbone.encode(x,
                                            timestep=timestep,
                                            class_labels=class_labels,
                                            pde_parameters=pde_parameters)

            hidden_states = output3d.hidden_states
            emb = output3d.embedding

        x = rearrange(
            hidden_states[-1],
            "(N HP WP DP) C HC WC DC -> N C (HP HC) (WP WC) (DP DC)",
            HP=n_regions_x,
            WP=n_regions_y,
            DP=n_regions_z,
        )

        #############################################
        ################## Global ###################
        #############################################

        seq_out, region_emb = self.seq2seq(x)

        if self.context_config.skip_connection:
            x = seq_out + x
        else:
            x = seq_out

        ########################
        # Add region embedding to embedding
        ########################
        enc1, enc2 = region_emb

        emb_updated = [emb[0] + enc1, emb[1], emb[2] + enc2, emb[3]]
        emb = emb_updated

        x = rearrange(
            x,
            "N C (HP HC) (WP WC) (DP DC) -> (N HP WP DP) C HC WC DC ",
            HP=n_regions_x,
            WP=n_regions_y,
            DP=n_regions_z,
            HC=int(x.shape[2] // n_regions_x),
            WC=int(x.shape[3] // n_regions_y),
            DC=int(x.shape[4] // n_regions_z),
        )

        hidden_states = list(hidden_states)
        hidden_states[-1] = x

        #############################################
        ################## Decoder ##################
        #############################################

        if n_regions > 1:

            n = x.shape[0]
            n_grad = math.ceil(n * grad_ratio_decoder)

            if grad_ratio_decoder >= 1.0:

                x = x.clone() # cloning is necessary if view was created in no_grad mode
                emb = [e.clone() for e in emb]
                hidden_states = [h.clone() for h in hidden_states]

                output3d = self.backbone.decode(x, emb, hidden_states)
                x = output3d.reconstructed

            elif n_grad == 0:
                with torch.no_grad():
                    output3d = self.backbone.decode(x, emb, hidden_states)
                    x = output3d.reconstructed

            else:
                idx = torch.randperm(n)
                idx_inv = torch.argsort(idx)

                x = x[idx]
                hidden_states = [h[idx] for h in hidden_states]

                if isinstance(emb, list):
                    emb = [e[idx] for e in emb]
                else:
                    emb = [emb[idx]]

                output3_grad = self.backbone.decode(x[:n_grad], [e[:n_grad] for e in emb],
                                                    [h[:n_grad] for h in hidden_states])

                with torch.no_grad():

                    output3d_stopgrad = self.backbone.decode(x[n_grad:],
                                                             [e[n_grad:] for e in emb],
                                                             [h[n_grad:] for h in hidden_states])

                x = torch.cat([output3_grad.reconstructed, output3d_stopgrad.reconstructed], dim=0)[idx_inv]

        else:

            output3d = self.backbone.decode(x, emb, hidden_states)
            x = output3d.reconstructed

        x = rearrange(
            x,
            "(N HP WP DP) C HC WC DC -> N C (HP HC) (WP WC) (DP DC)",
            HP=n_regions_x,
            WP=n_regions_y,
            DP=n_regions_z,
        )

        return Output3D(reconstructed=x, sample=x)

    def nested_tokenization(self, x, timestep, class_labels, pde_parameters):

        n_regions_x = x.shape[2] // self.crop_size
        n_regions_y = x.shape[3] // self.crop_size
        n_regions_z = x.shape[4] // self.crop_size
        n_regions = n_regions_x * n_regions_y * n_regions_z

        x = rearrange(
            x,
            "N C (HP HC) (WP WC) (DP DC) -> (N HP WP DP) C HC WC DC ",
            HP=n_regions_x,
            WP=n_regions_y,
            DP=n_regions_z,
            HC=self.crop_size,
            WC=self.crop_size,
            DC=self.crop_size,
        )

        if not (timestep is None):
            timestep = timestep.repeat_interleave(n_regions, dim=0)

        if not (class_labels is None):
            class_labels = class_labels.repeat_interleave(n_regions, dim=0)

        if not (pde_parameters is None):
            pde_parameters = pde_parameters.repeat_interleave(n_regions, dim=0)

        return x, timestep, class_labels, pde_parameters
