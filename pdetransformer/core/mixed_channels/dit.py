from typing import Optional, Dict, Any

from diffusers import DiTTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_version
import torch.nn.functional as F

import torch

class ModifiedDiTTransformer2DModel(DiTTransformer2DModel):
    
    def __init__(self, num_attention_heads = 16, attention_head_dim = 72, in_channels = 4, out_channels = None,
                 num_layers = 28, dropout = 0, norm_num_groups = 32, attention_bias = True, sample_size = 32,
                 patch_size = 2, activation_fn = "gelu-approximate",
                 num_embeds_ada_norm = 1000, upcast_attention = False,
                 norm_type = "ada_norm_zero", norm_elementwise_affine = False,
                 norm_eps = 0.00001):
        super().__init__(num_attention_heads, attention_head_dim, in_channels, out_channels, num_layers, dropout, norm_num_groups, attention_bias, sample_size, patch_size, activation_fn, num_embeds_ada_norm, upcast_attention, norm_type, norm_elementwise_affine, norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        return_dict: bool = True,
    ):
        """
        The [`DiTTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
        hidden_states = self.pos_embed(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    None,
                    None,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

        # 3. Output
        conditioning = self.transformer_blocks[0].norm1.emb(timestep, class_labels, hidden_dtype=hidden_states.dtype)
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        hidden_states = self.proj_out_2(hidden_states)

        # unpatchify
        #height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)  

class CustomDiTTransformer2DModel(ModifiedDiTTransformer2DModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            return_dict: bool = True,
    ):

        if class_labels is None:
            class_labels = torch.zeros(hidden_states.shape[0], dtype=torch.long)
            class_labels = class_labels.to(hidden_states.device)

        if timestep is None:
            timestep = torch.Tensor([0]).to(hidden_states.device)

        if len(timestep.shape) == 0:
            timestep = timestep.unsqueeze(0)
            timestep = timestep.repeat(hidden_states.shape[0])
            timestep = timestep.to(hidden_states.device)

        # timestep scaling (from 0 - 1 to 0 - 1000)
        timestep = timestep * 1000.0

        return super().forward(hidden_states, timestep, class_labels,
                               cross_attention_kwargs, return_dict)