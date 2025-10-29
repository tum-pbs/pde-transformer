import torch.nn as nn
from torch import arange

class PixelShuffle3D(nn.Module):
    def __init__(self, upscale_factor=None):
        super().__init__()

        if upscale_factor is None:
            raise TypeError(
                "__init__() missing 1 required positional argument: 'upscale_factor'"
            )

        self.upscale_factor = upscale_factor

    def forward(self, x):
        if x.ndim < 3:
            raise RuntimeError(
                f"pixel_shuffle expects input to have at least 3 dimensions, but got input with {x.ndim} dimension(s)"
            )
        elif x.shape[-4] % self.upscale_factor**3 != 0:
            raise RuntimeError(
                f"pixel_shuffle expects its input's 'channel' dimension to be divisible by the cube of upscale_factor, but input.size(-4)={x.shape[-4]} is not divisible by {self.upscale_factor**3}"
            )

        channels, in_depth, in_height, in_width = x.shape[-4:]
        nOut = channels // self.upscale_factor**3

        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        input_view = x.contiguous().view(
            *x.shape[:-4],
            nOut,
            self.upscale_factor,
            self.upscale_factor,
            self.upscale_factor,
            in_depth,
            in_height,
            in_width,
        )

        axes = arange(input_view.ndim)[:-6].tolist() + [-3, -6, -2, -5, -1, -4]
        output = input_view.permute(axes).contiguous()

        return output.view(*x.shape[:-4], nOut, out_depth, out_height, out_width)