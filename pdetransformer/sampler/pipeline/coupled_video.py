import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm


import numpy as np

@dataclass
class VideoPipelineOutput(BaseOutput):
    """
    Output class for video pipelines.

    Args:
        videos (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    videos: Union[List[np.ndarray]]

class VideoPipelineDirect(DiffusionPipeline):
    r"""
    Pipeline for video generation (in a single step). Adapted from DDPMPipeline.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet):
        super().__init__()
        self.register_modules(unet=unet)

    @torch.no_grad()
    def __call__(
        self,
        data: torch.Tensor,
        num_frames: int = 10,
        output_type: Optional[str] = "",
        return_dict: bool = True,
        class_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[VideoPipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            data (`torch.Tensor`):
                The first video frame. The shape should be `(batch_size, num_channels, height, width)`.
            num_frames (`int`, *optional*, defaults to 10):
                The number of frames to generate.
            output_type (`str`, *optional*, defaults to `""`):
                The output format of the generated image. Currently only `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            kwargs (`dict`): Additional keyword arguments to be passed to the model.
        ```

        Returns:
            [`~pipelines.VideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        batch_size = data.shape[0]

        frames = [data.cpu().numpy()]
        previous_frame = data.to(self.device)

        for _ in tqdm(range(num_frames)):

            input = previous_frame
            model_output = self.unet(input, class_labels=class_labels).sample

            previous_frame = model_output
            frames.append(model_output.cpu().numpy())

        vid = np.array(frames)
        vid = np.swapaxes(vid, 0, 1)

        if not return_dict:
            return (vid,)

        return VideoPipelineOutput(videos=vid)

class VideoPipeline(DiffusionPipeline):
    r"""
    Pipeline for video generation. Adapted from DDPMPipeline.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, max_channels=3):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.max_channels = max_channels

    def init_shape(self, batch_size: int):
        r"""
        Initialize the shape of the image to be generated.

        Args:
            batch_size (`int`):
                The batch size of the input data.
        """
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels // 2,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels // 2, *self.unet.config.sample_size)

        return image_shape

    @torch.no_grad()
    def __call__(
        self,
        data: torch.Tensor,
        num_frames: int = 10,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: Optional[int] = None,
        output_type: Optional[str] = "",
        return_dict: bool = True,
        **kwargs
    ) -> Union[VideoPipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            data (`torch.Tensor`):
                The first video frame. The shape should be `(batch_size, num_channels, height, width)`.
            args (`tuple`): Additional arguments to be passed to the model.
            num_frames (`int`, *optional*, defaults to 10):
                The number of frames to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `""`):
                The output format of the generated image. Currently only `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            kwargs (`dict`): Additional keyword arguments to be passed to the model.
        ```

        Returns:
            [`~pipelines.VideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        if num_inference_steps is None:
            num_inference_steps = self.scheduler.num_train_timesteps

        batch_size = data.shape[0]

        num_channels_data = data.shape[1]
        if num_channels_data < self.max_channels:
            data = torch.cat((data, torch.zeros(data.shape[0],
                                            self.max_channels - data.shape[1],
                                            *data.shape[2:], device=data.device)), dim=1)

        image_shape = data.shape  # self.init_shape(batch_size)
        frames = [data.cpu().numpy()]
        previous_frame = data.to(self.device)

        for _ in range(num_frames):

            if self.device.type == "mps":
                # randn does not work reproducibly on mps
                x0 = randn_tensor(image_shape, generator=generator)
                x0 = x0.to(self.device)
            else:
                x0 = randn_tensor(image_shape, generator=generator, device=self.device)

            input = torch.cat([x0, previous_frame], dim=1)

            # set step values
            self.scheduler.set_timesteps(num_inference_steps)

            for t in self.progress_bar(self.scheduler.timesteps):
                # 1. predict noise model_output
                model_output = self.unet(input, t, **kwargs).sample

                # 2. compute previous image: x_t -> x_t-1
                x0 = self.scheduler.step(model_output, t, x0, generator=generator).prev_sample
                input = torch.cat([x0, previous_frame], dim=1)

            previous_frame = x0
            x0 = x0.cpu().numpy()
            frames.append(x0)

        vid = np.array(frames)
        vid = np.swapaxes(vid, 0, 1)
        if not return_dict:
            return (vid,)

        return VideoPipelineOutput(videos=vid)

class VideoPipeline3D(VideoPipeline):

    def __init__(self, unet, scheduler):
        super().__init__(unet, scheduler)

    def init_shape(self, batch_size: int):

        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels // 2,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
                self.unet.config.sample_size
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels // 2, *self.unet.config.sample_size)

        return image_shape
