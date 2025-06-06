
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from diffusers import SchedulerMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class OdeEulerSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class OdeEulerScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 100,
        sigma_min: float = 1e-4,
    ):
        timesteps = np.linspace(1, num_train_timesteps,
                                num_train_timesteps, dtype=np.float32).copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        self.timesteps = timesteps / num_train_timesteps
        self.sigma_min = sigma_min
        self.dt = 1.0 / num_train_timesteps

        self._step_index = None
        self._begin_index = None

        self.timesteps = self.timesteps.to("cpu")  # to avoid too much CPU/GPU communication

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the beginning index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The beginning index for the scheduler.
        """
        self._begin_index = begin_index

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[int, torch.IntTensor],
        noise: torch.FloatTensor,
    ) -> torch.Tensor:
        """
        Forward process in flow-matching

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.
            noise (`torch.FloatTensor`):
                The noise tensor to be scaled.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        timestep = self.timesteps[self.step_index]
        sample = (1 - (1 - self.sigma_min)) * timestep * noise + timestep * sample

        return sample

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        self.num_inference_steps = num_inference_steps

        timesteps = np.linspace(
            1, num_inference_steps, num_inference_steps, dtype=np.float32
        ).copy()

        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        self.dt = 1.0 / num_inference_steps
        self.timesteps = timesteps / num_inference_steps
        self.timesteps = self.timesteps.to(device=device)

        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):

        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[OdeEulerSchedulerOutput, Tuple]:
        """
        Solves neural ODE via Euler method.
        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether to return a [`~OdeEulerSchedulerOutput`] or
                tuple.

        Returns:
            [`~OdeEulerSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~OdeEulerSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        prev_sample = sample + model_output * self.dt
        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return OdeEulerSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps