from typing import Optional

import torch.nn as nn
from abc import ABC

from omegaconf import DictConfig

from ..objectives.diffusion.diffusion_base import DiffusionBase

import torch

from ..utils import instantiate_from_config


class SamplerBase(ABC, nn.Module):

    name = "SampleBase"

    def __init__(self):
        super().__init__()
        self.register_buffer('device_reference', torch.tensor(0))

    def sample(self, n_samples: int,
               model: nn.Module, diffusion: DiffusionBase,
               noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample from the model
        :param n_samples: number of samples
        :param model: score model
        :param diffusion: diffusion process
        :param noise: initialization
        :return: sample at $t=0$
        """
        pass

    def sample_conditional(self, model: nn.Module, diffusion: DiffusionBase,
               y: torch.tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample from the model conditioned on $y$
        :param model:
        :param diffusion:
        :param y:
        :param noise:
        :return:
        """
        pass

def get_samplers(configs: DictConfig):

    samplers = []
    for sampler in configs.values():
        samplers.append(instantiate_from_config(sampler))
    return samplers

