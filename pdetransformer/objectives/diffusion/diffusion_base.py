from abc import ABC, abstractmethod
from torch import nn
from typing import Tuple, Optional

import torch
import torch.utils.data
import torch.utils.data

class DiffusionBase(ABC, nn.Module):

        @abstractmethod
        def sample_time(self, batch_size: int) -> torch.Tensor:
            """
            Sample diffusion time
            :param batch_size: batch size
            :return: diffusion time
            """
            pass

        @abstractmethod
        def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            #### Get $q(x_t|x_0)$ distribution
            :param x0: noise sample
            :param t: diffusion time
            :return: mean and variance of $q(x_t|x_0)$
            """
            pass


        def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None)\
                -> Tuple[torch.Tensor, torch.Tensor]:
            """
            #### Sample from $q(x_t|x_0)$
            :param x0:
            :param t:
            :param eps:
            :return:
            """
            if eps is None:
                eps = torch.randn_like(x0)

            # get $q(x_t|x_0)$
            mean, var = self.q_xt_x0(x0, t)
            # Sample from $q(x_t|x_0)$
            return mean + (var ** 0.5) * eps, eps

        @abstractmethod
        def loss(self, x0: torch.Tensor, eps_model: nn.Module, noise: Optional[torch.Tensor] = None):
            """
            Denoising loss
            :param x0: data
            :param eps_model:
            :param noise: noise sample
            :return: estimated noise
            """
            pass