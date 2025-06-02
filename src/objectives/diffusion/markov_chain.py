from typing import Tuple, Optional

from .diffusion_base import DiffusionBase

import torch.nn.functional as F
import torch
import torch.nn as nn

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

class MarkovChain(DiffusionBase):

    def __init__(self, n_steps: int, beta_start:float=0.001, beta_end:float=0.02):
        """
        Diffusion as defined using discrete Markov Chain in DDPM paper
        :param n_steps: number of Markov Chain Steps
        :param beta_start: beta value for the first step (i=1)
        :param beta_end: beta value for the last step (i=n_steps)
        """

        super().__init__()

        assert beta_start > 0, f"beta_start ({beta_start}) must be greater than 0"
        assert beta_end > beta_start, f"beta_end ({beta_end}) must be greater than beta_start {beta_start}"

        self.beta_tensor = torch.linspace(0.0001, 0.02, n_steps)
        self.register_buffer('beta', self.beta_tensor)

        self.alpha_tensor = 1. - self.beta
        self.register_buffer('alpha', self.alpha_tensor)

        self.alpha_bar_tensor = torch.cumprod(self.alpha, dim=0)
        self.register_buffer('alpha_bar', self.alpha_bar_tensor)

        self.n_steps = n_steps

        self.sigma2_tensor = self.beta
        self.register_buffer('sigma2', self.sigma2_tensor)

    def __call__(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q_xt_x0(x0, t)

    def sample_time(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.n_steps, (batch_size,), dtype=torch.long)

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        mean = gather(self.alpha_bar, t) ** 0.5 * x0

        var = 1 - gather(self.alpha_bar, t)

        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        if eps is None:
            eps = torch.randn_like(x0)

        mean, var = self.q_xt_x0(x0, t)

        return mean + (var ** 0.5) * eps, eps

    def p_sample_model(self, xt: torch.Tensor, t: torch.Tensor, model: nn.Module):
        """
        Sample from $p_\theta(x_{t-1}|x_t)$
        :param xt: state at time $t$
        :param t: diffusion time
        :param model: noise model
        """

        eps_theta = model(xt, t)

        alpha_bar = gather(self.alpha_bar, t)

        alpha = gather(self.alpha, t)

        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5

        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)

        var = gather(self.sigma2, t)

        eps = torch.randn(xt.shape, device=xt.device)

        return mean + (var ** .5) * eps

    def loss(self, x0: torch.Tensor, eps_model: nn.Module, noise: Optional[torch.Tensor] = None) \
            -> torch.Tensor:

        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), dtype=torch.long).to(x0)

        if noise is None:
            noise = torch.randn_like(x0).to(x0)

        xt = self.q_sample(x0, t, eps=noise)

        eps_theta = self.eps_model(xt, t)

        return F.mse_loss(noise, eps_theta)