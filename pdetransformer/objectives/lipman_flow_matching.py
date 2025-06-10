from typing import Tuple, Dict

import torch
from torch import nn as nn

from .paths import PathBase
from .training_objective import TrainingObjective
from .utils import sample_time, get_weighting_function
from ..utils import instantiate_from_config

class LipmanFlowMatching(TrainingObjective):
    """
    Objective for matching the flow of a given path.
    See Lipman et al. 2023 "Flow Matching for Generative Modeling" for details.
    """

    def __init__(self, path: Dict, weighting: str = 'constant'):

        super().__init__()

        self.path : PathBase = instantiate_from_config(path)
        self.weighting_function = get_weighting_function(weighting)
        self.start_time = 0.0
        self.end_time = 1.0

    def loss(self, model: nn.Module, batch: dict, **kwargs) -> Tuple[torch.Tensor, Dict]:

        data = batch.pop("data")
        t = sample_time(data.device, data.shape[0], t0=self.start_time,
                        t1=self.end_time)

        # get all relevant quantities: x_t = alpha_t * x_1 + sigma_t * epsilon
        alpha = self.path.alpha(t)
        grad_alpha = self.path.grad_alpha(t)
        sigma = self.path.sigma(t)
        grad_sigma = self.path.grad_sigma(t)

        # sample noise
        epsilon = torch.randn(size=data.shape, device=data.device, dtype=data.dtype)
        noise = torch.einsum("abcd,a->abcd", epsilon, sigma)

        # get x_t
        mu_x = torch.einsum('abcd,a->abcd', data, alpha)
        x = mu_x + noise

        # prediction of flow
        flow_prediction = model.forward(x, t, **batch).sample

        grad_mu_x = torch.einsum('abcd,a->abcd', data, grad_alpha)

        w = torch.multiply(grad_sigma, 1 / sigma)

        u_t = torch.einsum("abcd,a->abcd", (x - mu_x), w) + grad_mu_x

        loss = torch.mean((u_t - flow_prediction) ** 2, dim=[1, 2, 3])

        weighting = self.weighting_function(t, torch.ones_like(t))

        weighted_loss = torch.mean(torch.multiply(weighting, loss))

        return weighted_loss, {}
