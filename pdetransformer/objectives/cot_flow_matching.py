from typing import Tuple, Dict

import torch
from torch import nn as nn

from .paths import PathBase
from .training_objective import TrainingObjective
from .utils import sample_time, get_weighting_function
from ..utils import instantiate_from_config


class COTFlowMatching(TrainingObjective):
    """
    Objective for conditional optimal transport matching the flow.
    See Lipman et al. 2023 "Flow Matching for Generative Modeling" for details.
    """

    def __init__(self, sigma_min: float = 1e-4, weighting: str = 'constant'):
        super().__init__()

        self.sigma_min = sigma_min
        self.weighting_function = get_weighting_function(weighting)
        self.start_time = 0.0
        self.end_time = 1.0

    def loss(self, model: nn.Module, batch: dict, **kwargs) -> Tuple[torch.Tensor, Dict]:
        data = batch.pop("data")
        t = sample_time(data.device, data.shape[0], t0=self.start_time,
                        t1=self.end_time)

        epsilon = torch.randn(size=data.shape, device=data.device, dtype=data.dtype)

        psi_t = (1 - (1 - self.sigma_min) * t)
        psi_t = torch.einsum("a,abcd->abcd", psi_t, epsilon)
        psi_t = psi_t + torch.einsum("a, abcd->abcd", t, data)

        target = (data - (1 - self.sigma_min) * epsilon)

        flow_prediction = model.forward(psi_t, t, **batch).sample

        loss = torch.mean((flow_prediction - target) ** 2, dim=[1, 2, 3])

        weighting = self.weighting_function(t, torch.ones_like(t))

        weighted_loss = torch.mean(torch.multiply(weighting, loss))

        return weighted_loss, {}


class COTFlowMatchingCoupled(TrainingObjective):
    """
    Objective for conditional optimal transport matching the flow.
    See Lipman et al. 2023 "Flow Matching for Generative Modeling" for details.
    """

    def __init__(self, sigma_min: float = 1e-4, weighting: str = 'constant',
                 max_channels: int = 3, dropout_state: float = 0.5):

        super().__init__()

        self.sigma_min = sigma_min
        self.weighting_function = get_weighting_function(weighting)
        self.start_time = 0.0
        self.end_time = 1.0
        self.max_channels = max_channels
        self.dropout_state = dropout_state

    def loss(self, model: nn.Module, batch: dict, **kwargs) -> Tuple[torch.Tensor, Dict]:

        data = batch.pop("data")
        y = batch.pop("y")

        num_channels_data = data.shape[1]

        # pad data and y if channels are less than max_channels
        if num_channels_data < self.max_channels:
            data = torch.cat((data, torch.zeros(data.shape[0],
                                                self.max_channels - data.shape[1],
                                                *data.shape[2:], device=data.device)), dim=1)

        if y.shape[1] < self.max_channels:
            y = torch.cat((y, torch.zeros(y.shape[0],
                                          self.max_channels - y.shape[1],
                                          *y.shape[2:], device=y.device)), dim=1)

        # drop conditioning with probability dropout_state
        if torch.rand(1) < self.dropout_state:
            y = torch.zeros_like(y)

        t = sample_time(data.device, data.shape[0], t0=self.start_time,
                        t1=self.end_time)

        epsilon = torch.randn(size=data.shape, device=data.device, dtype=data.dtype)

        psi_t = (1 - (1 - self.sigma_min) * t)
        psi_t = torch.einsum("a,a...->a...", psi_t, epsilon)
        psi_t = psi_t + torch.einsum("a, a...->a...", t, data)

        target = (data - (1 - self.sigma_min) * epsilon)

        input = torch.cat((psi_t, y), dim=1)
        flow_prediction = model.forward(input, t, **batch).sample

        dims_to_reduce = tuple(range(1, target.ndim))
        # compute loss only for data channels
        loss = torch.mean((flow_prediction[:, :num_channels_data] -
                           target[:, :num_channels_data]) ** 2, dim=dims_to_reduce)

        weighting = self.weighting_function(t, torch.ones_like(t))

        weighted_loss = torch.mean(torch.multiply(weighting, loss))

        return weighted_loss, {}
