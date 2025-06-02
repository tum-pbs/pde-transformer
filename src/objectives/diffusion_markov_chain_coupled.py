from src.objectives.training_objective import TrainingObjective

from typing import Tuple, Dict
import torch.nn as nn
import torch

from src.utils import instantiate_from_config

class DiffusionMarkovChainCoupled(TrainingObjective):

    def __init__(self, diffusion: Dict, loss: Dict):

        super().__init__()
        self.diffusion = instantiate_from_config(diffusion)
        self._loss = instantiate_from_config(loss)

    def loss(self, model: nn.Module, batch, **kwargs) -> Tuple[torch.Tensor, Dict]:

        data = batch.pop("data")

        y = batch.pop("y")

        t = self.diffusion.sample_time(data.shape[0]).to(data.device)

        samples, noise = self.diffusion.q_sample(data, t)

        input = torch.cat((samples, y), dim=1)
        score_estimate = model.forward(input, t, **batch).sample

        score_loss = self._loss(data, score_estimate, noise)

        return score_loss, {}