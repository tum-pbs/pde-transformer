from typing import Optional, Tuple, List

from .sampler_base import SamplerBase

import torch.nn as nn
import torch

from ..objectives.diffusion.diffusion_base import DiffusionBase


class AncestralSampler(SamplerBase):

    name = "AncestralSampler"
    def __init__(self,
                 n_steps_inference: int,
                 n_steps_train: int,
                 shape: Tuple[int]):

        self.n_steps_inference = n_steps_inference
        self.n_steps_diffusion = n_steps_train
        self.shape = shape

        super().__init__()

    def sample(self, n_samples: int, model: nn.Module,
               diffusion: DiffusionBase,
               noise: Optional[torch.Tensor] = None) -> List[torch.Tensor]:

          if noise is None:
              noise = torch.randn(list((n_samples,) + self.shape),
                                  device=self.device_reference.device)

          else:
              noise = noise.to(self.device_reference.device)

          return self._sample(model, diffusion, noise)

    def _sample(self, model: nn.Module, diffusion: DiffusionBase,
               noise: Optional[torch.Tensor]) -> List[torch.Tensor]:

        with torch.no_grad():

            x = noise

            # for t_ in tqdm(range(self.n_steps_inference), file=sys.stdout):

            for t_ in range(self.n_steps_inference):

                t = self.n_steps_inference - t_ - 1

                x = diffusion.p_sample_model(x,
                                             x.new_full((noise.shape[0],), t, dtype=torch.long),
                                             model)

            return list(x)