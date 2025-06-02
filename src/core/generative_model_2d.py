from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import torch
import lightning

class GenerativeModel2D(lightning.LightningModule, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor,
                conditioning: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the model.
        :param x: noised data
        :param t: diffusion time
        :param conditioning:
        :return: score \nabla_x \log p_t(x)
        """
        pass

    def init_from_ckpt(self, path: str, ignore_keys:List[str]=None):
        if ignore_keys is None:
            ignore_keys = list()

        if Path(path).is_dir():
            path = Path(path).joinpath("last.ckpt")
        else:
            path = Path(path)

        if path.is_file():
            sd = torch.load(path, map_location="cpu")["state_dict"]
            keys = list(sd.keys())
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        print("Deleting key {} from state_dict.".format(k))
                        del sd[k]
            self.load_state_dict(sd, strict=False)
            print(f"Restored from {path}")