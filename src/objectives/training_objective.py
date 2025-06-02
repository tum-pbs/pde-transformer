from abc import ABC, abstractmethod
from typing import Tuple, Dict
import torch.nn as nn
import torch

class TrainingObjective(ABC, nn.Module):
    """
    Base class for training objectives.
    """

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def loss(self, model: nn.Module, batch, **kwargs) \
            -> Tuple[torch.Tensor, Dict]:
        """
        Compute the loss function for the given model and batch.
        :param model:
        :param batch:
        :param kwargs: additional arguments
        :return:
        """
        pass