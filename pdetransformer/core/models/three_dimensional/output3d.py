from dataclasses import dataclass
from typing import Optional, List, Union
import torch
from diffusers.utils import BaseOutput


@dataclass
class Output3D(BaseOutput):
    """
    The output of 3D models.

    Args:
        reconstructed: The reconstructed output.
        hidden_states: The hidden states of the model.
        embedding: The embedding of the
    """

    sample: Optional[torch.Tensor] = None
    reconstructed: Optional[torch.Tensor] = None
    hidden_states: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
    embedding: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
