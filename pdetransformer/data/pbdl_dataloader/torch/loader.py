import numpy as np
import torch
import torch.utils.data

from torch.utils.data import random_split
from torch.utils.data import DataLoader as TorchDataLoader

from .dataset import Dataset


def _collate_fn_(batch):
    """
    Concatenates data arrays with inflated constant layers and stacks them into batches.

    Returns:
        torch.Tensor: Data batch tensor
        torch.Tensor: Target batch tensor
    """

    data = np.stack(  # stack batch items
        [
            np.concatenate(  # concatenate data and constant layers
                [item[0]]
                + [
                    [
                        np.full_like(
                            item[0][0], constant
                        )  # inflate constants to constant layers
                    ]
                    for constant in item[2]
                ],
                axis=0,
            )
            for item in batch
        ],
        axis=0,
    )

    targets = np.stack([item[1] for item in batch])

    return torch.tensor(data), torch.tensor(targets)


class Dataloader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):

        torch_loader_args = TorchDataLoader.__init__.__code__.co_varnames[
            2:
        ]  # omit `self` and `dataset`

        # extract torch loader args
        loader_kwargs = {k: kwargs.pop(k) for k in torch_loader_args if k in kwargs}

        dataset = Dataset(
            *args, **kwargs
        )  # remaining kwargs are expected to be config parameters

        if "collate_fn" not in loader_kwargs:
            loader_kwargs["collate_fn"] = _collate_fn_

        super().__init__(dataset, **loader_kwargs)

    def info(self):
        return self.dataset.info()

    def new_split(split: list[int], *args, **kwargs):

        torch_loader_args = TorchDataLoader.__init__.__code__.co_varnames[
            2:
        ]  # omit `self` and `dataset`

        # extract torch loader args
        loader_kwargs = {k: kwargs.pop(k) for k in torch_loader_args if k in kwargs}

        dataset = Dataset(*args, **kwargs)
        splitted = random_split(dataset, split)

        if "collate_fn" not in loader_kwargs:
            loader_kwargs["collate_fn"] = _collate_fn_

        loader = [TorchDataLoader(d, **loader_kwargs) for d in splitted]
        return tuple(loader)
