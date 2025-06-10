import numpy as np
import torch
import phi.torch.flow as pf

from pbdl.torch.phi.dataset import Dataset
from pbdl.torch.phi.sampler import ConstantBatchSampler
from pbdl.logging import warn

PHIFLOW_SPATIAL_DIM = ["x", "y", "z"]


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

    return (torch.tensor(data), torch.tensor(targets))


def _collate_fn_const_(batch):
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

    return (
        torch.tensor(data),
        torch.tensor(targets),
        batch[0][3],
    )  # all constants are equal


class Dataloader(torch.utils.data.DataLoader):
    def __init__(self, *args, batch_by_const=None, ret_batch_const=False, **kwargs):

        if ret_batch_const and not batch_by_const:
            ret_batch_const = False
            warn(
                f"Warning: Flag ret_batch_const cannot be enabled when batch_by_const is disabled. Ignoring flag."
            )

        # dispatch arguments
        loader_kwargs = {
            k: kwargs.pop(k)
            for k in [
                "batch_size",
                "shuffle",
                "sampler",
                "batch_sampler",
                "num_workers",
                "collate_fn",
                "pin_memory",
                "drop_last",
                "timeout",
                "worker_init_fn",
                "multiprocessing_context",
                "generator",
                "prefetch_factor",
                "persistent_workers",
                "pin_memory_device",
            ]
            if k in kwargs
        }
        dset_kwargs = {
            k: kwargs.pop(k)
            for k in [
                "intermediate_time_steps",
                "normalize",
                "sel_sims",
                "trim_start",
                "trim_end",
                "step_size",
                "disable_progress",
            ]
            if k in kwargs
        }

        self.dataset = Dataset(
            *args[:2], **dset_kwargs, **kwargs
        )  # remaining kwargs are expected to be config parameters

        if batch_by_const:
            batch_size = loader_kwargs.pop("batch_size")

            loader_kwargs["batch_sampler"] = ConstantBatchSampler(
                self.dataset, batch_size=batch_size, batch_by_const=batch_by_const
            )

        if "collate_fn" not in kwargs:
            loader_kwargs["collate_fn"] = (
                _collate_fn_const_ if ret_batch_const else _collate_fn_
            )

        super().__init__(self.dataset, **loader_kwargs)

    def info(self):
        return self.dataset.info()

    def to_phiflow(self, data):
        """Convert network input to solver input. Constant layers are ignored."""
        spatial_dim = ",".join(PHIFLOW_SPATIAL_DIM[0 : self.dataset.num_spatial_dim])

        # if necessary, cut off constant layers
        data = data[:, 0 : self.dataset.num_sca_fields, ...]

        if self.dataset.norm_strat_data:
            data = self.dataset.norm_strat_data.normalize_rev(data)

        return pf.tensor(
            data,
            pf.batch("b"),
            pf.instance("time"),
            pf.spatial(spatial_dim),
        )

    def from_phiflow(self, data):
        """Convert solver output to a network output-like format."""
        spatial_dim = ",".join(PHIFLOW_SPATIAL_DIM[0 : self.dataset.num_spatial_dim])

        data = data.native(["b", "time", spatial_dim])

        if self.dataset.norm_strat_data:
            data = self.dataset.norm_strat_data.normalize(data)

        return data

    def cat_constants(self, data, like):
        """Concatenate constants from `like` to `data`. Useful for mapping network outputs to network inputs of the next iteration."""
        return torch.cat(
            [
                data,
                like[
                    :,
                    self.dataset.num_sca_fields : self.dataset.num_sca_fields
                    + self.dataset.num_const,
                    ...,
                ],
            ],
            axis=1,
        )  # dim 0 is batch dimension
