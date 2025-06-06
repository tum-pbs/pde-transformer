import math
from typing import Optional

import lightning.pytorch
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from omegaconf import ListConfig

from .pbdl_module import PBDLDataModule

class JointDataModule(lightning.pytorch.LightningDataModule):
    r'''
    Simple wrapper around multiple PBDLDataModules for joint datasets.
    In comparison to the multi module, this class creates multiple LightningDataModules and iterates them jointly, instead of joining the actual datasets.
    See https://lightning.ai/docs/pytorch/stable/data/iterables.html for more information on how combining DataModules works in PyTorch Lightning.

    All arguments apart from dataset_name are either individual objects if they are the same for all sub modules,
    or lists of the corresponding type for individual configuration of each sub modules.
    Args:
        path_index: index dictionary with the directory for each dataset type
        dataset_names: list with names of local datasets (from datasets/datasets_local.json)
        others: see PBDLDataModule
    '''

    sub_modules: list[PBDLDataModule]

    def __init__(self,
                 path_index: dict,
                 dataset_names: list[str],
                 dataset_type: str | list[str],
                 unrolling_steps: int | list[int],
                 batch_size: int | list[int],
                 num_workers: int | list[int],
                 test_unrolling_steps: Optional[int | list[int]] = None,
                 variable_dt: bool | list[bool] = False,
                 variable_dt_stride_maximum: int | list[int] = 1,
                 cache_strategy: str | list[str] = "none",
                 max_cache_size: int | list[int] = math.inf,
                 normalize_data: Optional[str] = None,
                 normalize_const: Optional[str] = None, ):

        super().__init__()
        self.path_index = path_index
        self.dataset_names = dataset_names
        self.dataset_type = dataset_type
        self.unrolling_steps = unrolling_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_unrolling_steps = test_unrolling_steps
        self.variable_dt = variable_dt
        self.variable_dt_stride_maximum = variable_dt_stride_maximum
        self.cache_strategy = cache_strategy
        self.max_cache_size = max_cache_size

        self.normalize_data = normalize_data
        self.normalize_const = normalize_const

        self.sub_modules = []

        for i in range(len(self.dataset_names)):
            name = self.dataset_names[i]
            datatype = self.dataset_type[i] if isinstance(self.dataset_type, ListConfig) else self.dataset_type
            unrolling = self.unrolling_steps[i] if isinstance(self.unrolling_steps, ListConfig) else self.unrolling_steps
            batch = self.batch_size[i] if isinstance(self.batch_size, ListConfig) else self.batch_size
            workers = self.num_workers[i] if isinstance(self.num_workers, ListConfig) else self.num_workers
            test_unrolling = self.test_unrolling_steps[i] if isinstance(self.test_unrolling_steps, ListConfig) else self.test_unrolling_steps
            variable_dt = self.variable_dt[i] if isinstance(self.variable_dt, ListConfig) else self.variable_dt
            variable_dt_stride = self.variable_dt_stride_maximum[i] if isinstance(self.variable_dt_stride_maximum, ListConfig) else self.variable_dt_stride_maximum
            cache = self.cache_strategy[i] if isinstance(self.cache_strategy, ListConfig) else self.cache_strategy
            max_cache = self.max_cache_size[i] if isinstance(self.max_cache_size, ListConfig) else self.max_cache_size

            module = PBDLDataModule(self.path_index, name, datatype, unrolling, batch, workers, test_unrolling,
                                    variable_dt, variable_dt_stride, cache, max_cache,
                                    normalize_data=self.normalize_data, normalize_const=self.normalize_const)

            self.sub_modules.append(module)

    # runs on single, main GPU
    def prepare_data(self):
        for module in self.sub_modules:
            module.prepare_data()

    # runs on every GPU
    def setup(self, stage: str):
        for module in self.sub_modules:
            module.setup(stage)

    # see https://lightning.ai/docs/pytorch/stable/data/iterables.html for more information on the returned batches
    def train_dataloader(self):
        return [module.train_dataloader() for module in self.sub_modules]

    def val_dataloader(self):
        return [module.val_dataloader() for module in self.sub_modules]

    def test_dataloader(self):
        return [module.test_dataloader() for module in self.sub_modules]


    #def train_dataloader(self):
    #    return CombinedLoader([module.train_dataloader() for module in self.sub_modules], mode="sequential")

    #def val_dataloader(self):
    #    return CombinedLoader([module.val_dataloader() for module in self.sub_modules], mode="sequential")

    #def test_dataloader(self):
    #    return CombinedLoader([module.test_dataloader() for module in self.sub_modules], mode="sequential")


