import math
from typing import TypeVar, List, Iterable, Union, Optional

import lightning.pytorch
import torch
import torch.distributed
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torchvision.transforms.v2 import Transform, ToDtype, Compose, Lambda

from .utils import SubsetSequentialSampler
from .cached_dataset import CachedDataset
from .metadata_dataset import MetadataDataset

from .pbdl_datatypes.acdm_2d import acdm_2d_datasets, acdm_2d_transforms
from .pbdl_datatypes.ape_2d import ape_2d_datasets, ape_2d_transforms
from .pbdl_datatypes.ape_2d_xxl import ape_2d_xxl_datasets, ape_2d_xxl_transforms
from .pbdl_datatypes.well_2d import well_2d_datasets, well_2d_transforms


def get_transforms(datatype:str, dataset_name:str) -> tuple[Transform, Transform, Transform]:
    if datatype == "2D_ACDM":
        return acdm_2d_transforms(dataset_name)
    elif datatype == "2D_APE":
        return ape_2d_transforms(dataset_name)
    elif datatype == "2D_APE_xxl":
        return ape_2d_xxl_transforms(dataset_name)
    elif datatype == "2D_WELL":
        return well_2d_transforms(dataset_name)
    else:
        raise ValueError(f"Unknown dataset type: {datatype}")


def get_datasets(datatype:str, dataset_name:str, directory:str, unrolling_steps:int, test_unrolling_steps:int,
                 variable_dt_stride_maximum:int,
                 test_variable_dt_stride_maximum:int,
                 normalize_data:str, normalize_const:str,
                 crop_size: Optional[int], **kwargs) -> tuple[Dataset, Dataset, Dataset]:
    params = {"dataset_name": dataset_name, "dataset_directory": directory, "unrolling_steps": unrolling_steps,
              "test_unrolling_steps": test_unrolling_steps, "variable_dt_stride_maximum": variable_dt_stride_maximum,
              "test_variable_dt_stride_maximum": test_variable_dt_stride_maximum,
              "normalize_data": normalize_data, "normalize_const": normalize_const,
              "crop_size": crop_size}

    params.update(kwargs)
    if datatype == "2D_ACDM":
        return acdm_2d_datasets(**params)
    elif datatype == "2D_APE":
        return ape_2d_datasets(**params)
    elif datatype == "2D_APE_xxl":
        return ape_2d_xxl_datasets(**params)
    elif datatype == "2D_WELL":
        return well_2d_datasets(**params)
    else:
        raise ValueError(f"Unknown dataset type: {datatype}")



class PBDLDataModule(lightning.pytorch.LightningDataModule):
    r'''
    Wrapper for individual PBDL datasets to be used with PyTorch Lightning.

    Args:
        path_index: index dictionary with the directory for each dataset type
        dataset_name: name of local dataset
        dataset_type: type of the dataset
        unrolling_steps: number of time steps between start and end of sequence
        batch_size: batch size for each GPU, i.e. larger total batch size depending on number of GPUs
        num_workers: number of worker threads in the dataloaders
        test_unrolling_steps: number of time steps between start and end of sequence for the test dataset
        variable_dt: determines if the datasets are loaded with variable time steps
        variable_dt_stride_maximum: maximum stride between time steps for variable dt datasets. A value of 1 is equal to disabling variable_dt.
        cache_strategy: strategy for caching sub data sets from ["none", "testOnly", testAndVal", "all"]
        max_cache_size: maximum number of cached items
    '''
    def __init__(self,
                 path_index: dict,
                 dataset_name: str,
                 dataset_type: str,
                 unrolling_steps: int,
                 batch_size: int,
                 num_workers: int,
                 test_unrolling_steps: Optional[int] = None,
                 variable_dt: bool = False,
                 variable_dt_stride_maximum: int = 1,
                 cache_strategy: str="none",
                 max_cache_size: int=math.inf,
                 normalize_data: Optional[str] = None,
                 normalize_const: Optional[str] = None):

        super().__init__()

        self.path_index = path_index
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.unrolling_steps = unrolling_steps
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.normalize_data = normalize_data
        self.normalize_const = normalize_const

        if test_unrolling_steps is None:
            self.test_unrolling_steps = unrolling_steps
        else:
            self.test_unrolling_steps = test_unrolling_steps

        self.variable_dt_stride_maximum = variable_dt_stride_maximum if variable_dt else 1

        self.cache_strategy = cache_strategy
        self.max_cache_size = max_cache_size

        self.set_train = None
        self.set_val = None
        self.set_test = None

        self.subset_indices_train = None
        self.subset_indices_val = None
        self.subset_indices_test = None


    # runs on single, main GPU
    def prepare_data(self):
        pass


    # runs on every GPU
    def setup(self, stage: str):
        # prevent reload when all datasets already exist
        if self.set_train and self.set_val and self.set_test:
            return

        name = self.dataset_name
        datatype = self.dataset_type
        directory = self.path_index[datatype]
        steps = self.unrolling_steps
        test_steps = self.test_unrolling_steps

        # prepare transforms and datasets
        transform_train, transform_val, transform_test = get_transforms(datatype, self.dataset_name)
        set_train, set_val, set_test = get_datasets(datatype, name, directory, steps, test_steps,
                                                    self.variable_dt_stride_maximum,
                                                    self.normalize_data, self.normalize_const)

        loading_metadata_train = {"type": datatype, "unrolling_steps": steps, "dataset_idx": 0}
        loading_metadata_test = {"type": datatype, "unrolling_steps": test_steps, "dataset_idx": 0}

        self._prepare_gpu_split_and_cache("fit", MetadataDataset(set_train, loading_metadata_train, transform_train))
        self._prepare_gpu_split_and_cache("validate", MetadataDataset(set_val, loading_metadata_train, transform_val))
        self._prepare_gpu_split_and_cache("test", MetadataDataset(set_test, loading_metadata_test, transform_test))


    def _prepare_gpu_split_and_cache(self, stage:str, dataset:Dataset, after_cache_transforms:Transform = None):
        r'''
        Creates split indices for each GPU and loads the dataset to the cache according to the cache strategy.
        Does not reload the cache if already loaded. Supplied dataset and transform should match the stage.
        '''
        if stage == "fit" and not self.set_train:
            self.subset_indices_train = self._compute_gpu_subset_indices(len(dataset))

            if self.cache_strategy in ["all"] :
                self.set_train = CachedDataset(dataset, after_cache_transforms, self.max_cache_size)
                self.set_train.fill_cache_sequentially(self.subset_indices_train)
            else:
                self.set_train = dataset


        if stage == "validate" and not self.set_val:
            self.subset_indices_val = self._compute_gpu_subset_indices(len(dataset))

            if self.cache_strategy in ["testAndVal", "all"] :
                self.set_val = CachedDataset(dataset, after_cache_transforms, self.max_cache_size)
                self.set_val.fill_cache_sequentially(self.subset_indices_val)
            else:
                self.set_val = dataset


        if stage == "test" and not self.set_test:
            self.subset_indices_test = self._compute_gpu_subset_indices(len(dataset))

            if self.cache_strategy in ["testOnly", "testAndVal", "all"] :
                self.set_test = CachedDataset(dataset, after_cache_transforms, self.max_cache_size)
                self.set_test.fill_cache_sequentially(self.subset_indices_test)
            else:
                self.set_test = dataset


    def _compute_gpu_subset_indices(self, dataset_length:int) -> list:
        r'''
        Computes local data indices for the current GPU. Discards the last few data samples to ensure even subset size across GPUs.
        '''
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        dataset_size = dataset_length - (dataset_length % world_size) # ensure even subset size
        subset_start = int(dataset_size * (rank / world_size))
        subset_end = int(dataset_size * ((rank + 1) / world_size))
        return range(subset_start, subset_end, 1)


    def train_dataloader(self):
        return DataLoader(
            self.set_train,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(self.subset_indices_train),
            num_workers = 0 if self.cache_strategy in ["all"] else self.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.set_val,
            batch_size=self.batch_size,
            sampler=SubsetSequentialSampler(self.subset_indices_val),
            num_workers = 0 if self.cache_strategy in ["testAndVal", "all"] else self.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.set_test,
            batch_size=self.batch_size,
            sampler=SubsetSequentialSampler(self.subset_indices_test),
            num_workers = 0 if self.cache_strategy in ["testOnly", "testAndVal", "all"] else self.num_workers)




# quantile min-max normalization to [0,1]
def quantile_normalization(x: torch.Tensor, qMin = 0.02, qMax = 0.98) -> torch.Tensor:
    x = x.float()
    xMin = torch.quantile(x, qMin)
    xMax = torch.quantile(x, qMax)
    if xMin >= xMax:
        return torch.zeros_like(x)
    return (x - xMin) / (xMax - xMin)


def get_plot_transform():
    return Compose(
        [
            Lambda(lambda x: quantile_normalization(x)),
            Lambda(lambda x: torch.clamp(128 * x + 128, 0, 255)),
            ToDtype(torch.uint8, scale=False)
        ]
    )
def prepare_plots():
    fn = get_plot_transform()
    return fn

