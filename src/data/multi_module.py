from typing import TypeVar, List, Iterable, Union, Optional, Tuple

import math

import lightning.pytorch
import torch
import torch.distributed
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset, Dataset
from torchvision.transforms.v2 import Transform

from omegaconf import ListConfig
from .cached_dataset import CachedDataset
from .metadata_dataset import MetadataDataset

from .utils import SubsetSequentialSampler
from .cached_dataset import CachedDataset

from .pbdl_module import get_transforms, get_datasets
from .metadata_remapping import update_boundary_condition


class MultiDataModule(lightning.pytorch.LightningDataModule):
    r'''
    Wrapper around multiple PBDL datasets for joint datasets with PyTorch Lightning.
    In comparison to the joint module, this class concatenates the datasets before attaching them to a LightningDataModule.

    Some arguments are either individual objects if they are the same for all sub datasets,
    or lists of the corresponding type for individual configuration of each sub datasets.
    Args:
        path_index: index dictionary with the directory for each dataset type
        dataset_names: list with names of local datasets
        dataset_type: type of the dataset
        unrolling_steps: number of time steps between start and end of sequence
        batch_size: batch size for each GPU, i.e. larger total batch size depending on number of GPUs
        num_workers: number of worker threads in the dataloaders
        test_unrolling_steps: number of time steps between start and end of sequence for the test dataset
        cache_strategy: strategy for caching sub data sets from ["none", "testOnly", testAndVal", "all"]
        different_resolution_strategy: strategy for handling different resolutions in the sub datasets from ["none", "rescale", "crop"]
        target_size: target size for spatial dimensions of the concatenated datasets
        max_cache_size: maximum number of cached items
    '''

    def __init__(self,
                 path_index: dict,
                 dataset_names: list[str],
                 dataset_type: str | list[str],
                 unrolling_steps: int | list[int],
                 batch_size: int,
                 num_workers: int,
                 test_unrolling_steps: Optional[int | list[int]] = None,
                 variable_dt: bool | list[bool] = False,
                 test_variable_dt: bool | list[bool] = False,
                 variable_dt_stride_maximum: int | list[int] = 1,
                 test_variable_dt_stride_maximum: int | list[int] = 1,
                 cache_strategy: str = "none",
                 max_cache_size: int = math.inf,
                 different_resolution_strategy: str = "none",
                 target_size: Optional[Tuple[int]] = None,
                 normalize_data: Optional[str] = None,
                 normalize_const: Optional[str] = None,
                 downsample_factor: int = 1,
                 max_channels: int = 1,
                 max_constants: int = 5,
                 crop_size: Optional[int] = None,
                 shuffle_train_set: bool = True,
                 **kwargs):

        super().__init__()

        self.prefetch_factor = 1
        self.path_index = path_index
        self.dataset_names = dataset_names
        self.dataset_type = dataset_type
        self.unrolling_steps = unrolling_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.downsample_factor = downsample_factor

        self.normalize_data = normalize_data
        self.normalize_const = normalize_const

        self.max_channels = max_channels
        self.max_constants = max_constants

        self.shuffle_train_set = shuffle_train_set

        self.crop_size = crop_size

        if test_unrolling_steps is None:
            self.test_unrolling_steps = unrolling_steps
        else:
            self.test_unrolling_steps = test_unrolling_steps

        self.variable_dt = variable_dt
        self.test_variable_dt = test_variable_dt
        self.variable_dt_stride_maximum = variable_dt_stride_maximum
        self.test_variable_dt_stride_maximum = test_variable_dt_stride_maximum

        self.cache_strategy = cache_strategy
        self.max_cache_size = max_cache_size
        self.different_resolution_strategy = different_resolution_strategy
        self.target_size = target_size

        self.subsets_train : List[MetadataDataset] = []
        self.subsets_val : List[MetadataDataset] = []
        self.subsets_test : List[MetadataDataset] = []

        self.set_train = None
        self.set_val = None
        self.set_test = None

        self.subset_indices_train = None
        self.subset_indices_val = None
        self.subset_indices_test = None
        
        self.additional_kwargs = kwargs


    # runs on single, main GPU
    def prepare_data(self):
        pass


    # runs on every GPU
    def setup(self, stage: str):
        # prevent reload when all datasets already exist
        if self.set_train and self.set_val and self.set_test:
            return

        for i in range(len(self.dataset_names)):
            name = self.dataset_names[i]
            datatype = self.dataset_type[i] if (isinstance(self.dataset_type, ListConfig) or isinstance(self.dataset_type, list)) else self.dataset_type
            directory = self.path_index[datatype]
            steps = self.unrolling_steps[i] if (isinstance(self.unrolling_steps, ListConfig) or isinstance(self.unrolling_steps, list)) else self.unrolling_steps
            test_steps = self.test_unrolling_steps[i] if (isinstance(self.test_unrolling_steps, ListConfig) or isinstance(self.test_unrolling_steps, list)) else self.test_unrolling_steps
            variable_dt = self.variable_dt[i] if isinstance(self.variable_dt, ListConfig) else self.variable_dt
            test_variable_dt = self.test_variable_dt[i] if isinstance(self.test_variable_dt, ListConfig) else self.test_variable_dt
            variable_dt_stride = self.variable_dt_stride_maximum[i] if isinstance(self.variable_dt_stride_maximum, ListConfig) else self.variable_dt_stride_maximum
            variable_dt_stride = variable_dt_stride if variable_dt else 1

            test_variable_dt_stride = self.test_variable_dt_stride_maximum[i] if isinstance(self.test_variable_dt_stride_maximum, ListConfig) else self.test_variable_dt_stride_maximum
            test_variable_dt_stride = test_variable_dt_stride if test_variable_dt else 1

            # prepare transforms and datasets
            transform_train, transform_val, transform_test = get_transforms(datatype, name)
            set_train, set_val, set_test = get_datasets(datatype, name, directory, steps, test_steps, variable_dt_stride,
                                                        test_variable_dt_stride, self.normalize_data, self.normalize_const,
                                                        self.crop_size, **self.additional_kwargs)

            # add metadata
            loading_metadata_train = {"type": datatype, "unrolling_steps": steps, "dataset_idx": i}
            loading_metadata_test = {"type": datatype, "unrolling_steps": test_steps, "dataset_idx": i}

            meta_train = MetadataDataset(set_train, loading_metadata_train, transform_train)
            meta_val = MetadataDataset(set_val, loading_metadata_train, transform_val)
            meta_test = MetadataDataset(set_test, loading_metadata_test, transform_test)

            self.subsets_train += [meta_train]
            self.subsets_val += [meta_val]
            self.subsets_test += [meta_test]

        self._prepare_gpu_split_and_cache("fit", ConcatDatasetDifferentShapes(self.subsets_train,
                                                                              self.different_resolution_strategy,
                                                                              downsample_factor=self.downsample_factor,
                                                                              max_channels=self.max_channels,
                                                                              max_constants=self.max_constants,
                                                                              target_size=self.target_size))
        self._prepare_gpu_split_and_cache("validate", ConcatDatasetDifferentShapes(self.subsets_val,
                                                                                   self.different_resolution_strategy,
                                                                                   downsample_factor=self.downsample_factor,
                                                                                   max_channels=self.max_channels,
                                                                                   max_constants=self.max_constants,
                                                                                   target_size=self.target_size))
        self._prepare_gpu_split_and_cache("test", ConcatDatasetDifferentShapes(self.subsets_test,
                                                                               self.different_resolution_strategy,
                                                                               downsample_factor=self.downsample_factor,
                                                                               max_channels=self.max_channels,
                                                                               max_constants=self.max_constants,
                                                                               target_size=self.target_size))


    def _prepare_gpu_split_and_cache(self, stage:str, dataset:Dataset, after_cache_transforms:Transform = None):
        r'''
        Creates split indices for each GPU and loads the dataset to the cache according to the cache strategy.
        Does not reload the cache if already loaded. Supplied dataset should match the stage.
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
        if self.shuffle_train_set:
            sampler = SubsetRandomSampler(self.subset_indices_train)
        else:
            sampler = SubsetSequentialSampler(self.subset_indices_train)

        return DataLoader(
            self.set_train,
            # prefetch_factor=self.prefetch_factor,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers = 0 if self.cache_strategy in ["all"] else self.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.set_val,
            # prefetch_factor=self.prefetch_factor,
            batch_size=self.batch_size,
            sampler=SubsetSequentialSampler(self.subset_indices_val),
            num_workers = 0 if self.cache_strategy in ["testAndVal", "all"] else self.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.set_test,
            # prefetch_factor=self.prefetch_factor,
            batch_size=self.batch_size,
            sampler=SubsetSequentialSampler(self.subset_indices_test),
            num_workers = 0 if self.cache_strategy in ["testOnly", "testAndVal", "all"] else self.num_workers)


# Extend ConcatDataset to adjust spatial and channel dimensions of concatenated datasets at their metadata
class ConcatDatasetDifferentShapes(ConcatDataset[MetadataDataset]):
    r"""Dataset as a concatenation of multiple MetadataDatasets.

    This class is used to assemble different existing datasets with different spatial or channel shapes.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """

    datasets: Iterable[MetadataDataset]
    cumulative_sizes: List[int]

    def __init__(self, datasets: Iterable[MetadataDataset], different_resolution_strategy: str,
                 downsample_factor: int, max_channels: int = 1, max_constants: int = 5, target_size: Optional[Tuple[int]] = None) -> None:
        super().__init__(datasets)

        assert(different_resolution_strategy in ["none", "rescale", "crop"])
        self.different_resolution_strategy = different_resolution_strategy

        self.downsample_factor = downsample_factor

        self.dimension = datasets[0].dimension
        # compute statistics across datasets
        self.min_channels = math.inf
        self.max_channels = max_channels

        self.min_constants = math.inf
        self.max_constants = max_constants
        self.min_resolution = [math.inf] * self.dimension
        self.max_resolution = [0] * self.dimension

        for dataset in datasets:
            if dataset.dimension != self.dimension:
                raise ValueError("Concatenating datasets with different dimensions is not supported!")

            self.min_channels = min(self.min_channels, dataset.num_channels)
            self.max_channels = max(self.max_channels, dataset.num_channels)

            self.min_constants = min(self.min_constants, dataset.num_constants)
            self.max_constants = max(self.max_constants, dataset.num_constants)

            for i in range(dataset.dimension):
                self.min_resolution[i] = min(self.min_resolution[i], dataset.resolution[i])
                self.max_resolution[i] = max(self.max_resolution[i], dataset.resolution[i])

        if target_size is None:
            if self.different_resolution_strategy == "none":
                self.target_size = None
            elif self.different_resolution_strategy == "rescale":
                self.target_size = self.max_resolution
            elif self.different_resolution_strategy == "crop":
                self.target_size = self.min_resolution
            else:
                raise ValueError(f"Unknown resolution strategy: {self.different_resolution_strategy}")

        else:
            self.target_size = target_size
            self.target_size = tuple(self.target_size)

        # assign target sizes from statistics
        self.target_channels = self.max_channels

        self.target_constants = self.max_constants

    def process_sample(self, sample, add_batch_dim=False):

        data = sample["data"]
        constants_norm = sample["constants_norm"]
        constants = sample["constants"]
        time_step_stride = sample["time_step_stride"].long()
        metadata = sample["physical_metadata"]
        loading_metadata = sample["loading_metadata"]

        # loading_metadata["dataset_idx"] = torch.Tensor([loading_metadata["dataset_idx"]]).long()

        # some datasets contain a larger number of dimensions in the domain extent -> TODO fix this in the underlying h5py file
        if self.dimension == 2:
            metadata['Domain Extent'] = metadata['Domain Extent'][:2]

        # downsampling
        if self.downsample_factor > 1:
            # average pooling
            data = torch.nn.functional.avg_pool2d(data, self.downsample_factor)

        # 2D spatial adjustment
        if self.dimension == 2 and self.target_size is not None:

            # shape already matches
            if data.shape[2] == self.target_size[0] and data.shape[3] == self.target_size[1]:
                pass

            # data too small -> bilinear interpolation
            elif data.shape[2] <= self.target_size[0] and data.shape[3] <= self.target_size[1]:
                data = torch.nn.functional.interpolate(data, size=self.target_size, mode="bilinear",
                                                       align_corners=False)

            # data too large -> random crop
            elif data.shape[2] >= self.target_size[0] and data.shape[3] >= self.target_size[1]:
                start = (
                    torch.randint(0, data.shape[2] - self.target_size[0] + 1, (1,)).item(),
                    torch.randint(0, data.shape[3] - self.target_size[1] + 1, (1,)).item(),
                )
                end = (
                    start[0] + self.target_size[0],
                    start[1] + self.target_size[1],
                )

                data = data[:, :, start[0]:end[0], start[1]:end[1]]

                if start[0] > 0:
                    metadata["Boundary Conditions"] = update_boundary_condition(metadata["Boundary Conditions"], "open",
                                                                                "x negative")
                if end[0] < data.shape[2]:
                    metadata["Boundary Conditions"] = update_boundary_condition(metadata["Boundary Conditions"], "open",
                                                                                "x positive")
                if start[1] > 0:
                    metadata["Boundary Conditions"] = update_boundary_condition(metadata["Boundary Conditions"], "open",
                                                                                "y negative")
                if end[1] < data.shape[3]:
                    metadata["Boundary Conditions"] = update_boundary_condition(metadata["Boundary Conditions"], "open",
                                                                                "y positive")

            else:
                raise NotImplementedError(
                    "Interpolating some dimensions while cropping others is currently not implemented.")

        # 3D spatial adjustment
        elif self.dimension == 3 and self.target_size is not None:
            # shape already matches
            if data.shape[2] == self.target_size[0] and data.shape[3] == self.target_size[1] and data.shape[4] == \
                    self.target_size[2]:
                pass

            # data too small -> trilinear interpolation
            elif data.shape[2] <= self.target_size[0] and data.shape[3] <= self.target_size[1] and data.shape[4] <= \
                    self.target_size[2]:

                print('Data ', data.shape)

                data = torch.nn.functional.interpolate(data, size=(data.shape[1],) + self.target_size, mode="trilinear",
                                                       align_corners=False)

            # data too large -> random crop
            elif data.shape[2] >= self.target_size[0] and data.shape[3] >= self.target_size[1] and data.shape[4] >= \
                    self.target_size[2]:
                start = (
                    torch.randint(0, data.shape[2] - self.target_size[0] + 1, (1,)).item(),
                    torch.randint(0, data.shape[3] - self.target_size[1] + 1, (1,)).item(),
                    torch.randint(0, data.shape[4] - self.target_size[2] + 1, (1,)).item(),
                )
                end = (
                    start[0] + self.target_size[0],
                    start[1] + self.target_size[1],
                    start[2] + self.target_size[2],
                )

                data = data[:, :, start[0]:end[0], start[1]:end[1], start[2]:end[2]]

                if start[0] > 0:
                    metadata["Boundary Conditions"] = update_boundary_condition(metadata["Boundary Conditions"], "open",
                                                                                "x negative")
                if end[0] < data.shape[2]:
                    metadata["Boundary Conditions"] = update_boundary_condition(metadata["Boundary Conditions"], "open",
                                                                                "x positive")
                if start[1] > 0:
                    metadata["Boundary Conditions"] = update_boundary_condition(metadata["Boundary Conditions"], "open",
                                                                                "y negative")
                if end[1] < data.shape[3]:
                    metadata["Boundary Conditions"] = update_boundary_condition(metadata["Boundary Conditions"], "open",
                                                                                "y positive")
                if start[2] > 0:
                    metadata["Boundary Conditions"] = update_boundary_condition(metadata["Boundary Conditions"], "open",
                                                                                "z negative")
                if end[2] < data.shape[4]:
                    metadata["Boundary Conditions"] = update_boundary_condition(metadata["Boundary Conditions"], "open",
                                                                                "z positive")

            else:
                raise NotImplementedError(
                    "Interpolating some dimensions while cropping others is currently not implemented.")

        elif self.dimension not in [2, 3]:
            raise ValueError("Datasets with dimensionality %d are not supported." % self.dimension)

        # pad channels and field metadata
        if data.shape[1] == self.target_channels:
            pass
        elif data.shape[1] < self.target_channels:
            pad = torch.zeros(data.shape[0], self.target_channels - data.shape[1], *data.shape[2:])
            data = torch.cat([data, pad], 1)

            pad_fields = torch.zeros(self.target_channels - len(metadata["Fields"]))
            metadata["Fields"] = torch.cat([metadata["Fields"], pad_fields], 0)
        else:
            raise NotImplementedError("Reducing the number of channels is currently not implemented.")

        # pad constants and constant metadata
        if constants.shape[0] == self.target_constants:
            pass
        elif constants.shape[0] < self.target_constants:
            pad = torch.zeros(self.target_constants - constants.shape[0])
            constants = torch.cat([constants, pad], 0)

            pad_norm = torch.zeros(self.target_constants - constants_norm.shape[0])
            constants_norm = torch.cat([constants_norm, pad_norm], 0)

            pad_constants = torch.zeros(self.target_constants - len(metadata["Constants"]))
            metadata["Constants"] = torch.cat([metadata["Constants"], pad_constants], 0)
        else:
            raise NotImplementedError("Reducing the number of constants is currently not implemented.")

        # convert metadata to correct data type -> not doing this gives errors for multiple workers
        metadata["Domain Extent"] = metadata["Domain Extent"].float()
        metadata["Dimension"] = metadata["Dimension"].long()
        metadata["PDE"] = metadata["PDE"].long()
        metadata["Fields"] = metadata["Fields"].long()
        metadata["Constants"] = metadata["Constants"].long()
        metadata["Boundary Conditions"] = metadata["Boundary Conditions"].long()

        loading_metadata["dataset_idx"] = torch.Tensor([loading_metadata["dataset_idx"]]).long()
        loading_metadata["unrolling_steps"] = torch.Tensor([loading_metadata["unrolling_steps"]]).long()

        if add_batch_dim:
            data = data.unsqueeze(0)
            constants = constants.unsqueeze(0)
            constants_norm = constants_norm.unsqueeze(0)
            time_step_stride = time_step_stride.unsqueeze(0)

            metadata["Domain Extent"] = metadata["Domain Extent"].unsqueeze(0)
            metadata["Dimension"] = metadata["Dimension"].unsqueeze(0)
            metadata["PDE"] = metadata["PDE"].unsqueeze(0)
            metadata["Fields"] = metadata["Fields"].unsqueeze(0)
            metadata["Constants"] = metadata["Constants"].unsqueeze(0)
            metadata["Boundary Conditions"] = metadata["Boundary Conditions"].unsqueeze(0)

            loading_metadata["dataset_idx"] = loading_metadata["dataset_idx"].unsqueeze(0)
            loading_metadata["unrolling_steps"] = loading_metadata["unrolling_steps"].unsqueeze(0)

        return {
            "data": data,
            "constants_norm": constants_norm,
            "constants": constants,
            "time_step_stride": time_step_stride,
            "physical_metadata": metadata,
            "loading_metadata": loading_metadata,
        }

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        return self.process_sample(sample)


    def __len__(self):
        return super().__len__()

def get_subdatasets_from_dataloader(dataloader:torch.utils.data.DataLoader) -> (
        Tuple)[Iterable[MetadataDataset], ConcatDatasetDifferentShapes]:

    dataset = dataloader.dataset

    while not isinstance(dataset, ConcatDatasetDifferentShapes):
        if isinstance(dataset, torch.utils.data.Subset) or isinstance(dataset, CachedDataset):
            dataset = dataset.dataset
        else:
            raise ValueError("Dataset {} is not supported".format(type(dataset)))
    subsets = dataset.datasets
    return subsets, dataset


