from torch.utils.data import Dataset
import bisect
from typing import Sequence, Callable
import copy
import numpy as np

from .dataset import PBDLUDataset

class MetadataDataset(Dataset):
    r'''
    A dataset wrapper that adds the metadata to each data sample from the underlying PBDLU dataset.
    '''

    def __init__(self, dataset: PBDLUDataset, transforms: Sequence[Callable] | None = None) -> None:
        """ Constructor for the MetadataDataset.

        Args:
            dataset (PBDLUDataset): The underlying PBDLU dataset to wrap.
            transforms (Sequence[Callable] | None, optional): A sequence of transformations to apply to each data sample. Defaults to None. 
                Be aware of the layout of the data when using transforms here.
        """
        
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Provided data sample at index `idx`.

        Returns:
            dict: A dictionary containing:
                "data": The data as returned by `PBDLUDataset` at index `idx`.
                "metadata": A deep copy of the metadata dictionary from the underlying dataset.
        """
        data = self.dataset[idx]

        if self.transforms is not None:
            for transform in self.transforms:
                data = transform(data)

        metadata = copy.deepcopy(self.dataset.get_meta_data())

        return {
            "data": data,
            "metadata": metadata
        }


class ConcatDatasetDifferentShapes(Dataset):
    r'''
    A dataset that concatenates multiple MetadataDatasets, even if they have different data shapes.
    When accessing a data sample, it pads the data to match the maximum shape across all datasets.
    It also returns the index of the original dataset from which the sample was taken.
    The metadata can then be accessed via the dataset index.
    '''
    
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Sequence[MetadataDataset]) -> None:
        """ Constructor for the ConcatDatasetDifferentShapes.

        Args:
            datasets (Sequence[MetadataDataset]): A sequence of MetadataDatasets to concatenate.
        """
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)
        
        self.max_num_const = 0
        self.max_num_points = 0
        self.max_num_in_fields = 0
        self.max_num_out_fields = 0
        
        first_metadata = datasets[0].dataset.get_meta_data()
        self.num_spatial_dims = first_metadata["num_spatial_dim"]
        self.time_steps = datasets[0].dataset.time_steps
        
        for dataset in datasets:
            metadata = dataset.dataset.get_meta_data()
            
            if metadata["num_spatial_dim"] != self.num_spatial_dims:
                raise ValueError("All datasets must have the same number of spatial dimensions.")
            if dataset.dataset.time_steps != self.time_steps:
                raise ValueError("All datasets must have the same number of time steps per sample.")
            
            self.max_num_const = max(self.max_num_const, metadata["num_const"])
            self.max_num_points = max(self.max_num_points, metadata["num_points"])
            self.max_num_in_fields = max(self.max_num_in_fields, metadata["num_in_fields"])
            self.max_num_out_fields = max(self.max_num_out_fields, metadata["num_out_fields"])
            
    def __len__(self):
        return self.cumulative_sizes[-1]
        
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("Index out of range")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        sample = self.datasets[dataset_idx][sample_idx]
        
        data = sample["data"]
        metadata = sample["metadata"]
        
        x, y, fx, const = data
        
        # pad -1 dimension of x, y, fx to max_num_points
        # pad -2 dimension of y to max_num_out_fields
        # pad -2 dimension of fx to max_num_in_fields
        # pad -1 dimension of const to max_num_const 
        num_pad_points = self.max_num_points - x.shape[-1]
        num_pad_out_fields = self.max_num_out_fields - y.shape[-2]
        num_pad_in_fields = self.max_num_in_fields - fx.shape[-2]
        num_pad_const = self.max_num_const - const.shape[-1]
        
        x = np.pad(x, ((0, 0), (0, 0), (0, num_pad_points)), mode='constant', constant_values=0.0)
        y = np.pad(y, ((0, 0), (0, num_pad_out_fields), (0, num_pad_points)), mode='constant', constant_values=0.0)
        fx = np.pad(fx, ((0, 0), (0, num_pad_in_fields), (0, num_pad_points)), mode='constant', constant_values=0.0)
        const = np.pad(const, ((0, num_pad_const)), mode='constant', constant_values=0.0)

        return {
            "data": (x, y, fx, const),
            "dataset_idx": dataset_idx
        }
        
    def get_metadata_from_index(self, dataset_idx: int):
        """ Get the metadata of the dataset at the given index.

        Args:
            dataset_idx (int): The index of the dataset.

        Returns:
            dict: The metadata dictionary of the dataset.
        """
        return self.datasets[dataset_idx].dataset.get_meta_data()