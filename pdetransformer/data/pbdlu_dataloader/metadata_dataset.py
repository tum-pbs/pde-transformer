from torch.utils.data import Dataset
from typing import Sequence, Callable
import copy

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
