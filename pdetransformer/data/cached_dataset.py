import math

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .utils import SubsetSequentialSampler


class CachedDataset(Dataset):
    r'''
    Caches items from the given dataset and applies optional data transformations.

    ### IMPORTANT NOTE:
    Adjust max_cache_size if dataset does not fully fit into memory to prevent memory issues.

    Args:
        dataset (Dataset): dataset to use
        transforms: optional transform to be applied to the data, especially for random operations that shouldn't be cached
        max_cached_size (int): optional maximum number of cached items. All items are cached by default.
    '''

    def __init__(self, dataset:Dataset, transforms=None, max_cache_size:int=math.inf):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.max_cache_size = max_cache_size
        self.cache = {}


    def __getitem__(self, index):

        # load from cache
        if index in self.cache:
            x = self.cache[index]

        # load from disk and store in cache if not full
        else:
            x = self.dataset[index]

            # store in cache
            if len(self.cache) < self.max_cache_size:
                self.cache[index] = x
            else:
                # max cache size is reached but removing older samples does not help for full training epochs anyway
                # -> keeping only the first max_cache_size samples is the best solution for now
                pass

        # transforms are not cached!
        if self.transforms:
            x = self.transforms(x)

        return x


    def __len__(self):
        return len(self.dataset)


    def fill_cache_sequentially(self, subset_indices:list):
        self.subset_indices = subset_indices
        loader = DataLoader(self, batch_size=64, sampler=SubsetSequentialSampler(self.subset_indices), num_workers=0)
        for i,_ in tqdm(enumerate(loader)):
            if i > self.max_cache_size:
                break

