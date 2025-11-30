import lightning
import torch

from .pbdlu_dataloader.dataset import Dataset
from typing import Sequence

class PBDLUDataModule(lightning.LightningDataModule):
    def __init__(
        self, 
        dataset_path: str,
        batch_size: int,
        num_workers: int = 1,
        pin_memory: bool = True,
        shuffle: bool = True,
        split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
        **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_path = dataset_path
        self.dataset = Dataset(dataset_path, **kwargs)
        
        total_size = len(self.dataset)
        self.train_size = int(split_ratios[0] * total_size)
        self.val_size = int(split_ratios[1] * total_size)
        self.test_size = total_size - self.train_size - self.val_size
        
        # Generate non-overlapping random indices for splitting the dataset
        if shuffle:
            self.index_set = torch.randperm(total_size)
        else:
            self.index_set = torch.arange(total_size)
        
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        
        self.set_train = None
        self.set_val = None
        self.set_test = None
        
    def setup(self, stage: str):
        if stage == 'fit' and not self.set_train:
            self.train_indices = self.index_set[:self.train_size]
            self.set_train = torch.utils.data.Subset(self.dataset, self.train_indices)
            
        if stage == 'validate' or stage is None:
            self.val_indices = self.index_set[self.train_size:self.train_size + self.val_size]
            self.set_val = torch.utils.data.Subset(self.dataset, self.val_indices)
            
        if stage == 'test' or stage is None:
            self.test_indices = self.index_set[self.train_size + self.val_size:]
            self.set_test = torch.utils.data.Subset(self.dataset, self.test_indices)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.set_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.set_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.set_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )