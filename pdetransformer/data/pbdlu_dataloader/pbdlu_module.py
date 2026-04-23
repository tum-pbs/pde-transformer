from typing import Sequence
import lightning
import torch

from .dataset import PBDLUDataset
from .metadata_dataset import MetadataDataset

class PBDLUDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        batch_size: int,
        num_workers: int = 1,
        pin_memory: bool = True,
        shuffle: bool = True,
        split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
        include_metadata: bool = False,
        **kwargs
    ):
        """ Construct a PBDLU data module for training with Pytorch Lightning

        This datamodule constructs an underlying PBDLU dataset or MetadataDataset and splits it into training, validation, and test sets.

        Args:
            dataset_path (str): Path to the dataset file
            batch_size (int): Batch size for the data loaders
            num_workers (int, optional): Number of worker processes for data loading. Defaults to 1.
            pin_memory (bool, optional): Whether to pin memory in data loaders. Defaults to True.
            shuffle (bool, optional): Whether to shuffle the dataset before splitting. Defaults to True.
            split_ratios (Sequence[float], optional): Ratios for splitting the dataset into train, validation, and test sets. Defaults to (0.8, 0.1, 0.1).
            include_metadata (bool, optional): Whether to use MetadataDataset instead of PBDLU dataset. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the PBDLU dataset constructor.
        """

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_path = dataset_path
        if include_metadata:
            dataset = PBDLUDataset(dataset_path, **kwargs)
            self.dataset = MetadataDataset(dataset)
        else:
            self.dataset = PBDLUDataset(dataset_path, **kwargs)

        total_size = len(self.dataset)
        self.train_size = int(split_ratios[0] * total_size)
        self.val_size = int(split_ratios[1] * total_size)
        self.test_size = total_size - self.train_size - self.val_size
        
        print("Total dataset size:", total_size)
        print("Dataset split ratios:", split_ratios)
        print(f"Dataset sizes -> Train: {self.train_size}, Val: {self.val_size}, Test: {self.test_size}")

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
            self.set_train = torch.utils.data.Subset(
                self.dataset, self.train_indices)

        if stage == 'validate' or stage is None:
            self.val_indices = self.index_set[self.train_size:self.train_size + self.val_size]
            self.set_val = torch.utils.data.Subset(
                self.dataset, self.val_indices)

        if stage == 'test' or stage is None:
            self.test_indices = self.index_set[self.train_size + self.val_size:]
            self.set_test = torch.utils.data.Subset(
                self.dataset, self.test_indices)

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
