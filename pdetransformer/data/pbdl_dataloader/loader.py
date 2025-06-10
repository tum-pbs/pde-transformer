import numpy as np
from multiprocessing import Pool

from .dataset import Dataset

def _collate_fn_(batch):
    """
    Concatenates data arrays with inflated constant layers and stacks them into batches.

    Returns:
        numpy.ndarray: Data batch array
        numpy.ndarray: Target batch array
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

    return data, targets


class Dataloader:
    def __init__(
        self,
        *args,
        batch_size=1,
        shuffle=True,
        collate_fn=_collate_fn_,
        num_workers=1,
        **kwargs,
    ):

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.num_workers = num_workers

        self.dataset = Dataset(*args, **kwargs)

    def __iter__(self):
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0

        if self.num_workers > 1:
            pass  # TODO dataset must support multiprocessing first
            # return self.__iter_multiprocessing__()
        else:
            return self

    def __next__(self):
        if self.current_idx >= len(self.indices):
            raise StopIteration
        batch_indices = self.indices[
            self.current_idx : self.current_idx + self.batch_size
        ]
        batch = [self.dataset[idx] for idx in batch_indices]
        self.current_idx += self.batch_size
        if self.collate_fn:
            return self.collate_fn(batch)
        return batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def get_sim_raw(self, sim):
        return self.dataset.get_sim_raw(sim)
    
    def info(self):
        return self.dataset.info()


    # parallel processing
    def __iter_multiprocessing__(self):
        with Pool(self.num_workers) as pool:
            batch_indices = [
                self.indices[i : i + self.batch_size]
                for i in range(0, len(self.indices), self.batch_size)
            ]
            for batch in pool.imap(self.__load_data__, batch_indices):
                if self.collate_fn:
                    yield self.collate_fn(batch)
                else:
                    yield batch

    def __load_data__(self, indices):
        return [self.dataset[idx] for idx in indices]
