import numpy as np
import pbdl.torch.phi.dataset
import torch.utils.data
import random

import pbdl.torch.phi.dataset
from pbdl.logging import info, success, warn, fail, corrupt

class ConstantBatchSampler(torch.utils.data.BatchSampler):
    def __init__(
        self,
        dataset: pbdl.torch.phi.dataset.Dataset,
        batch_size=None,
        batch_by_const=None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_constants = batch_by_const

        self.groups = self.group_by_constants()

    def group_by_constants(self):
        info("Prepare constant batch sampling...")

        groups = {}
        for sim_range in self.dataset.iterate_sims():

            # the constants of the first sample are representative for the sim
            _, _, _, const = self.dataset[sim_range[0]]

            group_by_const = tuple(
                [const[i] for i in self.group_constants]
            )  # project constants onto self.group_constants

            if group_by_const not in groups:
                groups[group_by_const] = []
            groups[group_by_const].extend(sim_range)
        
        for group in groups.values():
            random.shuffle(group)

        return list(groups.values())

    def __iter__(self):
        for group in self.groups:
            for batch in [
                group[i : i + self.batch_size]
                for i in range(0, len(group), self.batch_size)
            ]:
                yield batch

    def __len__(self):
        return sum(len(group) // self.batch_size for group in self.groups)
