from typing import Optional

import torch
from torch.utils.data import random_split
from torchvision.transforms.v2 import Transform, ToDtype, Compose, Lambda, Normalize, RandomHorizontalFlip, RandomVerticalFlip
from ..pbdl_dataloader.dataset import Dataset as PBDLDataset
from typing import Sequence
from ..pbdl_dataloader.utilities import get_sel_const_sim
import numpy as np

from .variable_dt_dataset import VariableDtDataset
'''
    2d datasets downloaded from The Well from Ohana et al.: The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning.
'''
seed = 48

class ModifiedPBDLDataset(PBDLDataset):
    
    def __init__(self, *args, sel_channels=None, **kwargs):

        super().__init__(*args, **kwargs)
        self.sel_channels = sel_channels

    def __getitem__(self, idx):
        """
        The data provided has the shape (channels, spatial dims...).

        Returns:
            numpy.ndarray: Input data (without constants)
            tuple: Constants
            numpy.ndarray: Target data
            tuple: Non-normalized constants (only if solver flag is set)
        """
        if idx >= len(self):
            raise IndexError

        # create input-target pairs with interval time_steps from simulation steps
        if self.sel_sims:
            sim_idx = self.sel_sims[idx // self.samples_per_sim]
        else:
            sim_idx = idx // self.samples_per_sim

        sim = self.dset["sims/sim" + str(sim_idx)]
        const = get_sel_const_sim(self.dset, sim_idx, self.sel_const)

        input_frame_idx = (
            self.trim_start + (idx % self.samples_per_sim) * self.step_size
        )
        target_frame_idx = input_frame_idx + self.time_steps
        
        input = sim[input_frame_idx]
        if self.intermediate_time_steps:
            target = sim[input_frame_idx + 1 : target_frame_idx + 1]
        else:
            target = sim[target_frame_idx]
        const_nnorm = const

        # normalize
        if self.norm_strat_data:
            input = self.norm_strat_data.normalize(input)

            if self.intermediate_time_steps:
                target = np.array(
                    [self.norm_strat_data.normalize(frame) for frame in target]
                )
            else:
                target = self.norm_strat_data.normalize(target)

        if self.norm_strat_const:
            const = self.norm_strat_const.normalize(const)
            
        if self.sel_channels is not None:
            input = input[self.sel_channels]
            if self.intermediate_time_steps:
                target = target[:,self.sel_channels]
            else:
                target = target[self.sel_channels]
        
        return (
            input,
            target,
            tuple(const),  # required by loader
            tuple(const_nnorm),  # needed by pbdl.torch.phi.loader
        )
        

def well_2d_datasets(dataset_name: str,
                 dataset_directory: str,
                 unrolling_steps: int,
                 intermediate_time_steps: bool = True,
                 variable_dt_stride_maximum: int = 1,
                 test_variable_dt_stride_maximum: int = 1,
                 test_unrolling_steps: Optional[int] = None,
                 test_intermediate_time_steps: Optional[bool] = None,
                 normalize_data: Optional[str] = None,
                 normalize_const: Optional[str] = None,
                **kwargs) -> tuple[PBDLDataset, PBDLDataset, PBDLDataset]:
    r'''
    Creates 2D well train, val, and test dataset objects.

    Args:
        dataset_name: name of local dataset file
        dataset_directory: directory where the data set is located
        unrolling_steps: number of time steps between start and end of sequence
        intermediate_time_steps: determines if intermediate time steps are included in the data or not
        variable_dt_stride_maximum: maximum number of time steps between start and end of sequence for variable dt training
        test_variable_dt_stride_maximum: maximum number of time steps between start and end of sequence for variable dt testing
        test_unrolling_steps: number of time steps between start and end of sequence for the test dataset
        test_intermediate_time_steps: determines if intermediate time steps are included in the data
                                        or not for the test dataset
        normalize_data: type of normalization to apply to the data (mean-std, std, zero-to-one, minus-one-to-one, None)
        normalize_const: type of normalization to apply to the constants (mean-std, std, zero-to-one, minus-one-to-one, None)

    Returns:
        tuple[PBDLDataset, PBDLDataset, PBDLDataset]: train, validation and test datasets
    '''

    if test_unrolling_steps is None:
        test_unrolling_steps = unrolling_steps
    if test_intermediate_time_steps is None:
        test_intermediate_time_steps = intermediate_time_steps
    if test_variable_dt_stride_maximum is None:
        test_variable_dt_stride_maximum = variable_dt_stride_maximum

    params_train = {
        "dset_name": dataset_name + "_train",
        "local_datasets_dir": dataset_directory,
        "time_steps": unrolling_steps,
        "intermediate_time_steps": intermediate_time_steps,
        "normalize_const": normalize_const,
        "normalize_data": normalize_data,
    }

    params_val = {
        "dset_name": dataset_name + "_valid",
        "local_datasets_dir": dataset_directory,
        "time_steps": unrolling_steps,
        "intermediate_time_steps": intermediate_time_steps,
        "normalize_const": normalize_const,
        "normalize_data": normalize_data,
    }

    params_test = {
        "dset_name": dataset_name + "_test",
        "local_datasets_dir": dataset_directory,
        "time_steps": test_unrolling_steps,
        "intermediate_time_steps": test_intermediate_time_steps,
        "normalize_const": normalize_const,
        "normalize_data": normalize_data,
    }
    
    for key,item in kwargs.items():
        if key.endswith('_train'):
            params_train[key[0:-len('_train')]] = item
        elif key.endswith('_val'):
            params_val[key[0:-len('_val')]] = item
        elif key.endswith('_test'):
            params_test[key[0:-len('_test')]]= item

    print('Params train ', params_train)

    if variable_dt_stride_maximum <= 1:
        train = ModifiedPBDLDataset(**params_train)
        val = ModifiedPBDLDataset(**params_val)
    else:
        train = VariableDtDataset(**params_train, maximum_dt=variable_dt_stride_maximum, seed=None)
        val = VariableDtDataset(**params_val, maximum_dt=variable_dt_stride_maximum, seed=None)
    if test_variable_dt_stride_maximum <= 1:
        test = ModifiedPBDLDataset(**params_test)
    else:
        test = VariableDtDataset(**params_test, maximum_dt=test_variable_dt_stride_maximum, seed=seed)


    return train, val, test



def well_2d_transforms(dataset_name: str)-> tuple[Transform, Transform, Transform]:
    r'''
    Creates 2d well train, val, and test transform objects.

    Args:
        dataset_name: name of local dataset

    Returns:
        tuple[Transform, Transform, Transform]: train, validation and test transforms
    '''

    transform_train = Compose(
        [
            ToDtype(torch.float32),
        ]
    )

    return transform_train, transform_train, transform_train


