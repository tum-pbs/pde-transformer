from typing import Optional

import torch
from torch.utils.data import random_split
from torchvision.transforms.v2 import Transform, ToDtype, Compose, Lambda, Normalize, RandomHorizontalFlip, RandomVerticalFlip
from ..pbdl_dataloader.dataset import Dataset as PBDLDataset

from .variable_dt_dataset import VariableDtDataset


'''
    Datasets from Kohl et al.: Benchmarking Autoregressive Conditional Diffusion Models for Turbulent Flow Simulation.
    https://github.com/tum-pbs/autoreg-pde-diffusion
'''
seed = 42


def acdm_2d_datasets(dataset_name: str,
                 dataset_directory: str,
                 unrolling_steps: int,
                 intermediate_time_steps: bool = True,
                 variable_dt_stride_maximum: int = 1,
                 test_variable_dt_stride_maximum: int = 1,
                 test_unrolling_steps: Optional[int] = None,
                 test_intermediate_time_steps: Optional[bool] = None,
                 normalize_data: Optional[str] = None,
                 normalize_const: Optional[str] = None) -> tuple[PBDLDataset, PBDLDataset, PBDLDataset]:
    r'''
    Creates ACDM train, val, and test dataset objects.

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

    # use different simulations for training and testing
    if dataset_name == "incompressible-wake-flow":
        train_sims = list(range(2,44)) + list(range(46,89))
        test_sims = [0,1,44,45,89,90]
    elif dataset_name == "transonic-cylinder-flow":
        train_sims = list(range(2,20)) + list(range(22,39))
        test_sims = [0,1,20,21,39,40]
    elif dataset_name == "isotropic-turbulence":
        train_sims = list(range(0,425)) + list(range(575,1000))
        test_sims = range(425,575)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    params_train = {
        "dset_name": dataset_name,
        "local_datasets_dir": dataset_directory,
        "sel_sims": train_sims,
        "time_steps": unrolling_steps,
        "intermediate_time_steps": intermediate_time_steps,
        "normalize_const": normalize_const,
        "normalize_data": normalize_data,
    }

    if variable_dt_stride_maximum <= 1:
        pbdl_all = PBDLDataset(**params_train)
    else:
        pbdl_all = VariableDtDataset(**params_train, maximum_dt=variable_dt_stride_maximum, seed=None)

    train, val = random_split(
        pbdl_all, [0.85, 0.15], generator=torch.Generator().manual_seed(seed)
    )
    # the data indices have to be sorted manually since random_split shuffles them
    train.indices = sorted(train.indices)
    val.indices = sorted(val.indices)


    params_test = {
        "dset_name": dataset_name,
        "local_datasets_dir": dataset_directory,
        "sel_sims": test_sims,
        "time_steps": test_unrolling_steps,
        "intermediate_time_steps": test_intermediate_time_steps,
        "normalize_const": normalize_const,
        "normalize_data": normalize_data,
    }

    if test_variable_dt_stride_maximum <= 1:
        test = PBDLDataset(**params_test)
    else:
        test = VariableDtDataset(**params_test, maximum_dt=test_variable_dt_stride_maximum, seed=seed)

    return train, val, test



def acdm_2d_transforms(dataset_name: str)-> tuple[Transform, Transform, Transform]:
    r'''
    Creates ACDM train, val, and test transform objects.

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

