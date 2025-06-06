from typing import Optional, List
import os
import copy
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset, Subset
from pbdl.torch.dataset import Dataset as PBDLDataset

from .pbdl_datatypes.variable_dt_dataset import VariableDtDataset
from .metadata_remapping import convert_pde, convert_fields, convert_constants
from .metadata_remapping import convert_domain_extent, convert_dt, convert_reynolds_number, convert_boundary_conditions


def unpack_pbdl_data_format(pbdl_data:tuple, variable_dt_format:bool = False) -> list:
    r'''
    Unpacks the data format from the PBDLDataset into separate tensors.

    Args:
        pbdl_data: data in the format from PBDLDataset
        variable_dt_format: determines if the data is in the variable dt format

    Returns:
        list: unpacked data format
    '''

    if variable_dt_format:
        conditioning, data, simulation_constants_norm, simulation_constants, time_step_stride = pbdl_data
    else:
        conditioning, data, simulation_constants_norm, simulation_constants = pbdl_data

    conditioning = torch.from_numpy(conditioning)
    data = torch.from_numpy(data)
    simulation_constants_norm = torch.tensor(simulation_constants_norm)
    simulation_constants = torch.tensor(simulation_constants)

    if len(conditioning.shape) == 4:
        conditioning = torch.unsqueeze(conditioning, 0)

    if len(conditioning.shape) < len(data.shape):
        conditioning = conditioning.unsqueeze(0)

    concat = torch.concatenate([conditioning, data], 0)

    time_step_stride = torch.tensor(time_step_stride) if variable_dt_format else torch.tensor([1])

    return concat, simulation_constants_norm, simulation_constants, time_step_stride



class MetadataDataset(Dataset):
    r'''
    Unpacks the underlying PBDL dataset, applies different data transformations and adds metadata.

    Args:
        dataset: pbdl dataset to use
        loading_metadata: metadata from loading the pbdl dataset
        transforms: optional data transformations to be applied to the data
    '''
    def __init__(self, dataset:PBDLDataset|VariableDtDataset|Subset, loading_metadata:dict, transforms=None):
        super().__init__()
        self.dataset = dataset
        self.loading_metadata = loading_metadata
        self.transforms = transforms

        # load physical metadata from h5py file
        if isinstance(dataset, (PBDLDataset, VariableDtDataset)):
            h5py_path = dataset.dset_file
        elif isinstance(dataset, Subset):
            h5py_path = dataset.dataset.dset_file
        else:
            raise TypeError("Unknown dataset type")

        if not os.path.isfile(h5py_path):
            raise FileNotFoundError(f"File {h5py_path} not found.")

        metadata = {}
        with h5py.File(h5py_path, "r") as h5py_file:
            for key in h5py_file["sims"].attrs.keys():
                val = h5py_file["sims"].attrs[key]
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                metadata[key] = val

        if isinstance(metadata["Boundary Conditions"][0], list):
            metadata["Boundary Conditions"] = metadata["Boundary Conditions"][0]

        self.dimension = metadata["Dimension"]
        #self.num_channels = len(metadata["Fields"])
        self.num_channels = dataset[0][0].shape[0]
        if hasattr(dataset, "sel_channels"):
            if dataset.sel_channels is not None:
                metadata["Fields"] = [metadata["Fields"][i] for i in dataset.sel_channels]
        if hasattr(dataset, "sel_constants"):
            if dataset.sel_constants is not None:
                metadata["Constants"] = [metadata["Constants"][i] for i in dataset.sel_constants]
        #self.num_constants = len(metadata["Constants"])
        self.num_constants = len(dataset[0][2])

        # adjust resolution format
        if isinstance(metadata["Resolution"], list):
            self.resolution = metadata["Resolution"]
        elif isinstance(metadata["Resolution"], (int, np.integer)):
            self.resolution = [metadata["Resolution"]] * self.dimension
        else:
            raise ValueError("Resolution %s must be a list or an integer but has type %s" % (metadata["Resolution"], type(metadata["Resolution"])))

        # adjust domain extent format to list
        if not "Domain Extent" in metadata:
            metadata["Domain Extent"] = [0] * self.dimension
        elif isinstance(metadata["Domain Extent"], list):
            pass
        elif isinstance(metadata["Domain Extent"], (int, np.integer, np.floating)):
            metadata["Domain Extent"] = [metadata["Domain Extent"]] * self.dimension
        else:
            raise ValueError("Domain Extent %s must be a list or a number but has type %s" % (metadata["Domain Extent"], type(metadata["Domain Extent"])))

        # adjust reynolds number format to float
        if not "Reynolds Number" in metadata:
            metadata["Reynolds Number"] = 0.0

        self.physical_metadata = {
            "Dimension": torch.tensor([self.dimension]),
            "PDE": convert_pde(metadata["PDE"]),
            "Fields": convert_fields(metadata["Fields"]),
            "Constants": convert_constants(metadata["Constants"]),
            "Boundary Conditions": convert_boundary_conditions(metadata["Boundary Conditions"], metadata["Boundary Conditions Order"]),
            "Dt": convert_dt(metadata["Dt"]) if "Dt" in metadata else torch.Tensor([0]),
            "Domain Extent": convert_domain_extent(metadata["Domain Extent"]),
            "Reynolds Number": convert_reynolds_number(metadata["Reynolds Number"]),
        }


    def __getitem__(self, index):
        from_loader = self.dataset[index]

        if isinstance(self.dataset, Subset):
            variable_dt_format = isinstance(self.dataset.dataset, VariableDtDataset)
        else:
            variable_dt_format = isinstance(self.dataset, VariableDtDataset)

        data, constants_norm, constants, time_step_stride = unpack_pbdl_data_format(from_loader, variable_dt_format)
        constants_norm = constants_norm.float()
        constants = constants.float()
        time_step_stride = time_step_stride.float()

        if self.transforms:
            data = self.transforms(data)

        return {
            "data": data,
            "constants_norm": constants_norm,
            "constants": constants,
            "time_step_stride": time_step_stride,
            "physical_metadata": copy.deepcopy(self.physical_metadata), # copy in case models or other dataset make changes to these tensors
            "loading_metadata": copy.deepcopy(self.loading_metadata), # copy in case models or other dataset make changes to these tensors
        }


    def __len__(self):
        return len(self.dataset)
