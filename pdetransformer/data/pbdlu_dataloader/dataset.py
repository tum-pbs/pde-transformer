import numpy as np
import os
import sys
import h5py
from h5py import Group
from itertools import groupby

from .logging import info, success, warn, fail, corrupt 

class Dataset:
    def __init__(
        self,
        dset_name,
        time_steps=None,  # by default num_frames - 1
        all_time_steps=False,  # sets time_steps=max time steps, intermediate_time_steps=True
        intermediate_time_steps=None,  # by default False
        normalize_data=None,  # by default no normalization
        normalize_const=None,  # by default no normalization
        sel_sims=None,  # if None, all simulations are loaded
        sel_const=None,  # if None, all constants are returned
        sel_channels=None, # if None, all channels are returned
        trim_start=None,  # by default 0
        trim_end=None,  # by default 0
        step_size=None,  # by default 1
        disable_progress=False,
        crop_size=None, # Not applicable for unstructured data
        seed=0,
        clear_norm_data=False,
        **kwargs,
    ):
        self.sel_sims = sel_sims
        self.sel_const = sel_const
        self.disable_progress = disable_progress
        self.crop_size = crop_size
        self.rng = np.random.default_rng(seed)
        
        dset_path = f"{dset_name}.hdf5"
        if os.path.exists(dset_path):
            self._load_dataset(dset_name, dset_path)
            self._validate_dataset()
        else:
            fail(
                f"Dataset '{dset_name}' not found."
            )
            sys.exit(0)
        
        # TODO: Improve time step handling
        if all_time_steps:
            self.time_steps = self.num_frames - 1
            self.intermediate_time_steps = True
            self.trim_start = 0
            self.trim_end = 0
            self.step_size = 1
            self.samples_per_sim = 1

            for attr, val in [
                ("time_steps", time_steps),
                ("intermediate_time_steps", intermediate_time_steps),
                ("trim_start", trim_start),
                ("trim_end", trim_end),
                ("step_size", step_size),
            ]:
                if val is not None:
                    warn(
                        f"`{attr}` is managed by `all_time_steps` and can therefore not be set manually."
                    )
        else:
            self.time_steps = time_steps or self.num_frames - 1
            self.intermediate_time_steps = intermediate_time_steps or False
            self.trim_start = trim_start or 0
            
            self.trim_end = trim_end or 0
            self.step_size = step_size or 1

            group = self.dset["sims"][f'{next(iter(self.dset["sims"]))}']
            if isinstance(group, h5py.Group):
                self.num_frames = len(group)
            else:
                self.num_frames = group.shape[0]

            self.samples_per_sim = (
                self.num_frames - self.time_steps - self.trim_start - self.trim_end
            ) // self.step_size
            
        success(
            f"Loaded { self.dset_name } with { self.num_sims } simulations "
            + (f"({len(self.sel_sims)} selected) " if self.sel_sims else "")
            + f"and {self.samples_per_sim} samples each."
        )
        
    
    def _load_dataset(self, dset_name, dset_file):
        """Loads the dataset and sets dataset specific attributes on `self`

        Args:
            dset_name (str): The name of the dataset
            dset_file (str): The path to the dataset file
        """
        self.dset_name = dset_name
        self.dset_file = dset_file
        
        # HDF5 file format
        self.dset = h5py.File(dset_file, "r")
        meta = get_meta_data(self.dset)
        
        # Set metadata as attibutes
        for key, value in meta.items():
            setattr(self, key, value)
        
    def _validate_dataset(self):
        # TODO: Validation
        pass
    
    def __len__(self):
        if self.sel_sims is not None:
            return len(self.sel_sims) * self.samples_per_sim
        else:
            return self.num_sims * self.samples_per_sim
        
    def __getitem__(self, idx):
        """
        The data provided has the shape (channels, spatial dims...).

        Returns:
            numpy.ndarray: Position data
            numpy.ndarray: Target data
            numpy.ndarray: Input feature data
            # tuple: Constants
            # tuple: Non-normalized constants (only if solver flag is set)
        """
        if idx >= len(self):
            raise IndexError
        
        

        # create input-target pairs with interval time_steps from simulation steps
        if self.sel_sims:
            sim_idx = self.sel_sims[idx // self.samples_per_sim]
        else:
            sim_idx = idx // self.samples_per_sim

        sim = self.dset["sims/sim" + str(sim_idx)]
        # const = get_sel_const_sim(self.dset, sim_idx, self.sel_const)

        input_frame_idx = (
                self.trim_start + (idx % self.samples_per_sim) * self.step_size
        )
        target_frame_idx = input_frame_idx + self.time_steps

        # dim_list = self.sim_shape[2:]

        # Note: Deleted cropping here as it is not applicable for unstructured data

        if self.intermediate_time_steps:
            target = sim[input_frame_idx + 1: target_frame_idx + 1]
        else:
            target = sim[input_frame_idx]


        # TODO: Normalization
        
        # const_nnorm = const

        # # normalize
        # if self.norm_strat_data:
        #     input = self.norm_strat_data.normalize(input)

        #     if self.intermediate_time_steps:
        #         target = np.array(
        #             [self.norm_strat_data.normalize(frame) for frame in target]
        #         )
        #     else:
        #         target = self.norm_strat_data.normalize(target)

        # if self.norm_strat_const:
        #     const = self.norm_strat_const.normalize(const)

        # if self.sel_channels is not None:
        #     input = input[self.sel_channels]
        #     if self.intermediate_time_steps:
        #         target = target[:, self.sel_channels]
        #     else:
        #         target = target[self.sel_channels]
        
        

        return (
            positions,
            targets,
            features
            # tuple(const),  # required by loader
            # tuple(const_nnorm),  # needed by pbdl.torch.phi.loader
        )
        
        
        
def get_meta_data(dset):
    # convert_key = lambda key: key.lower().replace(" ", "_")

    field_mapping = {
        "PDE": "pde",
        "Fields Scheme": "fields_scheme",
        "Fields": "fields",
        "Constants": "const",
        "Field Desc": "field_desc",
        "Constant Desc": "const_desc",
        "Dt": "dt",
    }

    meta_attrs = dset["sims"].attrs

    meta = {field_mapping[field]: meta_attrs[field] for field in field_mapping.keys() if field in meta_attrs}

    group = dset["sims"][f'{next(iter(dset["sims"]))}']
    if isinstance(group, Group):
        first_sim_x = dset["sims"][f'{next(iter(dset["sims"]))}/x']
        first_sim_y = dset["sims"][f'{next(iter(dset["sims"]))}/y']
        first_sim_fx = dset["sims"][f'{next(iter(dset["sims"]))}/fx']
        sim_shape = first_sim_x.shape
        num_spatial_dim = first_sim_x.shape[1]
    else:
        raise ValueError("Dataset for unstructured data must contain 'x', 'y', and 'fx' datasets per simulation.")

    num_fields = len(list(groupby(meta["fields_scheme"])))  # TODO

    meta.update(
        {
            "num_sims": len(dset["sims"]),
            "num_const": len(meta["const"]),
            "sim_shape": sim_shape,
            "num_frames": sim_shape[0],
            "num_sca_fields": sim_shape[1],
            "num_fields": num_fields,
            "num_spatial_dim": num_spatial_dim,
        }
    )

    return meta