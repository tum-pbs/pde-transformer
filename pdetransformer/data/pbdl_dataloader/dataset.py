import os
import io
import sys
import json
import urllib.request
import pkg_resources
import h5py
import numpy as np
from itertools import groupby

from h5py import Group

from . import normalization as norm
from . import fetcher
from . import logging
from . import utilities

# from pbdl.colors import colors
from .logging import info, success, warn, fail, corrupt
from .utilities import get_sel_const_sim, get_meta_data, scan_local_dset_dir, get_sel_const_sim_v2

config_path = pkg_resources.resource_filename(__name__, "config.json")

# load configuration
# try:
#     with open(config_path, "r") as f:
#         print(f)
#         config = json.load(f)
# except json.JSONDecodeError:
#     raise ValueError("Invalid configuration file.")

config = {
    "hf_repo_id": "thuerey-group/pde-transformer-ape2d",
    "local_datasets_dir": "./datasets/",
    "global_dataset_dir": "./datasets/",
    "dataset_ext": ".hdf5"
}

def _load_index():
    global local_index
    global global_index

    # load local dataset index
    local_index = scan_local_dset_dir(config)
    global_index = fetcher.fetch_index(config)


def index():
    return global_index | local_index


def datasets():
    return list((global_index | local_index).keys())


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
        crop_size=None,
        seed=0,
        clear_norm_data=False,
        **kwargs,
    ):

        self.sel_sims = sel_sims
        self.sel_const = sel_const
        self.disable_progress = disable_progress

        config.update(kwargs)
        _load_index()

        if dset_name in local_index.keys():
        
            dset_file = os.path.join(
                config["local_datasets_dir"], dset_name + config["dataset_ext"]
            )
            self._load_dataset(dset_name, dset_file)
        elif dset_name in global_index.keys():
            # self.__download_dataset__(dset_name, sel_sims)
            if global_index[dset_name]["isSingleFile"]:
                warn(
                    f"`{dset_name}` is stored in single-file format. The download might take some time."
                )
                fetcher.dl_single_file(
                    dset_name, config, disable_progress=self.disable_progress
                )  # ignore sel_sims
            else:
                fetcher.dl_parts(
                    dset_name,
                    config,
                    sims=sel_sims,
                    disable_progress=self.disable_progress,
                )

            dset_file = os.path.join(
                config["global_dataset_dir"], dset_name + config["dataset_ext"]
            )
            self._load_dataset(dset_name, dset_file)
        else:
            print(global_index.keys())
            suggestions = ", ".join(datasets())
            fail(
                f"Dataset '{dset_name}' not found, datasets available are: {suggestions}."
            )
            sys.exit(0)

        self.crop_size = crop_size

        self.set_seed(seed)

        self._validate_dataset()

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
            if isinstance(group, Group):
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

        if clear_norm_data:
            self._change_file_mode("r+")
            norm.clear_cache(self.dset)
            self._change_file_mode("r")

        # check if norm statistics are attached to dataset
        if (
            normalize_data or normalize_const
        ) and not norm.NormStrategy.check_norm_data(self.dset):
            info(
                "No precomputed normalization data found (or not complete). Calculating data..."
            )

            self._change_file_mode("r+")
            norm.NormStrategy.calculate_norm_data(self.dset)
            self._change_file_mode("r")

        self.norm_strat_data = (
            norm.get_norm_strat_from_str(normalize_data, self.dset, self.sel_const, const=False)
            if normalize_data
            else None
        )
        self.norm_strat_const = (
            norm.get_norm_strat_from_str(normalize_const, self.dset, self.sel_const, const=True)
            if normalize_const
            else None
        )
        
        self.sel_channels = sel_channels

    def set_seed(self, seed):

        self.rng = np.random.default_rng(seed)
    
    def _load_dataset(self, dset_name, dset_file):
        """Load hdf5 dataset, setting attributes of the dataset instance, doing basic validation checks."""

        # load dataset
        self.dset_name = dset_name
        self.dset_file = dset_file
        self.dset = h5py.File(dset_file, "r")

        # load metadata and setting attributes
        meta = get_meta_data(self.dset)
        for key, value in meta.items():
            setattr(self, key, value)

    def _validate_dataset(self):
        # basic validation checks on shape
        if len(self.sim_shape) < 3:
            corrupt(
                "Simulations data must have shape (frames, fields, spatial dim [...])."
            )
            sys.exit(0)

        if len(self.fields_scheme) != self.sim_shape[1]:
            raise ValueError(
                f"Inconsistent number of fields between metadata ({len(self.fields_scheme) }) and simulations ({ self.sim_shape[1]})."
            )

        for sim in self.dset["sims/"]:
            
            # shape must be consistent through all sims
            if (self.dset["sims/" + sim].shape) != self.sim_shape:
                corrupt(
                    f"The shape of all simulations must be consistent: Shape of first sim and sim {sim} do not match)."
                )
                sys.exit(0)

            # all sims must define the declared constants
            missing = set(self.const) - set(self.dset["sims/" + sim].attrs.keys())
            if missing:
                corrupt(
                    f"Simulation {sim} does not define all declared constants: {missing}."
                )
                sys.exit(0)

    def __len__(self):
        if self.sel_sims:
            return len(self.sel_sims) * self.samples_per_sim
        else:
            return self.num_sims * self.samples_per_sim

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

        dim_list = self.sim_shape[2:]

        if self.crop_size is None:

            input = sim[input_frame_idx]

        else:

            crop_dim_list = [self.rng.integers(low=0, high=dim-self.crop_size, size=1)[0] for dim in dim_list]
        
            # 2D
            if len(dim_list) == 2:
                input = sim[input_frame_idx, :, crop_dim_list[0]:crop_dim_list[0]+self.crop_size,
                                                crop_dim_list[1]:crop_dim_list[1]+self.crop_size]
            # 3D
            elif len(dim_list) == 3:
                input = sim[input_frame_idx, :, crop_dim_list[0]:crop_dim_list[0]+self.crop_size,
                                                crop_dim_list[1]:crop_dim_list[1]+self.crop_size,
                                                crop_dim_list[2]:crop_dim_list[2]+self.crop_size]
            else:
                raise ValueError(f'Dimension {self.sim_shape} not supported')

        if self.intermediate_time_steps:

            if self.crop_size is None:

                target = sim[input_frame_idx + 1: target_frame_idx + 1]

            else:

                if len(dim_list) == 2:
                    target = sim[input_frame_idx + 1: target_frame_idx + 1, :, crop_dim_list[0]:crop_dim_list[0]+self.crop_size,
                                                                               crop_dim_list[1]:crop_dim_list[1]+self.crop_size]

                elif len(dim_list) == 3:
                    target = sim[input_frame_idx + 1: target_frame_idx + 1, :, crop_dim_list[0]:crop_dim_list[0]+self.crop_size,
                                                                               crop_dim_list[1]:crop_dim_list[1]+self.crop_size,
                                                                               crop_dim_list[2]:crop_dim_list[2]+self.crop_size]
        
        else:

            if self.crop_size is None:

                target = sim[input_frame_idx]

            else:

                if len(dim_list) == 2:
                    target = sim[input_frame_idx, :, crop_dim_list[0]:crop_dim_list[0]+self.crop_size,
                                                                               crop_dim_list[1]:crop_dim_list[1]+self.crop_size]

                elif len(dim_list) == 3:
                    target = sim[input_frame_idx, :, crop_dim_list[0]:crop_dim_list[0]+self.crop_size,
                                                                               crop_dim_list[1]:crop_dim_list[1]+self.crop_size,
                                                                               crop_dim_list[2]:crop_dim_list[2]+self.crop_size]

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
                target = target[:, self.sel_channels]
            else:
                target = target[self.sel_channels]

        return (
            input,
            target,
            tuple(const),  # required by loader
            tuple(const_nnorm),  # needed by pbdl.torch.phi.loader
        )



    def _change_file_mode(self, mode):
        if self.dset:
            self.dset.close()

        self.dset = h5py.File(self.dset_file, mode)

    def get_sim_raw(self, sim):
        return self.dset["sims/sim" + str(sim)]

    def get_h5_raw(self):
        return self.dset

    def iterate_sims(self):
        num_sel_sims = len(self.sel_sims) if self.sel_sims else self.num_sims
        for s in range(num_sel_sims):
            yield range(s * self.samples_per_sim, (s + 1) * self.samples_per_sim)

    def info(self):

        info_str = f"{logging.BOLD}PDE:{logging.R_BOLD} {self.pde}\n"
        info_str += (
            f"{logging.BOLD}Fields Scheme:{logging.R_BOLD} {self.fields_scheme}\n"
        )
        info_str += f"{logging.BOLD}Dt:{logging.R_BOLD} {self.dt}\n"
        info_str += f"\n{logging.BOLD}Fields:{logging.R_BOLD}\n"

        for i, field in enumerate(self.fields):
            if hasattr(self, "field_desc"):
                info_str += f"   {field}:\t{self.field_desc[i]}\n"
            else:
                info_str += f"   {field}:\n"

        info_str += f"\n{logging.BOLD}Constants:{logging.R_BOLD}\n"
        for i, field in enumerate(self.fields):
            if hasattr(self, "const_desc"):
                info_str += f"   {field}:\t{self.field_desc[i]}\n"
            else:
                info_str += f"   {field}\n"
        return info_str

class Dataset3D(Dataset):

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
            sel_channels=None,  # if None, all channels are returned
            trim_start=None,  # by default 0
            trim_end=None,  # by default 0
            step_size=None,  # by default 1
            disable_progress=False,
            crop_size=None,
            seed=0,
            clear_norm_data=False,
            **kwargs,
    ):
        super().__init__(
            dset_name,
            time_steps,
            all_time_steps,
            intermediate_time_steps,
            normalize_data,
            normalize_const,
            sel_sims,
            sel_const,
            sel_channels,
            trim_start,
            trim_end,
            step_size,
            disable_progress,
            crop_size,
            seed,
            clear_norm_data,
            **kwargs
        )


        group = self.dset["sims"][f'{next(iter(self.dset["sims"]))}']
        self.num_frames = len(group)

        self.samples_per_sim = (
                    self.num_frames - self.time_steps - self.trim_start - self.trim_end
                    ) // self.step_size


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

        if self.sel_sims:
            sim_idx = self.sel_sims[idx // self.samples_per_sim]
        else:
            sim_idx = idx // self.samples_per_sim

        const = get_sel_const_sim_v2(self.dset, sim_idx, self.sel_const)

        input_frame_idx = (
                self.trim_start + (idx % self.samples_per_sim) * self.step_size
        )

        dim_list = self.sim_shape[:3]

        crop_dim_list = [self.rng.integers(low=0, high=dim-self.crop_size, size=1)[0] if self.crop_size < dim else 0 for dim in dim_list]

        input = self.dset[f"sims/sim{sim_idx}/{input_frame_idx}"][
                                        crop_dim_list[0]:crop_dim_list[0]+self.crop_size,
                                        crop_dim_list[1]:crop_dim_list[1]+self.crop_size,
                                        crop_dim_list[2]:crop_dim_list[2]+self.crop_size, :]

        target_frame_idx = input_frame_idx + self.time_steps

        if self.intermediate_time_steps:
            target_list = []
            for i in range(self.time_steps):
                data = self.dset[f"sims/sim{sim_idx}/{input_frame_idx + i + 1}"][
                                            crop_dim_list[0]:crop_dim_list[0]+self.crop_size,
                                            crop_dim_list[1]:crop_dim_list[1]+self.crop_size,
                                            crop_dim_list[2]:crop_dim_list[2]+self.crop_size, :]
                target_list.append(data)
            target = np.stack(target_list)
        else:
            target = self.dset[f"sims/sim{sim_idx}/{target_frame_idx}"][
                                            crop_dim_list[0]:crop_dim_list[0]+self.crop_size,
                                            crop_dim_list[1]:crop_dim_list[1]+self.crop_size,
                                            crop_dim_list[2]:crop_dim_list[2]+self.crop_size, :]

        const_nnorm = const

        if self.sel_channels is not None:
            input = input[self.sel_channels]
            if self.intermediate_time_steps:
                target = target[:, ..., self.sel_channels]
            else:
                target = target[..., self.sel_channels]

        input = input.transpose(3, 0, 1, 2)[None]
        target = target.transpose(0, 4, 1, 2, 3)

        return (
            input,
            target,
            tuple(const),  # required by loader
            tuple(const_nnorm),  # needed by pbdl.torch.phi.loader
        )

    def _load_dataset(self, dset_name, dset_file):
        """Load hdf5 dataset, setting attributes of the dataset instance, doing basic validation checks."""

        # load dataset
        self.dset_name = dset_name
        self.dset_file = dset_file
        self.dset = h5py.File(dset_file, "r")

        # load metadata and setting attributes
        meta = get_meta_data(self.dset) #, index_time=True)
        for key, value in meta.items():
            setattr(self, key, value)

    def _validate_dataset(self):
        # basic validation checks on shape
        if len(self.sim_shape) < 3:
            corrupt(
                "Simulations data must have shape (frames, fields, spatial dim [...])."
            )
            sys.exit(0)

        if len(self.fields_scheme) != self.sim_shape[-1]:
            raise ValueError(
                f"Inconsistent number of fields between metadata ({len(self.fields_scheme) }) and simulations ({ self.sim_shape[1]})."
            )

        for sim in self.dset["sims/"]:
            # shape must be consistent through all sims

            if (self.dset[f"sims/{sim}/0"].shape) != self.sim_shape:
                corrupt(
                    f"The shape of all simulations must be consistent: Shape of first sim and sim {sim} do not match)."
                )
                sys.exit(0)

            # all sims must define the declared constants
            missing = set(self.const) - set(self.dset[f"sims/{sim}/0"].attrs.keys())
            if missing:
                corrupt(
                    f"Simulation {sim} does not define all declared constants: {missing}."
                )
                sys.exit(0)



_load_index()

