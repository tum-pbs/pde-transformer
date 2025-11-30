import numpy as np
import os
import sys
import h5py

import pdetransformer.data.pbdlu_dataloader.normalization as norm

from h5py import Group
from typing import Type
from .logging import info, success, warn, fail, corrupt
from pdetransformer.data.pbdlu_dataloader import logging


class Dataset:
    REQUIRED_DSET_ATTRS = {
        "pde": str,
        "in_fields_scheme": str,
        "in_fields": list,
        "out_fields_scheme": str,
        "out_fields": list,
        "const": list,
        "dt": float,
    }
    
    RESERVED_META_ATTRS = {
        "num_sims": int,
        "num_const": int,
        "num_frames": int,
        "num_in_fields": int,
        "num_out_fields": int,
        "num_spatial_dim": int,
    }
    
    REQUIRED_DSET_ATTRS_MAPPING = {
        "PDE": "pde",
        "Input Fields Scheme": "in_fields_scheme",
        "Input Fields": "in_fields",
        "Output Fields Scheme": "out_fields_scheme",
        "Output Fields": "out_fields",
        "Constants": "const",
        "Dt": "dt",
    }
    
    RESERVED_NORM_ATTRS = {
        "norm_pos_mean",
        "norm_pos_std",
        "norm_pos_min",
        "norm_pos_max",
        "norm_in_fields_mean",
        "norm_in_fields_std",
        "norm_in_fields_min",
        "norm_in_fields_max",
        "norm_out_fields_mean",
        "norm_out_fields_std",
        "norm_out_fields_min",
        "norm_out_fields_max",
        "norm_const_mean",
        "norm_const_std",
        "norm_const_min",
        "norm_const_max",
    }
    
    
    def __init__(
        self,
        dset_path: str,
        sel_sims: list[int] | None = None,  # if None, all simulations are loaded
        sel_const: list[str] | None = None,  # if None, all constants are returned
        sel_in_channels: list[int] | None = None, # if None, all input channels are returned
        sel_out_channels: list[int] | None = None, # if None, all target channels are returned
        time_steps: int | None = None,  # by default num_frames
        step_size: int | None = None,  # by default 1
        trim_start: int | None = None,  # by default 0
        trim_end: int | None = None,  # by default 0
        intermediate_time_steps: bool = False,  # by default False
        all_time_steps: bool = False,  # overrides time_steps, step_size, intermediate_time_steps, trim_start, trim_end
        normalize_pos: Type[norm.NormStrategy] | None = None,  # by default no normalization
        normalize_in: Type[norm.NormStrategy] | None = None,  # by default no normalization
        normalize_out: Type[norm.NormStrategy] | None = None,  # by default no normalization
        normalize_const: Type[norm.NormStrategy] | None = None,  # by default no normalization
        seed: int = 0,
        **kwargs,
    ):
        self.sel_sims = sel_sims
        self.sel_const = sel_const
        self.sel_in_channels = sel_in_channels
        self.sel_out_channels = sel_out_channels
        self.rng = np.random.default_rng(seed)
        
        if os.path.exists(dset_path):
            self._load_dataset(dset_path)
            # Set metadata as attibutes
            for key, value in self.get_meta_data().items():
                setattr(self, key, value)
        else:
            fail(f"Dataset '{dset_path}' not found.")
            sys.exit(0)
        
        # time step handling
        if all_time_steps:
            self.time_steps = self.num_frames # Set by _load_dataset
            self.intermediate_time_steps = True
            self.trim_start = 0
            self.trim_end = 0
            self.step_size = 1
            self.samples_per_sim = 1

            # Warn if other time step related args are set
            for attr, val in [
                ("time_steps", time_steps),
                ("intermediate_time_steps", intermediate_time_steps),
                ("trim_start", trim_start),
                ("trim_end", trim_end),
                ("step_size", step_size),
            ]:
                if val is not None:
                    warn(f"`{attr}` is managed by `all_time_steps` and can therefore not be set manually.")
        else:
            self.time_steps = time_steps or self.num_frames # Set by _load_dataset
            self.intermediate_time_steps = intermediate_time_steps or False
            self.trim_start = trim_start or 0
            self.trim_end = trim_end or 0
            self.step_size = step_size or 1 
        
        # Calculate number of samples per simulation        
        self.samples_per_sim = (
            self.num_frames - self.time_steps - self.trim_start - self.trim_end + 1
        )
        if self.step_size > 1:
            self.samples_per_sim += 1
            self.samples_per_sim //= self.step_size
            
        success(
            f"Loaded { self.dset_name } with { self.num_sims } simulations "
            + (f"({len(self.sel_sims)} selected) " if self.sel_sims else "")
            + f"and {self.samples_per_sim} samples each."
        )
        
        # TODO: Implement normalization for unstructured data       
        if normalize_pos or normalize_in or normalize_out or normalize_const:
            if not self.check_norm_data():
                info("No precomputed normalization data found (or not complete). Calculating data...")
                self.calculate_norm_data()
                
            self.load_norm_data()
            
        self.norm_strat_pos = normalize_pos({
                "mean": self.norm_attrs["norm_pos_mean"],
                "std": self.norm_attrs["norm_pos_std"],
                "min": self.norm_attrs["norm_pos_min"],
                "max": self.norm_attrs["norm_pos_max"],
            }) if normalize_pos else None
        
        self.norm_strat_in = normalize_in({
                "mean": self.norm_attrs["norm_in_fields_mean"],
                "std": self.norm_attrs["norm_in_fields_std"],
                "min": self.norm_attrs["norm_in_fields_min"],
                "max": self.norm_attrs["norm_in_fields_max"],
            }) if normalize_in else None
        
        self.norm_strat_out = normalize_out({
                "mean": self.norm_attrs["norm_out_fields_mean"],
                "std": self.norm_attrs["norm_out_fields_std"],
                "min": self.norm_attrs["norm_out_fields_min"],
                "max": self.norm_attrs["norm_out_fields_max"],
            }) if normalize_out else None
        
        self.norm_strat_const = normalize_const({
                "mean": self.norm_attrs["norm_const_mean"],
                "std": self.norm_attrs["norm_const_std"],
                "min": self.norm_attrs["norm_const_min"],
                "max": self.norm_attrs["norm_const_max"],
            }) if normalize_const else None
    
    def _load_dataset(self, dset_path):
        """Loads the dataset and sets dataset specific attributes on `self`

        Args:
            dset_path (str): The path to the dataset file
        """
        self.dset_name = dset_path.split(".")[-2].split("/")[-1]
        self.dset_ext = dset_path.split(".")[-1]
        if self.dset_ext != "hdf5" and self.dset_ext != "h5":
            fail(f"Dataset file format '{self.dset_ext}' not supported. Only HDF5 files are supported.")
            sys.exit(0)
        self.dset_path = dset_path
        
        # HDF5 file format
        self.dset = h5py.File(self.dset_path, "r")
    
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
            tuple: Constants
            
        """
        if idx >= len(self):
            raise IndexError
        
        # create input-target pairs with interval time_steps from simulation steps
        if self.sel_sims:
            sim_idx = self.sel_sims[idx // self.samples_per_sim]
        else:
            sim_idx = idx // self.samples_per_sim

        sim = self.dset["sims/sim" + str(int(sim_idx))]
        sim_x : np.ndarray = sim["x"]
        sim_y : np.ndarray = sim["y"]
        sim_fx : np.ndarray = sim["fx"]

        input_frame_idx = (
                self.trim_start + (idx % self.samples_per_sim) * self.step_size
        )
        target_frame_idx = input_frame_idx + self.time_steps - 1

        if self.intermediate_time_steps:
            positions = sim_x[input_frame_idx:target_frame_idx+1]
            targets = sim_y[input_frame_idx:target_frame_idx+1]
            features = sim_fx[input_frame_idx:target_frame_idx+1]
        else:
            positions = sim_x[input_frame_idx:input_frame_idx + 1]
            targets = sim_y[target_frame_idx:target_frame_idx + 1]
            features = sim_fx[input_frame_idx:input_frame_idx + 1]
            
        # Get simulation constants
        const = self.get_const_sim(sim_idx, selected=True)
        
        # Apply normalization
        if self.norm_strat_pos:
            positions = self.norm_strat_pos.normalize(positions)
        if self.norm_strat_in:
            features = self.norm_strat_in.normalize(features)
        if self.norm_strat_out:
            targets = self.norm_strat_out.normalize(targets)
        if self.norm_strat_const:
            const = self.norm_strat_const.normalize(const)

        # Filter selected channels
        if self.sel_in_channels is not None:
            features = features[:,self.sel_in_channels]
        if self.sel_out_channels is not None:
            targets = targets[:,self.sel_out_channels]
        
        return (
            positions,
            targets,
            features,
            const
        )
        
    def info(self):
        info_str = f"{logging.BOLD}PDE:{logging.R_BOLD} {self.pde}\n"
        info_str += (f"{logging.BOLD}Input Fields Scheme:{logging.R_BOLD} {self.in_fields_scheme}\n")
        info_str += (f"{logging.BOLD}Output Fields Scheme:{logging.R_BOLD} {self.out_fields_scheme}\n")
        info_str += f"{logging.BOLD}Dt:{logging.R_BOLD} {self.dt}\n"
        info_str += f"\n{logging.BOLD}Input Fields:{logging.R_BOLD}\n"
        for i, field in enumerate(self.in_fields):
            if hasattr(self, "in_field_desc"):
                info_str += f"   {field}:\t{self.in_fields_desc[i]}\n"
            else:
                info_str += f"   {field}:\n"
        info_str += f"\n{logging.BOLD}Output Fields:{logging.R_BOLD}\n"
        for i, field in enumerate(self.out_fields):
            if hasattr(self, "out_field_desc"):
                info_str += f"   {field}:\t{self.out_fields_desc[i]}\n"
            else:
                info_str += f"   {field}:\n"
        # TODO: Ask if this is correct
        info_str += f"\n{logging.BOLD}Constants:{logging.R_BOLD}\n"
        for i, field in enumerate(self.const):
            if hasattr(self, "const_desc"):
                info_str += f"   {field}:\t{self.const_desc[i]}\n"
            else:
                info_str += f"   {field}\n"
        return info_str
    
    def get_meta_data(self):
        if not self.dset:
            raise ValueError("Dataset not loaded.")
        
        required_meta_attrs = {}
        reserved_meta_attrs = {}

        # TODO: Allow for different file formats
        # Fetch required meta attributes from dataset
        meta_attrs = self.dset["sims"].attrs
        required_meta_attrs.update({
            Dataset.REQUIRED_DSET_ATTRS_MAPPING[field]: meta_attrs[field] for field in Dataset.REQUIRED_DSET_ATTRS_MAPPING.keys() if field in meta_attrs
        })
        
        # Check for required attributes
        missing_attrs = [attr for attr in Dataset.REQUIRED_DSET_ATTRS.keys() if attr not in required_meta_attrs]
        if missing_attrs:
            raise ValueError(f"Dataset is missing required attributes: {', '.join(missing_attrs)}. Also look in `Dataset.REQUIRED_DSET_ATTRS_MAPPING` for correct naming in dataset file.")
        
        # TODO: Allow for different file formats
        # Calculate reserved meta attributes 
        group = self.dset["sims"][f'{next(iter(self.dset["sims"]))}']
        if isinstance(group, Group):
            first_sim_x = self.dset["sims"][f'{next(iter(self.dset["sims"]))}/x']
            first_sim_y = self.dset["sims"][f'{next(iter(self.dset["sims"]))}/y']
            first_sim_fx = self.dset["sims"][f'{next(iter(self.dset["sims"]))}/fx']
            num_spatial_dim = first_sim_x.shape[1]
            reserved_meta_attrs.update({
                "num_sims": len(self.dset["sims"]),
                "num_const": len(required_meta_attrs["const"]),
                "num_frames": first_sim_x.shape[0],
                "num_in_fields": first_sim_fx.shape[1],
                "num_out_fields": first_sim_y.shape[1],
                "num_spatial_dim": num_spatial_dim,
            })
        else:
            raise ValueError("Dataset for unstructured data must contain 'x', 'y', and 'fx' datasets per simulation.")

        # Check for reserved attributes
        missing_attrs = [attr for attr in Dataset.RESERVED_META_ATTRS.keys() if attr not in reserved_meta_attrs]
        if missing_attrs:
            raise ValueError(f"Some reserved metadata attributes were not determined correctly: {', '.join(missing_attrs)}")
        
        # Construct final meta dictionary
        meta = {}
        meta.update(required_meta_attrs)
        meta.update(reserved_meta_attrs)
        return meta
    
    def get_const_sim(self, sim_idx: int, selected: bool = False):
        attrs = self.dset["sims/sim" + str(int(sim_idx))].attrs
        if selected and self.sel_const:
            const = self.sel_const
        else:
            const = self.dset["sims/"].attrs["Constants"]
        return np.array([attrs[key] for key in const])

    def check_norm_data(self):
        return all(attr in self.dset for attr in Dataset.RESERVED_NORM_ATTRS)

    def calculate_norm_data(self):
        # Clear old norm data
        if self.dset:
            self.dset.close()
        self.dset = h5py.File(self.dset_path, "r+")
        
        for attr in self.RESERVED_NORM_ATTRS:
            self.dset.pop(attr, None)
        
        # Calculate new norm data
        pos_mean = np.full((1, self.num_spatial_dim, 1), 0)
        pos_std = np.full((1, self.num_spatial_dim, 1), 0)
        pos_max = np.full((1, self.num_spatial_dim, 1), -np.inf)
        pos_min = np.full((1, self.num_spatial_dim, 1), np.inf)
        in_fields_mean = np.full((1, self.num_in_fields, 1), 0)
        in_fields_std = np.full((1, self.num_in_fields, 1), 0)
        in_fields_max = np.full((1, self.num_in_fields, 1), -np.inf)
        in_fields_min = np.full((1, self.num_in_fields, 1), np.inf)
        out_fields_mean = np.full((1, self.num_out_fields, 1), 0)
        out_fields_std = np.full((1, self.num_out_fields, 1), 0)
        out_fields_max = np.full((1, self.num_out_fields, 1), -np.inf)
        out_fields_min = np.full((1, self.num_out_fields, 1), np.inf)
        const_stacked = []
        
        for sim_name, sim in self.dset["sims"].items():
            sim_pos = sim["x"]
            sim_in_fields = sim["fx"]
            sim_out_fields = sim["y"]
            sim_const = self.get_const_sim(int(sim_name[3:]))
            
            pos_mean = np.add(pos_mean, np.mean(sim_pos, axis=(0, 2), keepdims=True))
            pos_std = np.add(pos_std, np.std(sim_pos, axis=(0, 2), keepdims=True))
            pos_max = np.maximum(pos_max, np.max(sim_pos, axis=(0, 2), keepdims=True))
            pos_min = np.minimum(pos_min, np.min(sim_pos, axis=(0, 2), keepdims=True))
            
            in_fields_mean = np.add(in_fields_mean, np.mean(sim_in_fields, axis=(0, 2), keepdims=True))
            in_fields_std = np.add(in_fields_std, np.std(sim_in_fields, axis=(0, 2), keepdims=True))
            in_fields_max = np.maximum(in_fields_max, np.max(sim_in_fields, axis=(0, 2), keepdims=True))
            in_fields_min = np.minimum(in_fields_min, np.min(sim_in_fields, axis=(0, 2), keepdims=True))
            
            out_fields_mean = np.add(out_fields_mean, np.mean(sim_out_fields, axis=(0, 2), keepdims=True))
            out_fields_std = np.add(out_fields_std, np.std(sim_out_fields, axis=(0, 2), keepdims=True))
            out_fields_max = np.maximum(out_fields_max, np.max(sim_out_fields, axis=(0, 2), keepdims=True))
            out_fields_min = np.minimum(out_fields_min, np.min(sim_out_fields, axis=(0, 2), keepdims=True))
            
            const_stacked.append(sim_const)
            
        pos_mean /= self.num_sims
        pos_std /= self.num_sims
        in_fields_mean /= self.num_sims
        in_fields_std /= self.num_sims
        out_fields_mean /= self.num_sims
        out_fields_std /= self.num_sims
        
        const_stacked = np.stack(const_stacked)
        const_mean = np.mean(const_stacked, axis=0)
        const_std = np.std(const_stacked, axis=0)
        const_max = np.max(const_stacked, axis=0)
        const_min = np.min(const_stacked, axis=0)
        
        # Save norm data
        self.dset["norm_pos_mean"] = pos_mean
        self.dset["norm_pos_std"] = pos_std
        self.dset["norm_pos_max"] = pos_max
        self.dset["norm_pos_min"] = pos_min
        self.dset["norm_in_fields_mean"] = in_fields_mean
        self.dset["norm_in_fields_std"] = in_fields_std
        self.dset["norm_in_fields_max"] = in_fields_max
        self.dset["norm_in_fields_min"] = in_fields_min
        self.dset["norm_out_fields_mean"] = out_fields_mean
        self.dset["norm_out_fields_std"] = out_fields_std
        self.dset["norm_out_fields_max"] = out_fields_max
        self.dset["norm_out_fields_min"] = out_fields_min
        self.dset["norm_const_mean"] = const_mean
        self.dset["norm_const_std"] = const_std
        self.dset["norm_const_max"] = const_max
        self.dset["norm_const_min"] = const_min
        
        if self.dset:
            self.dset.close()
        self.dset = h5py.File(self.dset_path, "r")
        
    def load_norm_data(self):
        norm_attrs = {}
        for attr in self.RESERVED_NORM_ATTRS:
            if attr in self.dset:
                norm_attrs[attr] = self.dset[attr][()]
            else:
                raise ValueError(f"Normalization data '{attr}' not found in dataset.")
        self.norm_attrs = norm_attrs