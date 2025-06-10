import os
import h5py
import numpy as np
from itertools import groupby

from h5py import Group

from .logging import info, success, warn, fail, corrupt

META_ATTRS_REQUIRED = {
    "PDE": str,
    "Fields Scheme": str,
    "Fields": list,
    "Constants": list,
    "Dt": float,
}


def get_sel_const_sim(dset, sim, sel_const):
    attrs = dset["sims/sim" + str(sim)].attrs
    const = sel_const if sel_const else dset["sims/"].attrs["Constants"]
    return [attrs[key] for key in const]

def get_sel_const_sim_v2(dset, sim, sel_const):
    attrs = dset[f"sims/sim{sim}/0"].attrs
    const = sel_const if sel_const else dset["sims/"].attrs["Constants"]
    return [attrs[key] for key in const]

def get_const_sim(dset, sim):
    attrs = dset["sims/sim" + str(sim)].attrs
    return [attrs[key] for key in dset["sims/"].attrs["Constants"]]

def get_const_sim_v2(dset, sim):
    attrs = dset[f"sims/sim{sim}/0"].attrs
    return [attrs[key] for key in dset["sims/"].attrs["Constants"]]

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
        first_sim = dset["sims"][f'{next(iter(dset["sims"]))}/0']
        sim_shape = first_sim.shape
        num_spatial_dim = len(sim_shape) - 1
    else:
        first_sim = group
        sim_shape = first_sim.shape
        num_spatial_dim = len(sim_shape) - 2 # subtract 2 for frame and field dimension

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


def scan_local_dset_dir(config):
    local_dset_dir = config["local_datasets_dir"]
    if os.path.isdir(local_dset_dir):
        return {
            os.path.splitext(file)[0]: _load_metadata_of_local_dset(
                os.path.join(local_dset_dir, file), os.path.splitext(file)[0]
            )
            for file in os.listdir(local_dset_dir)
            if file.endswith(config["dataset_ext"])
        }
    else:
        warn(
            f"{local_dset_dir} is not a valid directory. No local datasets will be available."
        )
        return {}


def _load_metadata_of_local_dset(file: str, dset_name):
    with h5py.File(file, "r") as f:
        metadata = {
            # h5py converts list automatically to arrays, undo this conversion
            key: val.tolist() if isinstance(val, np.ndarray) else val
            for key, val in f["sims/"].attrs.items()
        }

        # check metadata
        missing_keys = META_ATTRS_REQUIRED.keys() - metadata.keys()
        incorrect_types = {
            key: type(metadata[key])
            for key in META_ATTRS_REQUIRED.keys() & metadata.keys()
            if not isinstance(metadata[key], META_ATTRS_REQUIRED[key])
        }

        if missing_keys:
            warn(
                f"'{dset_name}' is missing metadata entries: {', '.join(missing_keys)}"
            )
        if incorrect_types:
            warn(
                f"'{dset_name}' has metadata entries with wrong type: {', '.join(f'{key} (expected {META_ATTRS_REQUIRED[key].__name__}, got {incorrect_types[key].__name__})' for key in incorrect_types)}"
            )

        return metadata

