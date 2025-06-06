import numpy as np
import os, json, time
import argparse
import h5py
import torch
from the_well.data import WellDataset

from render import render_trajectory
from download_setups_2d import get_setup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PDE simulation data")
    parser.add_argument("--dataset_name", type=str, default="shear_flow")
    parser.add_argument("--split_name", type=str, default="train")
    parser.add_argument("--out_name", type=str, default="shear_flow_train")
    parser.add_argument("--out_path", type=str, default="/mnt/ssdraid/pbdl-datasets-local/2d_well_new")
    parser.add_argument("--reduce_fraction", type=float, default=1.0) # randomly reduce number of simulations by this fraction, i.e., 0.1 means 10% of the simulations
    parser.add_argument("--reduce_seed", type=int, default=0)

    args = parser.parse_args()



def prepare_data_and_log(json_path_log, h5py_path, p_fixed):
    # create log file for this set of simulations
    with open(json_path_log, "w") as f:
        newDict = {"all": p_fixed}
        json.dump(newDict, f, indent=4)
        f.close()

    # create h5py file for dataset
    with h5py.File(h5py_path, "a") as h5py_file:
        if not "sims" in h5py_file:
            dataset = h5py_file.create_group("sims", track_order=True)
        else:
            dataset = h5py_file["sims"]

        for key in p_fixed:
            dataset.attrs[key] = p_fixed[key]
        h5py_file.close()


def update_param_log(json_path_log, p_varying, sim_id):
    # append parameter log file with parameters for each simulation
    with open(json_path_log, "r") as f:
        newDict = {"sim_%04d" % sim_id: p_varying}
        data = json.load(f)
        data.update(newDict)
        f.close()
        os.remove(json_path_log)
    with open(json_path_log, "w") as f:
        json.dump(data, f, indent=4)
        f.close()


def download_data(
        dataset_name: str,
        split_name: str,
        out_name: str,
        out_path: str,
        reduce_fraction: float = 1.0,
        reduce_seed: int = 0,
    ):

    # Create directories, paths, and logs
    out_dir = os.path.join(out_path, out_name)
    out_dir = out_dir[:-1] if out_dir[-1] == "/" else out_dir
    os.makedirs(out_dir, exist_ok=True)

    json_path_log = os.path.join(out_path, out_name + ".json")
    h5py_path = out_dir + ".hdf5"
    p_fixed = get_setup(dataset_name)
    prepare_data_and_log(json_path_log, h5py_path, p_fixed)

    dataset = WellDataset(
        well_base_path="hf://datasets/polymathic-ai/", # directly stream from huggingface
        well_dataset_name=dataset_name,
        well_split_name=split_name,
        n_steps_input=0,
        n_steps_output=p_fixed["Time Steps"],
        return_grid=False,
        boundary_return_type=None,
        full_trajectory_mode=True,
        max_rollout_steps=9999,
    )
    metadata = dataset.metadata
    if not all(p_fixed["Time Steps"] == steps for steps in metadata.n_steps_per_trajectory):
        print("Warning: Number of timesteps in setup and dataset do not match. This is only explicitly handled in a correct manner for the viscoelastic_instability setup.")

    dataset_size = len(dataset) if reduce_fraction >= 1.0 else int(len(dataset) * reduce_fraction)

    print("Number of simulations: %d%s" % (len(dataset), " (reduced by a factor of %1.1f to %d)" % (reduce_fraction, dataset_size) if reduce_fraction < 1.0 else ""))
    print(metadata)
    torch.manual_seed(reduce_seed)
    loader = torch.utils.data.DataLoader(dataset, shuffle=dataset_size < len(dataset), batch_size=1, num_workers=0)

    for (sim, id) in zip(loader, range(dataset_size)):

        data = sim["output_fields"][0].numpy().astype(np.float32)
        if "constant_fields" in sim:
            const_fields = sim["constant_fields"][0].numpy().astype(np.float32)

            const_fields_time = np.expand_dims(const_fields, axis=0) # add time dim
            const_fields_time = np.repeat(const_fields_time, data.shape[0], axis=0) # expand to match temporal dimension
            data = np.concatenate([data, const_fields_time], axis=-1) # stack to data

        data = data.transpose(0, 3, 1, 2) # change order to channels first
        constants = sim["constant_scalars"][0].numpy().astype(np.float32)

        print("DOWNLOADED SIM %d from %s with shape %s" % (id, dataset_name + " " + split_name, str(data.shape)))
        print("Constants: ", constants)

        assert constants.shape[0] == len(p_fixed["Constants"]), "Warning: Number of constants in setup and dataset do not match"

        p_varying = {}
        for idx, key in enumerate(p_fixed["Constants"]):
            p_varying[key] = constants[idx].item()

        update_param_log(json_path_log, p_varying, id)

        # Save to h5py
        with h5py.File(h5py_path, "a") as h5py_file:
            if "sims/sim%d" % (id) in h5py_file:
                del h5py_file["sims/sim%d" % (id)]
            dataset = h5py_file.create_dataset("sims/sim%d" % (id), data=data)

            for idx, key in enumerate(p_fixed["Constants"]):
                dataset.attrs[key] = constants[idx].item()
            h5py_file.close()

        # Render
        vmin = None
        vmax = None
        render_trajectory(
            data=data,
            dimension=2,
            output_path=out_dir,
            sim_id=id,
            time_steps=p_fixed["Time Steps"],
            steps_plot=10,
            vmin=vmin,
            vmax=vmax,
        )

        print("\n")



if __name__ == '__main__':
    download_data(args.dataset_name, args.split_name, args.out_name, args.out_path, args.reduce_fraction, args.reduce_seed)