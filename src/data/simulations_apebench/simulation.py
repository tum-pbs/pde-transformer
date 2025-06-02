
import os
import time
import json
import gc
import argparse
import h5py

# parse arguments before importing jax
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PDE simulation data")
    parser.add_argument("--pde", type=str, default="kolm_flow")
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--test_set", default=False, action='store_true') # NOTE: omit argument to set to false
    parser.add_argument("--out_name", type=str, default="kolm_flow")
    parser.add_argument("--out_path", type=str, default="/mnt/data2/pbdl-datasets-local/2d_apebench_256_large")
    parser.add_argument("--num_sims", type=int, default=60)
    parser.add_argument("--render_only", default=False, action='store_true') # NOTE: omit argument to set to false
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--low-res", default=False, action='store_true') # NOTE: omit argument to set to false

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
    #os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import numpy as np
import exponax as ex
import jax
import jax.numpy as jnp
#jax.config.update("jax_platform_name", "cpu")

from simulation_setups_2d import get_setup_2d
from simulation_setups_3d import get_setup_3d
from render import render_trajectory


def prepare_data_and_log(json_path_log, h5py_path, p_fixed):
    # create log file for this set of simulations
    with open(json_path_log, "w") as f:
        newDict = {"all": p_fixed}
        json.dump(newDict, f, indent=4)
        f.close()

    # create h5py file for dataset
    with h5py.File(h5py_path, "w") as h5py_file:
        dataset = h5py_file.create_group("sims", track_order=True)
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



def generate_data(
        sim_type: str,
        dimension: int,
        is_test_set: bool,
        out_name: str,
        out_path: str,
        num_sims: int,
        render_only: bool,
        low_res: bool = False
    ):

    from simulation_setups_2d import get_setup_2d

    # if not low_res:
    #     from simulation_setups_2d import get_setup_2d
    # else:
    #     from simulation_setups_2d_low_res import get_setup_2d


    # Create directories, paths, and logs
    out_dir = os.path.join(out_path, out_name)
    out_dir = out_dir[:-1] if out_dir[-1] == "/" else out_dir
    os.makedirs(out_dir, exist_ok=True)

    json_path_log = os.path.join(out_path, out_name + ".json")
    h5py_path = out_dir + ".hdf5"
    if dimension == 2:
        p_fixed, _, _, _ = get_setup_2d(sim_type, is_test_set, 0)
    elif dimension == 3:
        p_fixed, _, _, _ = get_setup_3d(sim_type, is_test_set, 0)
    else:
        raise ValueError("Invalid dimension: %d" % dimension)
    prepare_data_and_log(json_path_log, h5py_path, p_fixed)

    data_id = 0
    sim_id = 0
    # Run a set of simulations
    while data_id < num_sims:

        print("SIMULATION COUNTER %d" % data_id)
        print("%s SIMULATION %d" % (sim_type.upper(), sim_id))

        # Initialize simulation setup and log parameters
        if dimension == 2:
            p_fixed, p_varying, stepper, u_next = get_setup_2d(sim_type, is_test_set, sim_id)
        else:
            p_fixed, p_varying, stepper, u_next = get_setup_3d(sim_type, is_test_set, sim_id)
        update_param_log(json_path_log, p_varying, sim_id)

        # Simulation loop
        time_steps = p_fixed["Time Steps"] + p_fixed.get("Warmup Steps", 0)
        data = []
        timing = []
        repeated_stepper = ex.RepeatedStepper(stepper, p_fixed["Sub Steps"]) # omit writing substeps

        sim_nan = False
        for i in range(time_steps):
            start = time.perf_counter()

            u_next = repeated_stepper(u_next)

            if jnp.isnan(u_next).any():
                sim_nan = True
                print("WARNING: NaN at in simulation %d at time step %d" % (sim_id, i))
                break

            # Timing
            timing.append((time.perf_counter() - start) / 3600.0)
            rem = sum(timing) / len(timing)
            rem = rem * (time_steps - i)
            remStr = "%1.2f h" % rem if rem > 1.0 else "%1.1f min" % (rem * 60.0)

            if i % 10 == 0 or i == time_steps - 1:
                print("\tSim %d/%d, Time step %d, mean: %s (%s remaining)" % (data_id+1, num_sims, i, jnp.mean(u_next), remStr))

            if "Warmup Steps" in p_fixed and i < p_fixed["Warmup Steps"]:
                continue

            data.append(np.asarray(u_next, dtype=np.float32))

        if sim_nan:
            print("Simulation %d had NaN values. Skipping..." % sim_id)
            sim_id += 1
            continue
        else:
            sim_id += 1

        # Save to h5py
        if not render_only:
            with h5py.File(out_dir + ".hdf5", "a") as h5py_file:
                data = np.stack(data, axis=0)

                if low_res:
                    # downsample (average pooling)
                    if len(data.shape) == 4:
                        data = data.reshape(data.shape[0], data.shape[1], 256, 8, 256, 8).mean(axis=(3, 5))
                    elif len(data.shape) == 3:
                        data = data.reshape(data.shape[0], 256, 8, 256, 8).mean(axis=(2, 4))
                    else:
                        raise ValueError('Data shape not supported')

                dataset = h5py_file.create_dataset("sims/sim%d" % (data_id), data=data)

                for key in p_fixed["Constants"]:
                    dataset.attrs[key] = p_varying[key]
                h5py_file.close()

        data_id += 1

        if sim_type in ["adv", "diff", "adv_diff", "disp", "hyp", "burgers", "kdv"]:
            vmin = -0.5
            vmax = 0.5
        elif sim_type in ["fisher", "gs_alpha", "gs_beta", "gs_gamma", "gs_delta", "gs_epsilon", "gs_theta", "gs_iota", "gs_kappa"]:
            vmin = 0.0
            vmax = 1.0
        else:
            # automatically determined via min and max of data
            vmin = None
            vmax = None

        try:
            # Render
            render_trajectory(
                data=data,
                dimension=dimension,
                output_path=out_dir,
                sim_id=sim_id,
                time_steps=p_fixed["Time Steps"],
                steps_plot=10,
                vmin=vmin,
                vmax=vmax,
            )
        except Exception as e:
            print("Error in rendering: %s" % str(e))

        del stepper, u_next, data, repeated_stepper
        gc.collect()

        print("\n\n")



if __name__ == "__main__":
    generate_data(args.pde, args.dimension, args.test_set, args.out_name, args.out_path,
                  args.num_sims, args.render_only, args.low_res)
