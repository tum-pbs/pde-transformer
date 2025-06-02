# convert data from apebench (scraped or generated) to pbdl dataloader format
import numpy as np
import os
import h5py
from .utils import print_h5py
import argparse
from typing import Optional
import yaml
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_folder",
        type=str,
        required=True,
        help="path to the folder containing the data",
    )  
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default=None,
        help="path to the output directory",
    )
    args = parser.parse_args()
    return args

def convert_npy(
    data_path:str,
    out_dir:str,
    out_name:str,
    metadata:dict,
    ):
    data = np.load(data_path)
    print(f"Shape of the original npy data: {data.shape}")

    const = np.arange(data.shape[0])  # simulation seed is only constant
    const = const[:, np.newaxis]

    with h5py.File(os.path.join(out_dir, out_name + ".hdf5"), "w") as h5py_file:
        group = h5py_file.create_group("sims", track_order=True)
        for key in metadata:
            if key != "Constants Sim":
                group.attrs[key] = metadata[key]
        for s in range(data.shape[0]):
            dataset = h5py_file.create_dataset("sims/sim%d" % (s), data=data[s])
            constant=metadata["Constants Sim"][s]
            for key,item in constant.items():
                dataset.attrs[key]=item
        h5py_file.close()

    print_h5py(os.path.join(out_dir, out_name + ".hdf5"))
    print("\n\n")
    
def convert_torchfsm(
    data_folder:str,
    out_dir:Optional[str]=None,
):
    files=tqdm([file_i.split(".")[0] for file_i in os.listdir(data_folder) if file_i.endswith(".npy")])
    for file_i in files:
        files.set_description(f"Processing {file_i}")
        with open(os.path.join(data_folder,f"{file_i}.yaml")) as f:
            config=yaml.safe_load(f)
        out_name=config["name"]
        out_dir=data_folder if out_dir is None else out_dir
        metadata=config["metadata"]
        convert_npy(
            data_path=os.path.join(data_folder,f"{file_i}.npy"),
            out_dir=out_dir,
            out_name=out_name,
            metadata=metadata,
        )
        
if __name__ == "__main__":
    args=parse_args()   
    convert_torchfsm(
        data_folder=args.data_folder,
        out_dir=args.out_dir,
    )
    