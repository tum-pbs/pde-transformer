import requests
import h5py
import os
import urllib
import io
import json
from . import normalization as norm
import pkg_resources
import sys

from .logging import info, success, warn, fail, DARKGREY, SUCCESS_CYAN, ENDC


def dl_parts(dset: str, config, sims: list[int] = None, disable_progress=False):
    os.makedirs(config["global_dataset_dir"], exist_ok=True)
    dest = os.path.join(config["global_dataset_dir"], dset + config["dataset_ext"])

    # TODO dispatching
    prog_hook = None if disable_progress else print_download_progress
    modified = dl_parts_from_huggingface(dset, dest, config, sims, prog_hook=prog_hook)

    # normalization data will not incorporate all sims after download
    if modified:
        with h5py.File(dest, "r+") as dset:
            norm.clear_cache(dset)


def dl_single_file(dset: str, config, disable_progress=False):
    os.makedirs(config["global_dataset_dir"], exist_ok=True)
    dest = os.path.join(config["global_dataset_dir"], dset + config["dataset_ext"])

    if os.path.exists(dest):
        # dataset already downloaded
        print(f"Dataset {dset} already downloaded in {dest}")
        return

    prog_hook = None if disable_progress else print_download_progress
    dl_single_file_from_huggingface(dset, dest, config, prog_hook=prog_hook)


def fetch_index(config):
    # TODO dispatching
    return fetch_index_from_huggingface(config)


def dl_single_file_from_huggingface(dset: str, dest: str, config, prog_hook=None):
    repo_id = config["hf_repo_id"]
    url_ds = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{dset}/{dset}{config['dataset_ext']}"

    with urllib.request.urlopen(url_ds) as response:
        total_size = int(response.info().get("Content-Length").strip())
        block_size = 16384 # 16 KB
        refresh_rate = 10 # update progress every 10 blocks
        with open(dest, "wb") as out_file:
            for count, data in enumerate(iter(lambda: response.read(block_size), b"")):
                out_file.write(data)
                if prog_hook and count % refresh_rate == 0:
                    prog_hook(count, block_size, total_size)

    if prog_hook:
        prog_hook(1, 1, 1, message="download completed")
    else:
        success("Download completed.")


def dl_parts_from_huggingface(
    dataset: str, dest: str, config, sel_sims: list[int] = None, prog_hook=None
):
    """Adds partitions to hdf5 file. If parts is not specified, alls partitions are added."""

    repo_id = config["hf_repo_id"]
    url_repo = f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
    url_meta_all = url_repo + f"{dataset}/meta_all.json"

    # create sim-to-file mapping
    files = get_hf_repo_file_list(repo_id)
    sim_to_file = {}
    for file in files:
        if file.startswith(dataset + "/sim"):
            parts = file.split("/sim")[-1].split(".")[0].split("-")
            sim_from = int(parts[0])
            sim_to = int(
                parts[-1]
            )  # is the same as sim_from if files contains only one sim
            for sim in range(sim_from, sim_to + 1):
                sim_to_file[sim] = file
    
    if not sel_sims:
        # expect numbering to be consecutive
        sel_sims = range(len(sim_to_file))

    modified = False
    with h5py.File(dest, "a") as f:
        for i, s in enumerate(sel_sims):
            if prog_hook:
                prog_hook(
                    i,
                    1,
                    len(sel_sims),
                    message=f"downloading sim {s}",
                )

            if "sims/sim" + str(s) not in f:
                modified = True

                url_sim = url_repo + sim_to_file[s]

                with urllib.request.urlopen(url_sim) as response:
                    with h5py.File(io.BytesIO(response.read()), "r") as dset_sim:
                        for sk in dset_sim["sims"]:
                            sim = f.create_dataset(
                                f"sims/{sk}", data=dset_sim[f"sims/{sk}"]
                            )
                            for key, value in dset_sim[f"sims/{sk}"].attrs.items():
                                sim.attrs[key] = value

        # update meta all
        with urllib.request.urlopen(url_meta_all) as response:
            meta_all = json.loads(response.read().decode())
            for key, value in meta_all.items():
                f["sims/"].attrs[key] = value

    if prog_hook:
        prog_hook(1, 1, 1, message="download completed")
    else:
        success("Download completed.")

    return modified


def fetch_index_from_huggingface(config):
    repo_id = config["hf_repo_id"]
    url_repo = f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
    index_path = pkg_resources.resource_filename(__name__, "global_index.json")

    try:
        files = get_hf_repo_file_list(repo_id)

        first_level_dirs = {file.split("/")[0] for file in files if "/" in file}

        first_level_dirs_data_file = {
            d for d in first_level_dirs if f"{d}/{d}{config['dataset_ext']}" in files
        }

        datasets_part = first_level_dirs - first_level_dirs_data_file
        datasets_single = first_level_dirs_data_file

        # partitioned datasets
        meta_all_combined = {}
        for d in datasets_part:
            url_meta_all = url_repo + d + "/meta_all.json"
            meta_all = json.load(urllib.request.urlopen(url_meta_all))
            meta_all["isSingleFile"] = False
            meta_all_combined[d] = meta_all

        # single-file datasets
        for r in datasets_single:
            url_meta_all = url_repo + r + "/meta_all.json"

            # meta data file for single-file datasets may not exist
            try:
                meta_all = json.load(urllib.request.urlopen(url_meta_all))
            except urllib.error.URLError as e:
                meta_all = dict()

            meta_all["isSingleFile"] = True
            meta_all_combined[r] = meta_all

        # cache index for offline access
        with open(index_path, "w") as f:
            json.dump(meta_all_combined, f)

    except (
        requests.exceptions.RequestException,
        json.JSONDecodeError,
        urllib.error.URLError,
    ):
        warn("Could not fetch global dataset index.")

    try:
        with open(index_path) as index_file:
            return json.load(index_file)
    except (FileNotFoundError, json.JSONDecodeError):
        warn(
            "Global index is not in cache or corrupted. Global datasets will not be accessible."
        )
        return {}


def get_hf_repo_file_list(repo_id: str):
    url_api = f"https://huggingface.co/api/datasets/{repo_id}"
    response = requests.get(url_api)
    response.raise_for_status()
    repo_info = response.json()
    siblings = repo_info.get("siblings", [])
    return [s["rfilename"] for s in siblings]


def dl_parts_from_lrz():
    pass


def fetch_index_from_lrz():
    pass


def print_download_progress(count, block_size, total_size, message=None):
    progress = count * block_size
    percent = int(progress * 100 / total_size)
    bar_length = 50
    bar = (
        "━" * int(percent / 2)
        + DARKGREY
        + "━" * (bar_length - int(percent / 2))
        + SUCCESS_CYAN
    )

    def format_size(size):
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"

    downloaded_str = format_size(progress)
    total_str = format_size(total_size)

    sys.stdout.write(
        SUCCESS_CYAN
        + "\r\033[K"
        + (message if message else f"{downloaded_str} / {total_str}")
        + f"\t {bar} {percent}%"
        + ENDC
    )
    sys.stdout.flush()

    if progress >= total_size:
        sys.stdout.write("\n")
        sys.stdout.flush()
