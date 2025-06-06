# Installation
[The Well](https://polymathic-ai.org/the_well/) can easily be installed on top of the required packages for the [main repository](../../../README.md). Follow the installation insctructions there, and afterwards activate the created environment and install the required packages for The Well in the following way:
```bash
pip install -r src/data/download_well/requirementsWell.txt
```

# Usage
Downloading the datasets is performed via the script `download_well_2d.py`. The script accept the following arguments:

- `--dataset_name`: Specifies the dataset to download.
- ``--split_name`: Specifies the split of the dataset to download. The available splits are `train`, `valid`, and `test`.
- `--out_name`: Specifies the name of the output dataset file.
- `--out_path`: Specifies the directory where the output files will be saved.
- `--reduce_fraction`: Specifies the fraction of the dataset to download. The fraction is applied to the number of simulations in the dataset.
- `-reduce_seed`: Specifies the seed for the random number generator used to select the simulations to download.

`render.py` contains functionality to render the generated data. Running `download_well_2d.py` automatically creates an image with a quick visualization for each downloaded simulation of the dataset. The rendered images are saved in a directory with name `--out_name` to the location specified by `--out_path`.

`download_setups_2d.py` contains the basic parameters for the downloads, which are imported by the download script. It also serves as an overview over supported datasets and parameter values. The name of the datasets in these file are the available values for `--dataset_name` in  `download_well_2d.py`. Adjusting the parameters while considering the well's documentation should be straightforward.

`generate_all_2d.sh` is a bash script that can be used to download all datasets sequentially via calling the python script described above with the respective arguments.

# Downloaded datasets from The Well in 2D [simulations s, time steps t, channels c, size x, size y]
The following PDEs are supported in 2D at the moment. Each 2D simulation uses different initial conditions and physical parameters.

- Turbulent Radiative Layer (`turbulent_radiative_layer_2D`)
    - Train: [s=72, t=101, c=4, x=128, y=384]
    - Valid: [s=9, ...]
    - Test:  [s=9, ...]
    - Channels: Density, Pressure, Velocity X/Y

- Active Matter (`active_matter`)
    - Train: [s=175, t=81, c=11, x=256, y=256]
    - Valid: [s=24, ...]
    - Test:  [s=26, ...]
    - Channels: Concentration, Velocity X/Y, Orientation XX/XY/YX/YY, Strain XX/XY/YX/YY

- Viscoelastic Instability (`viscoelastic_instability`)
    - Train: [s=213, t=20, c=8, x=512, y=512]
    - Valid: [s=22, ...]
    - Test:  [s=22, ...]
    - Channels: Pressure, Conformation ZZ, Velocity X/Y, Conformation XX/XY/YX/YY

- Helmholtz Staircase (`helmholtz_staircase`)
    - Train: [s=416, t=50, c=3, x=1024, y=256]
    - Valid: [s=48, ...]
    - Test:  [s=48, ...]
    - Channels: Pressure (real), Pressure (imaginary), Mask

- Rayleigh Benard (`rayleigh_benard`) [NOTE: only a random 20% split of all available simulations from The Well]
    - Train: [s=280, t=200, c=4, x=512, y=128]
    - Valid: [s=35, ...]
    - Test:  [s=35, ...]
    - Channels: Buoyancy, Pressure, Velocity X/Y

- Shear Flow (`shear_flow`) [NOTE: only a random 10% split of all available simulations from The Well]
    - Train: [s=89, t=200, c=4, x=256, y=512]
    - Valid: [s=11, ...]
    - Test:  [s=11, ...]
    - Channels: Density, Pressure, Velocity X/Y

- [NOT DOWNLOADED] Euler Multi Quadrant (`euler_multi_quadrants_periodicBC` and `euler_multi_quadrants_openBC`)
    - CURRENTLY NOT AVAILABLE ON HUGGINGFACE; BUT SHOULD BE POSSIBLE WITH MANUAL DOWNLOAD ONCE CURL IS INSTALLED
    - Train: [s=, t=100, c=5, x=512, y=512]
    - Valid: [s=, ...]
    - Test:  [s=, ...]
    - Channels: Density, Energy, Pressure, Velocity X/Y

The individual parameters for each downloaded dataset are described in more detail in `download_setups_2d.py`.