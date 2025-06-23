# The Well Datasets

The Well datasets are a collection of physics-based simulations from [The Well](https://polymathic-ai.org/the_well/) project. These datasets provide high-quality simulations of various physical phenomena for machine learning research.

## Installation

The Well datasets can be installed on top of the required packages for the main repository. After following the main repository's installation instructions, activate the created environment and install The Well's required packages:

```bash
pip install the_well
```

## Usage

Downloading the datasets is performed via the script `download_well_2d.py`. The script accepts the following arguments:

- `--dataset_name`: Specifies the dataset to download
- `--split_name`: Specifies the split of the dataset to download (`train`, `valid`, or `test`)
- `--out_name`: Specifies the name of the output dataset file
- `--out_path`: Specifies the directory where the output files will be saved
- `--reduce_fraction`: Specifies the fraction of the dataset to download (applied to number of simulations)
- `--reduce_seed`: Specifies the seed for random selection of simulations

For example, to download the training split of the turbulent radiative layer dataset:

```bash
python download_well_2d.py --dataset_name turbulent_radiative_layer_2D --split_name train --out_name turb_rad_layer --out_path ./datasets
```

The script will automatically create visualization images for each downloaded simulation in a directory named after your output name.

!!! note
    Make sure the datasets are stored in the folder linked in the environment YAML file `env/local.yaml` in `paths.PBDL_index.2D_WELL`. This is where the dataloader will look for them by default.

## Available Datasets

The following datasets are available in 2D:

### Turbulent Radiative Layer (`turbulent_radiative_layer_2D`)
- **Data Shape**: [s=72, t=101, c=4, x=128, y=384]
- **Channels**: Density, Pressure, Velocity X/Y
- **Splits**: Train (72 sims), Valid (9 sims), Test (9 sims)

### Active Matter (`active_matter`)
- **Data Shape**: [s=175, t=81, c=11, x=256, y=256]
- **Channels**: Concentration, Velocity X/Y, Orientation XX/XY/YX/YY, Strain XX/XY/YX/YY
- **Splits**: Train (175 sims), Valid (24 sims), Test (26 sims)

### Viscoelastic Instability (`viscoelastic_instability`)
- **Data Shape**: [s=213, t=20, c=8, x=512, y=512]
- **Channels**: Pressure, Conformation ZZ, Velocity X/Y, Conformation XX/XY/YX/YY
- **Splits**: Train (213 sims), Valid (22 sims), Test (22 sims)

### Helmholtz Staircase (`helmholtz_staircase`)
- **Data Shape**: [s=416, t=50, c=3, x=1024, y=256]
- **Channels**: Pressure (real), Pressure (imaginary), Mask
- **Splits**: Train (416 sims), Valid (48 sims), Test (48 sims)

### Rayleigh Benard (`rayleigh_benard`)
- **Data Shape**: [s=280, t=200, c=4, x=512, y=128]
- **Channels**: Buoyancy, Pressure, Velocity X/Y
- **Splits**: Train (280 sims), Valid (35 sims), Test (35 sims)
- **Note**: Only a random 20% split of all available simulations from The Well

### Shear Flow (`shear_flow`)
- **Data Shape**: [s=89, t=200, c=4, x=256, y=512]
- **Channels**: Density, Pressure, Velocity X/Y
- **Splits**: Train (89 sims), Valid (11 sims), Test (11 sims)
- **Note**: Only a random 10% split of all available simulations from The Well

The individual parameters for each downloaded dataset are described in more detail in `download_setups_2d.py`.

## Data Format

Each dataset consists of:

- Multiple simulations (number varies by dataset)
- Each simulation contains multiple timesteps (varies by dataset)
- Each timestep has multiple channels (varies by dataset)
- Spatial resolution varies by dataset

## Visualization

The `render.py` script provides functionality to visualize the downloaded data. Running `download_well_2d.py` automatically creates visualization images for each simulation during download.

## Batch Download

A bash script `download_all_2d.sh` is provided to download all datasets sequentially with appropriate arguments. 