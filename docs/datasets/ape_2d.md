# APEbench 2D Datasets

The APEbench 2D datasets consist of various partial differential equation (PDE) simulations in two dimensions. These datasets are generated using the [exponax](https://pypi.org/project/exponax/) package and provide a comprehensive collection of physics-based simulations for machine learning research.

## Installation

To generate these datasets, you'll need to set up a separate conda environment with the required dependencies:

```bash
conda create -n exponax python=3.12
conda activate exponax
pip install -r pdetransformer/data/simulations_apebench/requirementsExponax.txt
```

## Available Datasets

The following PDEs are supported in 2D:

### Advection (`adv`)
- **Data Shape**: [s=60, t=30, c=1, x=2048, y=2048]
- **Channels**: Density
- **Varied Parameters**: Velocity X/Y (random), Initial Conditions

### Diffusion (`diff`)
- **Data Shape**: [s=60, t=30, c=1, x=2048, y=2048]
- **Channels**: Density
- **Varied Parameters**: Viscosity X/Y (random), Initial Conditions

### Advection-Diffusion (`adv_diff`)
- **Data Shape**: [s=60, t=30, c=1, x=2048, y=2048]
- **Channels**: Density
- **Varied Parameters**: Velocity X/Y, Viscosity X/Y (random), Initial Conditions

### Dispersion (`disp`)
- **Data Shape**: [s=60, t=30, c=1, x=2048, y=2048]
- **Channels**: Density
- **Varied Parameters**: Dispersivity X/Y (random), Initial Conditions

### Hyper-Diffusion (`hyp_diff`)
- **Data Shape**: [s=60, t=30, c=1, x=2048, y=2048]
- **Channels**: Density
- **Varied Parameters**: Hyper-Diffusivity (random), Initial Conditions

### Burgers Equation (`burgers`)
- **Data Shape**: [s=60, t=30, c=2, x=2048, y=2048]
- **Channels**: Velocity X/Y
- **Varied Parameters**: Viscosity (random), Initial Conditions

### Korteweg-de Vries Equation (`kdv`)
- **Data Shape**: [s=60, t=30, c=2, x=2048, y=2048]
- **Channels**: Velocity X/Y
- **Varied Parameters**: Domain Extent, Viscosity (random), Initial Conditions

### Kuramoto-Sivashinsky Equation (`ks`)
- **Data Shape**: [s=60, t=30, c=1, x=2048, y=2048]
- **Channels**: Density
- **Varied Parameters**: Domain Extent (random), Initial Conditions
- **Longer Rollout Test Set**: [s=5, t=200, c=1, x=2048, y=2048]

### Fisher-KPP Equation (`fisher`)
- **Data Shape**: [s=60, t=30, c=1, x=2048, y=2048]
- **Channels**: Concentration
- **Varied Parameters**: Diffusivity, Reactivity (random), Initial Conditions

### Gray-Scott Equation
Multiple configurations available: `gs_alpha`, `gs_beta`, `gs_gamma`, `gs_delta`, `gs_epsilon`, `gs_theta`, `gs_iota`, `gs_kappa`

- **Data Shape**: [s=10, t=30, c=2, x=2048, y=2048]
- **Channels**: Concentration A, Concentration B
- **Varied Parameters**: Initial Conditions
- **Longer Rollout Test Sets**: Available for `gs_alpha`, `gs_beta`, `gs_gamma`, `gs_epsilon` with [s=3, t=100, c=2, x=2048, y=2048]

### Swift-Hohenberg Equation (`sh`)
- **Data Shape**: [s=60, t=30, c=1, x=2048, y=2048]
- **Channels**: Concentration
- **Varied Parameters**: Reactivity, Critical Number (random), Initial Conditions

### Navier-Stokes: Decaying Turbulence (`decay_turb`)
- **Data Shape**: [s=60, t=30, c=1, x=2048, y=2048]
- **Channels**: Vorticity
- **Varied Parameters**: Viscosity (random), Initial Conditions
- **Longer Rollout Test Set**: [s=5, t=200, c=1, x=2048, y=2048]

### Navier-Stokes: Kolmogorov Flow (`kolm_flow`)
- **Data Shape**: [s=60, t=30, c=1, x=2048, y=2048]
- **Channels**: Vorticity
- **Varied Parameters**: Viscosity (random), Initial Conditions
- **Longer Rollout Test Set**: [s=5, t=200, c=1, x=2048, y=2048]

## Data Generation

To generate these datasets, use the `simulation.py` script with the following arguments:

```bash
python simulation.py --pde <pde_type> --out_name <output_name> --out_path <output_path> --num_sims <number_of_simulations> [--gpu_id <gpu_id>]
```

For example, to generate the advection dataset:

```bash
python simulation.py --pde adv --out_name adv --out_path ./datasets --num_sims 100
```

The script will automatically create visualization images for each simulation in a directory named after your output name.

!!! note
    Make sure the datasets are stored in the folder linked in the environment YAML file `env/local.yaml` in `paths.PBDL_index.2D_APE`. This is where the dataloader will look for them by default.

## Data Format

Each dataset consists of:

- Multiple simulations (specified by `--num_sims`)
- Each simulation contains multiple timesteps (t=30 by default)
- Each timestep has one or more channels (c=1 or c=2 depending on the PDE)
- The spatial resolution is 2048x2048 for all 2D datasets. For training/inference we downsample the spatial resolution to 256x256.

## Simulation Type Label

The PDE type is encoded as a class label in the metadata, which can be used as an input to the model. The mapping is shown in the following table:

| PDE Type | Class Label | Dataset tag |
|----------|-------------|----------------------|
| Advection | 1 | `adv` |
| Diffusion | 2 | `diff` |
| Advection-Diffusion | 3 | `adv_diff` |
| Dispersion | 4 | `disp` |
| Hyper-Diffusion | 5 | `hyp_diff` |
| Burgers Equation | 6 | `burgers` |
| Korteweg-de Vries | 7 | `kdv` |
| Kuramoto-Sivashinsky | 8 | `ks` |
| Fisher-KPP | 9 | `fisher` |
| Gray-Scott | 10 | `gs_*` |
| Swift-Hohenberg | 11 | `sh` |
| Navier-Stokes: Decaying Turbulence | 15 | `decay_turb` |
| Navier-Stokes: Kolmogorov Flow | 16 | `kolm_flow` |

!!! note
    A label class value of `num_classes=1000` is used as a special indicator that the PDE type is unknown and needs to be determined from the data itself.

## Visualization

The `render.py` script provides functionality to visualize the generated data. Running `simulation.py` automatically creates visualization images for each simulation during generation. 