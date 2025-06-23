# Installation
The [exponax package](https://pypi.org/project/exponax/) requires [jax](https://jax.readthedocs.io/en/latest/installation.html), and it is recommended to install the simulation to a separate conda environment. To install follow these steps:
```bash
conda create -n exponax python=3.12
conda activate exponax
```
Install required packages:
```bash
pip install -r pdetransformer/data/simulations_apebench/requirementsExponax.txt
```


# Usage
The generation of volumetric PDE datasets is performed via the script `simulation.py`. It accepts the following arguments:

- `--pde`: Specifies the type of partial differential equation (PDE) to be solved.
- `--out_name`: Specifies the name of the output dataset file.
- `--out_path`: Specifies the directory where the output files will be saved.
- `--num_sims`: Specifies the number of simulations to be performed.
- `--render_only`: If this argument is provided, the script will only render the volumes without saving the data.
- `--gpu_id`: Specifies the ID of the GPU to be used for computation.

`render.py` contains functionality to render the generated data. Running `simulation.py` automatically creates a single image with different visualizations for each simulation of the dataset during generation. The rendered images are saved in a directory with name `--out_name` to the location specified by `--out_path`.

`simulation_setups.py` contains the basic setups for the PDE simulation, which are imported by `simulation.py`. It also servers as a overview over supported PDEs and parameter values. The name of the simulation types in this file are the available values for `--pde` in `simulation.py`. The corresponding function for each PDE returns fixed and varied simulation parameters, an exponax stepper that performs the simulation, and a initial condition tensor. Expanding or altering the simulation setups is straightforward by extending or modifying this file. Note that the choices for the varied parameter ranges for each PDE are optimized for the given resolution, domain size, time steps, substeps, etc. and changing one of these values may require adjusting the others.

`generate_all_2d.sh` and `generate_all_3d.sh` are bash script that can be used to generate all datasets sequentially via calling the python scripts described above with the respective arguments.

# Supported PDEs in 2D
The following PDEs are supported in 2D:

- Advection (`adv`)
    - Data: [s=60, t=30, c=1, x=2048, y=2048]
    - Channels: Density
    - Varied Parameters: Velocity X/Y (all randomly), ICs

- Diffusion (`diff`)
    - Data: [s=60, t=30, c=1, x=2048, y=2048]
    - Channels: Density
    - Varied Parameters: Viscosity X/Y (all randomly), ICs

- Advection-Diffusion (`adv_diff`)
    - Data: [s=60, t=30, c=1, x=2048, y=2048]
    - Channels: Density
    - Varied Parameters: Velocity X/Y, Viscosity X/Y (all randomly), ICs

- Dispersion (`disp`)
    - Data: [s=60, t=30, c=1, x=2048, y=2048]
    - Channels: Density
    - Varied Parameters: Dispersivity X/Y (all randomly), ICs

- Hyper-Diffusion (`hyp_diff`)
    - Data: [s=60, t=30, c=1, x=2048, y=2048]
    - Channels: Density
    - Varied Parameters: Hyper-Diffusivity (randomly), ICs

- Burgers Equation (`burgers`)
    - Data: [s=60, t=30, c=2, x=2048, y=2048]
    - Channels: Velocity X/Y
    - Varied Parameters: Viscosity (randomly), ICs

- Korteweg-de Vries Equation (`kdv`)
    - Data: [s=60, t=30, c=2, x=2048, y=2048]
    - Channels: Velocity X/Y
    - Varied Parameters: Domain Extent, Viscosity (all randomly), ICs

- Kuramoto-Sivashinsky Equation (`ks`)
    - Data: [s=60, t=30, c=1, x=2048, y=2048]
    - Channels: Density
    - Varied Parameters: Domain Extent (randomly), ICs
    - Longer Rollout Test set with [s=5, t=200, c=1, x=2048, y=2048]

- Fisher-KPP Equation (`fisher`)
    - Data: [s=60, t=30, c=1, x=2048, y=2048]
    - Channels: Concentration
    - Varied Parameters: Diffusivity, Reactivity (randomly), ICs

- Gray-Scott Equation with various configurations that exhibit substantially different behavior (`gs_alpha`, `gs_beta`, `gs_gamma`, `gs_delta`, `gs_epsilon`, `gs_theta`, `gs_iota`, `gs_kappa`)
    - Data (each configuration): [s=10, t=30, c=2, x=2048, y=2048]
    - Channels (each configuration): Concentration A, Concentration B
    - Varied Parameters (each configuration): ICs
    - Longer Rollout Test sets for `gs_alpha`, `gs_beta`, `gs_gamma`, `gs_epsilon` with [s=3, t=100, c=2, x=2048, y=2048]

- Swift-Hohenberg Equation (`sh`)
    - Data: [s=60, t=30, c=1, x=2048, y=2048]
    - Channels: Concentration
    - Varied Parameters: Reactivity, Critical Number (randomly), ICs

- Navier-Stokes Equations: Decaying Turbulence (`decay_turb`)
    - Data: [s=60, t=30, c=1, x=2048, y=2048]
    - Channels: Vorticity
    - Varied Parameters: Viscosity (randomly), ICs
    - Longer Rollout Test set with [s=5, t=200, c=1, x=2048, y=2048]

- Navier-Stokes Equations: Kolmogorov Flow (`kolm_flow`)
    - Data: [s=60, t=30, c=1, x=2048, y=2048]
    - Channels: Vorticity
    - Varied Parameters: Viscosity (randomly), ICs
    - Longer Rollout Test set with [s=5, t=200, c=1, x=2048, y=2048]

The individual parameters for each PDE setup are also described in more detail in `simulation_setups_2d.py`.

# Supported PDEs in 3D
The following PDEs are supported in 3D:

- Advection (`adv`)
    - Data: [s=60, t=30, c=1, x=384, y=384, z=384]
    - Channels: Density
    - Varied Parameters: Velocity X/Y/Z (all randomly), ICs

- Diffusion (`diff`)
    - Data: [s=60, t=30, c=1, x=384, y=384, z=384]
    - Channels: Density
    - Varied Parameters: Viscosity X/Y/Z (all randomly), ICs

- Advection-Diffusion (`adv_diff`)
    - Data: [s=60, t=30, c=1, x=384, y=384, z=384]
    - Channels: Density
    - Varied Parameters: Velocity X/Y/Z, Viscosity X/Y/Z (all randomly), ICs

- Dispersion (`disp`)
    - Data: [s=60, t=30, c=1, x=384, y=384, z=384]
    - Channels: Density
    - Varied Parameters: Dispersivity X/Y/Z (all randomly), ICs

- Hyper-Diffusion (`hyp_diff`)
    - Data: [s=60, t=30, c=1, x=384, y=384, z=384]
    - Channels: Density
    - Varied Parameters: Hyper-Diffusivity (randomly), ICs

- Burgers Equation (`burgers`)
    - Data: [s=60, t=30, c=3, x=384, y=384, z=384]
    - Channels: Velocity X/Y/Z
    - Varied Parameters: Viscosity (randomly), ICs

- Korteweg-de Vries Equation (`kdv`)
    - Data: [s=60, t=30, c=3, x=384, y=384, z=384]
    - Channels: Velocity X/Y/Z
    - Varied Parameters: Domain Extent, Viscosity (all randomly), ICs

- Kuramoto-Sivashinsky Equation (`ks`)
    - Data: [s=60, t=30, c=1, x=384, y=384, z=384]
    - Channels: Density
    - Varied Parameters: Domain Extent (randomly), ICs
    - Longer Rollout Test set with [s=5, t=200, c=1, x=384, y=384, z=384]

- Fisher-KPP Equation (`fisher`)
    - Data: [s=60, t=30, c=1, x=384, y=384, z=384]
    - Channels: Concentration
    - Varied Parameters: Diffusivity, Reactivity (randomly), ICs

- Gray-Scott Equation with various configurations that exhibit substantially different behavior (`gs_alpha`, `gs_beta`, `gs_gamma`, `gs_delta`, `gs_epsilon`, `gs_theta`, `gs_iota`, `gs_kappa`)
    - Data (each configuration): [s=10, t=30, c=2, x=320, y=320, z=320]
    - Channels (each configuration): Concentration A, Concentration B
    - Varied Parameters (each configuration): ICs
    - Longer Rollout Test sets for `gs_alpha`, `gs_beta`, `gs_gamma` with [s=3, t=100, c=2, x=320, y=320, z=320] and for `gs_epsilon` [s=5, t=100, c=2, x=320, y=320, z=320]

- Swift-Hohenberg Equation (`sh`)
    - Data: [s=60, t=30, c=1, x=384, y=384, z=384]
    - Channels: Concentration
    - Varied Parameters: Reactivity, Critical Number (randomly), ICs

The individual parameters for each PDE setup are also described in more detail in `simulation_setups_3d.py`.