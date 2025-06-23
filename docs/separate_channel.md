# Separate Channels Version of PDE-Transformer

The separate channels (SC) version of PDE-Transformer embeds different physical channels independently, learning a more disentangled representation. This approach offers several advantages and trade-offs compared to the mixed channels version.

## Key Characteristics

### Architecture
- Embeds different physical channels independently as separate tokens
- Uses channel-wise self-attention for interaction between channels
- Maintains distinct representations for each physical quantity

### Foundation Models and Transfer Learning

- Significantly improved transferability
- Can be adapted to different simulation setups in 2D
- Model architecture allows joint learning of input/output pairs, which can be extended to many different tasks like data assimilation or inverse problems
- More conditioning options like PDE parameters or channel types

### Limitations

The separate channels incurs an increased computational overhead compared to the mixed channels approach. This includes a higher memory footprint, more complex attention mechanisms that increase computation time, slower training and inference speeds, and an increased number of model parameters.

## Initialization

Here's how to initialize the PDETransformer model with separate channels:

```python
from pdetransformer.core.separate_channels import PDETransformer

# Initialize the model
model = PDETransformer(
    sample_size=256,        
    num_timesteps=2,
    type="PDE-S",        
    periodic=True,          
    carrier_token_active=False,
    patch_size=4,
)
```

### Parameter Explanation

- `sample_size`: The spatial dimensions of the input/output grid. This is a positional argument, but it is not used right now and only there to provide a unified initialization of different models. Therefore it can be set to an arbitrary value and the model can be applied to grids of variable sizes.
- `timesteps`: How many timesteps are used as model input/output. For autoregressive prediction with 1 time input and 1 time output, there are in total 2 timesteps when considering the joint pair of input/output time states. 
- `type`: Defines the model config, i.e. PDE-S, PDE-B or PDE-L
- `periodic`: Whether to use periodic boundary conditions in the simulation
- `carrier_token_active`: Whether to use carrier tokens for global information exchange. This enables hierarchical attention (https://arxiv.org/abs/2306.06189). Not compatible with periodic boundary conditions at the moment; default: False.
- `patch_size`: Size of patches used for token embedding (smaller patches preserve more spatial detail but increase computation)

## Usage

Here's an example of how to use the model for forward prediction:

```python
import torch

# Prepare input data
batch_size = 4
height, width = 64, 64
num_timesteps = 2

# Input tensor: list of tensors with shape (B, T, H, W)
# Each tensor represents a different physical channel (e.g., velocity, density)
x = [
    torch.randn(batch_size, num_timesteps, height, width),  # Channel 1
    torch.randn(batch_size, num_timesteps, height, width),  # Channel 2
]

# Simulation time: list of tensors with shape (B,)
simulation_time = [torch.tensor([0.0] * batch_size)] * 2

# Channel type: list of tensors with shape (B,)
# Identifies the type of each channel (e.g., velocity, density)
channel_type = [torch.tensor([0] * batch_size).int(), torch.tensor([1] * batch_size).int()]

# PDE type: list of tensors with shape (B,)
# Identifies the type of PDE being solved
pde_type = [torch.tensor([0] * batch_size).int()] * 2

# PDE parameters: list of tensors with shape (B, num_pde_parameters)
# Contains physical parameters of the PDE (e.g., viscosity, diffusion coefficient)
pde_parameters = [torch.randn(batch_size, 5)] * 2 # Assuming 10 PDE parameters

# PDE parameters class: list of tensors with shape (B, num_pde_parameters)
# Categorical encoding of PDE parameters
pde_parameters_class = [torch.zeros(batch_size, 5).int()] * 2

# Simulation timestep: list of tensors with shape (B,)
simulation_dt = [torch.ones(batch_size)] * 2

# Task: list of tensors with shape (B,). 
task = [torch.zeros(batch_size).int()] * 2  

# Timestep: list of tensors with shape (B,)
# Current timestep in the diffusion process
t = [torch.zeros(batch_size)] * 2

# Forward pass
output = model(
    x=x,
    simulation_time=simulation_time,
    channel_type=channel_type,
    pde_type=pde_type,
    pde_parameters=pde_parameters,
    pde_parameters_class=pde_parameters_class,
    simulation_dt=simulation_dt,
    task=task,
    t=t,
    return_dict=True
)

print('Length output.sample: ', len(output.sample))
print('Shape: ', output.sample[0].shape)
```

### Forward Pass Parameters

- `x`: List of input tensors, each with shape (B, T, H, W)
    - B: Batch size
    - T: Number of timesteps
    - H, W: Height and width of the spatial grid
    - Each tensor in the list represents a different physical channel

- `simulation_time`: List of tensors with shape (B,). Simulation time for each sample in the batch. Can be set to 0 for all samples.

- `channel_type`: List of tensors with shape (B,). Integer identifiers for each channel type. Used to distinguish between different physical quantities. See the table below for an overview of the different channels. 

- `pde_type`: List of tensors with shape (B,). Integer identifier for the type of PDE being solved. Used for conditioning the model on the specific PDE. Special identifiers if PDE is unknown. 

- `pde_parameters`: List of tensors with shape (B, num_pde_parameters). Up to `num_pde_parameters=5` physical parameter values of the PDE. Examples: viscosity, diffusion coefficient, etc. See the table below for an overview of the different PDE parameters.

- `pde_parameters_class`: List of tensors with shape (B, num_pde_parameters). Integer identifier for the PDE parameter type. Used for conditioning the model.

- `simulation_dt`: List of tensors with shape (B,). Timestep size for the simulation timesteps. Used for temporal conditioning; fixed to a standard value in our tests, but can be varied to train with different lead times. 

- `task`: List of tensors with shape (B,). Identifies the task the model should solve. The value 0 is used for autoregressive prediction (given timestep 0, predict timestep 1 for 2 timesteps). Can be extended to multiple tasks, e.g. inpainting, denoising, interpolation, etc. 

- `t`: List of tensors with shape (B,). Currently used as time in the diffusion process when training PDE-Transformer as a probabilistic model. 

- `return_dict`: Boolean
    - If True, returns output as a dictionary
    - If False, returns output as a `PDETransformerOutput` object.

## Table of Channel Types

| Channel Type | Label ID | Description |
|--------------|----------|-------------|
| Velocity | 0 | General velocity field |
| Velocity X | 1 | X-component of velocity |
| Velocity Y | 2 | Y-component of velocity |
| Velocity Z | 3 | Z-component of velocity |
| Vorticity | 4 | Curl of velocity field |
| Density | 5 | Mass per unit volume |
| Pressure | 6 | Force per unit area |
| Concentration | 7 | General concentration field |
| Concentration A | 8 | First species concentration |
| Concentration B | 9 | Second species concentration |
| Magnetic Field X | 10 | X-component of magnetic field |
| Magnetic Field Y | 11 | Y-component of magnetic field |
| Magnetic Field Z | 12 | Z-component of magnetic field |
| Vector Potential X | 13 | X-component of vector potential |
| Vector Potential Y | 14 | Y-component of vector potential |
| Vector Potential Z | 15 | Z-component of vector potential |
| Orientation XX | 16 | XX-component of orientation tensor |
| Orientation XY | 17 | XY-component of orientation tensor |
| Orientation YX | 18 | YX-component of orientation tensor |
| Orientation YY | 19 | YY-component of orientation tensor |
| Strain XX | 20 | XX-component of strain tensor |
| Strain XY | 21 | XY-component of strain tensor |
| Strain YX | 22 | YX-component of strain tensor |
| Strain YY | 23 | YY-component of strain tensor |
| Conformation XX | 24 | XX-component of conformation tensor |
| Conformation XY | 25 | XY-component of conformation tensor |
| Conformation YX | 26 | YX-component of conformation tensor |
| Conformation YY | 27 | YY-component of conformation tensor |
| Conformation ZZ | 28 | ZZ-component of conformation tensor |
| Pressure (Real) | 29 | Real part of pressure field |
| Pressure (Imaginary) | 30 | Imaginary part of pressure field |
| Mask | 31 | Binary mask field |
| Buoyancy | 32 | Buoyancy force field |
| Energy | 33 | Energy density field |
| Deformation XX | 34 | XX-component of deformation tensor |
| Deformation YY | 35 | YY-component of deformation tensor |
| Deformation ZZ | 36 | ZZ-component of deformation tensor |

## Table of PDE Parameter Classes 

| Parameter Type | Label ID | Description |
|--------------|----------|-------------|
| Unknown | 0 | Unspecified parameter type |
| Reynolds Number | 1 | Ratio of inertial to viscous forces |
| Mach Number | 2 | Ratio of flow velocity to speed of sound |
| Z Slice | 3 | Position of 2D slice in 3D domain |
| Velocity X | 4 | X-component of velocity parameter |
| Velocity Y | 5 | Y-component of velocity parameter |
| Velocity Z | 6 | Z-component of velocity parameter |
| Viscosity | 7 | Fluid's resistance to deformation |
| Viscosity X | 8 | X-component of viscosity |
| Viscosity Y | 9 | Y-component of viscosity |
| Viscosity Z | 10 | Z-component of viscosity |
| Dispersivity X | 11 | X-component of dispersivity |
| Dispersivity Y | 12 | Y-component of dispersivity |
| Dispersivity Z | 13 | Z-component of dispersivity |
| Hyper-Diffusivity | 14 | Higher-order diffusion coefficient |
| Domain Extent | 15 | Size of computational domain |
| Diffusivity | 16 | Rate of diffusive transport |
| Reactivity | 17 | Rate of chemical reaction |
| Feed Rate | 18 | Rate of species addition |
| Kill Rate | 19 | Rate of species removal |
| Critical Number | 20 | Threshold parameter |
| Cooling Time | 21 | Characteristic time for cooling |
| Particle Alignment Strength | 22 | Strength of particle orientation |
| Active Dipol Strength | 23 | Strength of active dipole |
| Weissenberg Number | 24 | Ratio of elastic to viscous forces |
| Viscosity Ratio | 25 | Ratio between viscosities |
| Kolmogorov Length Scale | 26 | Smallest length scale in turbulence |
| Maximum Polymer Extensibility | 27 | Maximum polymer stretch |
| Frequency | 28 | Oscillation frequency |
| Rayleigh Number | 29 | Ratio of buoyancy to viscous forces times thermal diffusion |
| Prandtl Number | 30 | Ratio of kinematic viscosity to thermal diffusivity |
| Schmidt Number | 31 | Ratio of kinematic viscosity to mass diffusivity |
| Gas Constant | 32 | Constant relating energy, temperature and amount of substance |
| Deformation XX | 33 | XX-component of deformation parameter |
| Deformation YY | 34 | YY-component of deformation parameter |
| Deformation ZZ | 35 | ZZ-component of deformation parameter |


!!! note
    New field types and parameter classes should only be added at the end of their respective lists to maintain consistent encoding across versions. 



### Example Notebook

An example notebook how to run inference for pretrained PDE-Transformer and additional explanations/code examples can be found at 
[notebooks/visualization_sc_ape2d.ipynb](https://github.com/tum-pbs/pde-transformer/blob/main/pdetransformer/notebooks/visualization_sc_ape2d.ipynb).