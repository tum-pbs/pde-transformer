import math
import torch

def normalize_positive_value(x:float) -> float:
    # normalizes x to ensure it lies in the range ]0, 1[
    if x <= 0:
        return ValueError("Input value must be strictly positive!")
    return x / (1 + x) # equivalent to 1 / (1 + (1/x)) or 1 / (1 + e^(-ln(x))), i.e. sigmoid of ln(x)

def normalize_value(x:float) -> float:
    if x < 0:
        return -normalize_positive_value(-x)
    elif x > 0:
        return normalize_positive_value(x)
    else:
        return 0.0


all_pdes = [
        "unknown",
        "advection",
        "diffusion",
        "advection-diffusion",
        "dispersion",
        "hyper-diffusion",
        "burgers",
        "korteweg-de-vries",
        "kuramoto-sivashinsky",
        "fisher-kpp",
        "gray scott",
        "swift-hohenberg",
        "navier-stokes: incompressible cylinder flow",
        "navier-stokes: compressible cylinder flow",
        "navier-stokes: isotropic turbulence",
        "navier-stokes: decaying turbulence",
        "navier-stokes: kolmogorov flow",
        "navier-stokes: channel flow",
        "navier-stokes: transitional boundary layer",
        "magnetohydrodynamic turbulence",
        "turbulent radiative layer",
        "active matter",
        "viscoelastic instability",
        "helmholtz",
        "rayleigh benard convection",
        "navier-stokes: shear flow",
        "euler",
        "piso: channel flow"
        # NOTE: only add new PDEs to the end of this list to ensure a consistent encoding!
    ]

def convert_pde(pde:str) -> torch.Tensor:
    '''
    Convert PDE string to tensor with integer index

    Args:
        pde: PDE string
    Return:
        encoded tensor
    '''

    result = all_pdes.index(pde.lower())
    return torch.Tensor([result]).float()


def convert_fields(fields:list[str]) -> torch.Tensor:
    '''
    Convert list of fields to tensor with integer indices

    Args:
        fields: list of fields
    Return:
        encoded tensor'''
    all_fields = [
        "unknown"
        "velocity",
        "velocity x",
        "velocity y",
        "velocity z",
        "vorticity",
        "density",
        "pressure",
        "concentration",
        "concentration a",
        "concentration b",
        "magnetic field x",
        "magnetic field y",
        "magnetic field z",
        "vector potential x",
        "vector potential y",
        "vector potential z",
        "orientation xx",
        "orientation xy", 
        "orientation yx",
        "orientation yy",
        "strain xx",
        "strain xy",
        "strain yx",
        "strain yy",
        "conformation xx",
        "conformation xy",
        "conformation yx",
        "conformation yy",
        "conformation zz",
        "pressure (real)",
        "pressure (imaginary)",
        "mask",
        "buoyancy",
        "energy",
        "deformation xx",
        "deformation yy",
        "deformation zz",
        # NOTE: only add new fields to the end of this list to ensure a consistent encoding!
    ]

    result = [all_fields.index(field.lower()) for field in fields]
    return torch.tensor(result).float()


def convert_constants(constants:list[str]) -> torch.Tensor:
    '''
    Convert list of constants to tensor with integer indices

    Args:
        constants: list of constants
    Return:
        encoded tensor
    '''
    all_constants = [
        "unknown",
        "reynolds number",
        "mach number",
        "z slice",
        "velocity x",
        "velocity y",
        "velocity z",
        "viscosity",
        "viscosity x",
        "viscosity y",
        "viscosity z",
        "dispersivity x",
        "dispersivity y",
        "dispersivity z",
        "hyper-diffusivity",
        "domain extent",
        "diffusivity",
        "reactivity",
        "feed rate",
        "kill rate",
        "critical number",
        "cooling time",
        "particle alignment strength",
        "active dipol strength",
        "weissenberg number",
        "viscosity ratio",
        "kolmogorov length scale",
        "maximum polymer extensibility",
        "frequency",
        "rayleigh number",
        "prandtl number",
        "schmidt number",
        "gas constant",
        "deformation xx",
        "deformation yy",
        "deformation zz",
        # NOTE: only add new constants to the end of this list to ensure a consistent encoding!
    ]

    result = [all_constants.index(constant.lower()) for constant in constants]
    return torch.tensor(result).float()


def convert_boundary_conditions(boundary_conditions:list[str], boundary_condition_order:list[str]) -> torch.Tensor:
    '''
    Convert list of boundary conditions to tensor with integer indices while respecting the boundary condition order

    Args:
        boundary_conditions: list of boundary conditions
        boundary_condition_order: list of boundary condition orders
    Return:
        encoded tensor
    '''

    assert len(boundary_conditions) == len(boundary_condition_order), "Boundary conditions and order must have the same length"

    result = torch.zeros(len(boundary_conditions)).float()

    for i in range(len(boundary_conditions)):
        result = update_boundary_condition(result, boundary_conditions[i], boundary_condition_order[i], update_opposite=False)

    return result


def update_boundary_condition(tensor:torch.Tensor, boundary_condition:str, boundary_condition_direction:str, update_opposite:bool=True) -> torch.Tensor:
    '''
    Update a tensor with boundary conditions at the position specified by the boundary condition direction

    Args:
        tensor: tensor to be updated
        boundary_condition: boundary condition to be encoded
        boundary_condition_direction: direction of the boundary condition
    Return:
        updated tensor
    '''

    all_boundary_conditions = [
        "periodic",
        "open",
        "wall",
        "inflow",
        # NOTE: only add new boundary conditions to the end of this list to ensure a consistent encoding!
    ]
    target_order = [
        "x negative",
        "x positive",
        "y negative",
        "y positive",
        "z negative",
        "z positive",
        # NOTE: for 2D and 3D data, keep this order unchanged for a consistent encoding!
    ]
    opposite_direction = {
        "x negative": "x positive",
        "x positive": "x negative",
        "y negative": "y positive",
        "y positive": "y negative",
        "z negative": "z positive",
        "z positive": "z negative",
    }

    idx = target_order.index(boundary_condition_direction)
    if idx > tensor.shape[0]:
        raise ValueError(f"Boundary condition direction {boundary_condition_direction} not possible for tensor with shape {tensor.shape}")

    tensor[idx] = all_boundary_conditions.index(boundary_condition)
    if not update_opposite:
        return tensor

    # if the boundary conditions were periodic, update the opposite boundary condition as well
    opposite_boundary_condition_idx = target_order.index(opposite_direction[boundary_condition_direction])
    opposite_boundary_condition = all_boundary_conditions[tensor[opposite_boundary_condition_idx].int().item()]

    if boundary_condition != "periodic" and opposite_boundary_condition == "periodic":
        tensor[opposite_boundary_condition_idx] = all_boundary_conditions.index("open")
    if boundary_condition == "periodic" and opposite_boundary_condition != "periodic":
        tensor[opposite_boundary_condition_idx] = all_boundary_conditions.index("periodic")
    return tensor


def convert_domain_extent(extent:list[float]) -> torch.Tensor:
    result = [normalize_value(e) for e in extent]
    return torch.tensor(result).float()

def convert_dt(dt:float) -> torch.Tensor:
    result = normalize_value(dt)
    return torch.Tensor([result]).float()

def convert_reynolds_number(reynolds_number:float) -> torch.Tensor:
    result = normalize_value(reynolds_number)
    return torch.Tensor([result]).float()