import numpy as np
import math

def get_setup(name)-> tuple[dict, dict, str]:
    if name == "turbulent_radiative_layer_2D":
        return get_turbulent_radiative_layer_2d()
    elif name == "active_matter":
        return get_active_matter()
    elif name == "viscoelastic_instability":
        return get_viscoelastic_instability()
    elif name == "helmholtz_staircase":
        return get_helmholtz_staircase()
    elif name == "rayleigh_benard":
        return get_rayleigh_benard()
    elif name == "shear_flow":
        return get_shear_flow()
    elif name == "euler_multi_quadrants_periodicBC":
        raise ValueError("CURRENTLY NOT AVAILABLE ON HUGGINFACE")
        return get_euler_multi_quadrants_periodicBC()
    elif name == "euler_multi_quadrants_openBC":
        raise ValueError("CURRENTLY NOT AVAILABLE ON HUGGINFACE")
        return get_euler_multi_quadrants_openBC()
    else:
        raise ValueError("Invalid simulation name")


def get_turbulent_radiative_layer_2d():
    return {
        "PDE": "Turbulent Radiative Layer",
        "Dimension": 2,
        "Fields Scheme": "dpVV",
        "Fields": ["Density", "Pressure", "Velocity X", "Velocity Y"],
        "Domain Extent": [1, 3],
        "Resolution": [128, 384],
        "Time Steps": 101,
        "Dt": 1.597033,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["periodic", "periodic", "open", "open"],
        "Constants": ["Cooling Time"],
        "Cooling Time (range)": [0.03, 0.06, 0.1, 0.18, 0.32, 0.56, 1.00, 1.78, 3.16],
    }

def get_active_matter():
    return {
        "PDE": "Active Matter",
        "Dimension": 2,
        "Fields Scheme": "cVVOOOOSSSS",
        "Fields": ["Concentration", "Velocity X", "Velocity Y", "Orientation XX", "Orientation XY", "Orientation YX", "Orientation YY", "Strain XX", "Strain XY", "Strain YX", "Strain YY"],
        "Domain Extent": [10, 10],
        "Resolution": [256, 256],
        "Time Steps": 81,
        "Dt": 0.25,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
        "Constants": ["Domain Extent", "Particle Alignment Strength", "Active Dipol Strength"],
        "Domain Extent (range)": [10],
        "Particle Alignment Strength (range)": [1, 3, 5, 7, 9, 11, 13, 15, 17],
        "Active Dipol Strength (range)": [-1, -2, -3, -4, -5],
    }

def get_viscoelastic_instability():
    return {
        "PDE": "Viscoelastic Instability",
        "Dimension": 2,
        "Fields Scheme": "pzVVCCCC",
        "Fields": ["Pressure", "Conformation ZZ", "Velocity X", "Velocity Y", "Conformation XX", "Conformation XY", "Conformation YX", "Conformation YY"],
        "Domain Extent": [2*math.pi, 2],
        "Resolution": [512, 512],
        "Time Steps": 20,
        "Dt": 1.0,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["periodic", "periodic", "wall", "wall"],
        "Constants": ["Reynolds Number", "Weissenberg Number", "Viscosity Ratio", "Kolmogorov Length Scale", "Maximum Polymer Extensibility"],
    }

def get_helmholtz_staircase():
    return {
        "PDE": "Helmholtz",
        "Dimension": 2,
        "Fields Scheme": "rim",
        "Fields": ["Pressure (real)", "Pressure (imaginary)", "Mask"],
        "Domain Extent": [16, 4],
        "Resolution": [1024, 256],
        "Time Steps": 50,
        "Dt": 1.0,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["open", "open", "open", "wall"],
        "Constants": ["Frequency"],
        "Frequency (range)": [0.06283032, 0.25123038, 0.43929689, 0.62675846, 0.81330465, 0.99856671, 1.18207893, 1.36324313, 1.5412579, 1.71501267, 1.88295798, 2.04282969, 2.19133479, 2.32367294, 2.4331094, 2.5110908]
    }

def get_rayleigh_benard():
    return {
        "PDE": "Rayleigh Benard Convection",
        "Dimension": 2,
        "Fields Scheme": "bpVV",
        "Fields": ["Buoyancy", "Pressure", "Velocity X", "Velocity Y"],
        "Domain Extent": [4, 1],
        "Resolution": [512, 128],
        "Time Steps": 200,
        "Dt": 0.25,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["periodic", "periodic", "wall", "wall"],
        "Constants": ["Rayleigh Number", "Prandtl Number"],
        "Rayleigh Number (range)": [1e6, 1e7, 1e8, 1e9, 1e10],
        "Prandtl Number (range)": [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
    }

def get_shear_flow():
    return {
        "PDE": "Navier-Stokes: Shear Flow",
        "Dimension": 2,
        "Fields Scheme": "dpVV",
        "Fields": ["Density", "Pressure", "Velocity X", "Velocity Y"],
        "Domain Extent": [1, 2],
        "Resolution": [256, 512],
        "Time Steps": 200,
        "Dt": 0.1,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
        "Constants": ["Reynolds Number", "Schmidt Number"],
        "Reynolds Number (range)": [10000, 50000, 100000, 500000],
        "Schmidt Number (range)": [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
    }

def get_euler_multi_quadrants_periodicBC():
    return {
        "PDE": "Euler",
        "Dimension": 2,
        "Fields Scheme": "depVV",
        "Fields": ["Density", "Energy", "Pressure", "Velocity X", "Velocity Y"],
        "Domain Extent": [1, 1],
        "Resolution": [512, 512],
        "Time Steps": 100,
        "Dt": 0.015,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
        "Constants": ["Gas Constant"],
        "Gas Constant (range)": [1.13, 1.22, 1.3, 1.33, 1.365, 1.4, 1.404, 1.453, 1.597, 1.76],
    }

def get_euler_multi_quadrants_openBC():
    return {
        "PDE": "Euler",
        "Dimension": 2,
        "Fields Scheme": "depVV",
        "Fields": ["Density", "Energy", "Pressure", "Velocity X", "Velocity Y"],
        "Domain Extent": [1, 1],
        "Resolution": [512, 512],
        "Time Steps": 100,
        "Dt": 0.015,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["open", "open", "open", "open"],
        "Constants": ["Gas Constant"],
        "Gas Constant (range)": [1.13, 1.22, 1.3, 1.33, 1.365, 1.4, 1.404, 1.453, 1.597, 1.76],
    }