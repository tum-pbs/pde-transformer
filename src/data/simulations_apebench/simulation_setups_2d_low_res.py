import exponax as ex
import numpy as np
import jax
import jax.numpy as jnp
import hashlib

def get_setup_2d(sim_type:str, is_test_set:bool, sim_id:int):
    if not is_test_set:
        seed_name = sim_type
    else:
        seed_name = sim_type + "_test"
    seed = int(hashlib.md5(seed_name.encode('utf-8')).hexdigest(), 16) % 2**30 + sim_id

    # linear
    if sim_type == "adv":
        return get_advection(is_test_set, seed)
    elif sim_type == "diff":
        return get_diffusion(is_test_set, seed)
    elif sim_type == "adv_diff":
        return get_advection_diffusion(is_test_set, seed)
    elif sim_type == "disp":
        return get_dispersion(is_test_set, seed)
    elif sim_type == "hyp":
        return get_hyper_diffusion(is_test_set, seed)

    # nonlinear
    elif sim_type == "burgers":
        return get_burgers(is_test_set, seed)
    elif sim_type == "kdv":
        return get_korteweg_de_vries(is_test_set, seed)
    elif sim_type == "ks":
        return get_kuramoto_sivashinsky(is_test_set, seed)

    # reaction-diffusion
    elif sim_type == "fisher":
        return get_fisher_kpp(is_test_set, seed)
    elif sim_type == "gs_alpha": # time-dependent
        return get_gray_scott(is_test_set, seed, variant="alpha")
    elif sim_type == "gs_beta": # time-dependent
        return get_gray_scott(is_test_set, seed, variant="beta")
    elif sim_type == "gs_gamma": # steady (with time-dependent defects)
        return get_gray_scott(is_test_set, seed, variant="gamma")
    elif sim_type == "gs_delta": # steady
        return get_gray_scott(is_test_set, seed, variant="delta")
    elif sim_type == "gs_epsilon": # chaotic
        return get_gray_scott(is_test_set, seed, variant="epsilon")
    elif sim_type == "gs_theta": # steady
        return get_gray_scott(is_test_set, seed, variant="theta")
    elif sim_type == "gs_iota": # steady
        return get_gray_scott(is_test_set, seed, variant="iota")
    elif sim_type == "gs_kappa": # steady-ish (very slow process)
        return get_gray_scott(is_test_set, seed, variant="kappa")
    elif sim_type == "sh":
        return get_swift_hohenberg(is_test_set, seed)

    # navier-stokes
    elif sim_type == "decay_turb":
        return get_decaying_turbulence(is_test_set, seed)
    elif sim_type == "kolm_flow":
        return get_kolmogorov_flow(is_test_set, seed)

    else:
        raise ValueError("Unknown simulation type: %s" % sim_type)


def initial_condition_generator(p:dict, varied:dict) -> ex.ic.BaseRandomICGenerator:
    init_type = np.random.choice(p["Initial Types"])
    varied["Initial Type"] = init_type
    if init_type == "Truncated Fourier":
        cutoff = np.random.randint(p["Initial Frequency Cutoff (min)"], p["Initial Frequency Cutoff (max)"])
        varied["Initial Frequency Cutoff"] = cutoff
        ic_gen = ex.ic.RandomTruncatedFourierSeries(2, cutoff=cutoff, max_one=True)

    elif init_type == "Gaussian Field":
        powerlaw = np.random.uniform(p["Initial Powerlaw Exponent (min)"], p["Initial Powerlaw Exponent (max)"])
        varied["Initial Powerlaw Exponent"] = powerlaw
        ic_gen = ex.ic.GaussianRandomField(2, powerlaw_exponent=powerlaw, max_one=True)

    elif init_type == "Diffused Noise":
        intensity = np.random.uniform(p["Initial Noise Intensity (min)"], p["Initial Noise Intensity (max)"])
        varied["Initial Noise Intensity"] = intensity
        ic_gen = ex.ic.DiffusedNoise(2, intensity=intensity, max_one=True)

    else:
        raise ValueError("Unknown initial condition type: %s" % init_type)

    return ic_gen



def get_advection(is_test_set:bool, seed:int) -> tuple[dict, dict, ex.BaseStepper, jnp.ndarray]:
    p = {
        "PDE": "Advection",
        "Dimension": 2,
        "Fields Scheme": "d",
        "Fields": ["Density"],
        "Domain Extent": 1.0,
        "Resolution": 256,
        "Time Steps": 30,
        "Dt": 0.01,
        "Sub Steps": 1,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
        "Constants": ["Velocity X", "Velocity Y"],
        "Initial Types": ["Truncated Fourier", "Gaussian Field", "Diffused Noise"],
        "Initial Frequency Cutoff (min)": 2,
        "Initial Frequency Cutoff (max)": 11,
        "Initial Powerlaw Exponent (min)": 2.3,
        "Initial Powerlaw Exponent (max)": 3.6,
        "Initial Noise Intensity (min)": 0.00005,
        "Initial Noise Intensity (max)": 0.01,
        "Velocity (min)": -5.0,
        "Velocity (max)": 5.0,
    }

    varied = {"Seed": seed}
    np.random.seed(seed)

    ic_gen = initial_condition_generator(p, varied)

    u_init = ic_gen(p["Resolution"], key=jax.random.PRNGKey(seed))

    vel = np.random.uniform(p["Velocity (min)"], p["Velocity (max)"], 2)
    stepper = ex.stepper.Advection(2, p["Domain Extent"], p["Resolution"], p["Dt"]/p["Sub Steps"], velocity=vel)

    varied["Velocity X"] = vel[0]
    varied["Velocity Y"] = vel[1]

    return p, varied, stepper, u_init


def get_diffusion(is_test_set:bool, seed:int) -> tuple[dict, dict, ex.BaseStepper, jnp.ndarray]:
    p = {
        "PDE": "Diffusion",
        "Dimension": 2,
        "Fields Scheme": "d",
        "Fields": ["Density"],
        "Domain Extent": 1.0,
        "Resolution": 256,
        "Time Steps": 30,
        "Dt": 0.01,
        "Sub Steps": 1,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
        "Constants": ["Viscosity X", "Viscosity Y"],
        "Initial Types": ["Truncated Fourier", "Gaussian Field", "Diffused Noise"],
        "Initial Frequency Cutoff (min)": 2,
        "Initial Frequency Cutoff (max)": 11,
        "Initial Powerlaw Exponent (min)": 2.3,
        "Initial Powerlaw Exponent (max)": 3.6,
        "Initial Noise Intensity (min)": 0.00005,
        "Initial Noise Intensity (max)": 0.01,
        "Viscosity (min)": 0.005,
        "Viscosity (max)": 0.05,
    }

    varied = {"Seed": seed}
    np.random.seed(seed)

    ic_gen = initial_condition_generator(p, varied)

    u_init = ic_gen(p["Resolution"], key=jax.random.PRNGKey(seed))

    nu = np.random.uniform(p["Viscosity (min)"], p["Viscosity (max)"], 2)
    stepper = ex.stepper.Diffusion(2, p["Domain Extent"], p["Resolution"], p["Dt"]/p["Sub Steps"], diffusivity=nu)

    varied["Viscosity X"] = nu[0]
    varied["Viscosity Y"] = nu[1]

    return p, varied, stepper, u_init


def get_advection_diffusion(is_test_set:bool, seed:int) -> tuple[dict, dict, ex.BaseStepper, jnp.ndarray]:
    p = {
        "PDE": "Advection-Diffusion",
        "Dimension": 2,
        "Fields Scheme": "d",
        "Fields": ["Density"],
        "Domain Extent": 1.0,
        "Resolution": 256,
        "Time Steps": 30,
        "Dt": 0.01,
        "Sub Steps": 1,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
        "Constants": ["Viscosity X", "Viscosity Y", "Velocity X", "Velocity Y"],
        "Initial Types": ["Truncated Fourier", "Gaussian Field", "Diffused Noise"],
        "Initial Frequency Cutoff (min)": 2,
        "Initial Frequency Cutoff (max)": 11,
        "Initial Powerlaw Exponent (min)": 2.3,
        "Initial Powerlaw Exponent (max)": 3.6,
        "Initial Noise Intensity (min)": 0.00005,
        "Initial Noise Intensity (max)": 0.01,
        "Velocity (min)": -5.0,
        "Velocity (max)": 5.0,
        "Viscosity (min)": 0.001,
        "Viscosity (max)": 0.03,
    }

    varied = {"Seed": seed}
    np.random.seed(seed)

    ic_gen = initial_condition_generator(p, varied)

    u_init = ic_gen(p["Resolution"], key=jax.random.PRNGKey(seed))

    nu = np.random.uniform(p["Viscosity (min)"], p["Viscosity (max)"], 2)
    vel = np.random.uniform(p["Velocity (min)"], p["Velocity (max)"], 2)
    stepper = ex.stepper.AdvectionDiffusion(2, p["Domain Extent"], p["Resolution"], p["Dt"]/p["Sub Steps"], velocity=vel, diffusivity=nu)

    varied["Viscosity X"] = nu[0]
    varied["Viscosity Y"] = nu[1]
    varied["Velocity X"] = vel[0]
    varied["Velocity Y"] = vel[1]

    return p, varied, stepper, u_init


def get_dispersion(is_test_set:bool, seed:int) -> tuple[dict, dict, ex.BaseStepper, jnp.ndarray]:
    p = {
        "PDE": "Dispersion",
        "Dimension": 2,
        "Fields Scheme": "d",
        "Fields": ["Density"],
        "Domain Extent": 1.0,
        "Resolution": 256,
        "Time Steps": 30,
        "Dt": 0.01,
        "Sub Steps": 1,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
        "Constants": ["Dispersivity X", "Dispersivity Y"],
        "Initial Types": ["Truncated Fourier", "Gaussian Field", "Diffused Noise"],
        "Initial Frequency Cutoff (min)": 2,
        "Initial Frequency Cutoff (max)": 11,
        "Initial Powerlaw Exponent (min)": 2.3,
        "Initial Powerlaw Exponent (max)": 3.6,
        "Initial Noise Intensity (min)": 0.00005,
        "Initial Noise Intensity (max)": 0.01,
        "Dispersivity (min)": 0.001,
        "Dispersivity (max)": 0.015,
    }

    varied = {"Seed": seed}
    np.random.seed(seed)

    ic_gen = initial_condition_generator(p, varied)

    u_init = ic_gen(p["Resolution"], key=jax.random.PRNGKey(seed))

    dispersivity = np.random.uniform(p["Dispersivity (min)"], p["Dispersivity (max)"], 2)
    stepper = ex.stepper.Dispersion(2, p["Domain Extent"], p["Resolution"], p["Dt"]/p["Sub Steps"], dispersivity=dispersivity)

    varied["Dispersivity X"] = dispersivity[0]
    varied["Dispersivity Y"] = dispersivity[1]

    return p, varied, stepper, u_init


def get_hyper_diffusion(is_test_set:bool, seed:int) -> tuple[dict, dict, ex.BaseStepper, jnp.ndarray]:
    p = {
        "PDE": "Hyper-Diffusion",
        "Dimension": 2,
        "Fields Scheme": "d",
        "Fields": ["Density"],
        "Domain Extent": 1.0,
        "Resolution": 256,
        "Time Steps": 30,
        "Dt": 0.01,
        "Sub Steps": 1,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
        "Constants": ["Hyper-Diffusivity"],
        "Initial Types": ["Truncated Fourier", "Gaussian Field", "Diffused Noise"],
        "Initial Frequency Cutoff (min)": 2,
        "Initial Frequency Cutoff (max)": 11,
        "Initial Powerlaw Exponent (min)": 2.3,
        "Initial Powerlaw Exponent (max)": 3.6,
        "Initial Noise Intensity (min)": 0.00005,
        "Initial Noise Intensity (max)": 0.01,
        "Hyper-Diffusivity (min)": 0.00005,
        "Hyper-Diffusivity (max)": 0.0005,
    }

    varied = {"Seed": seed}
    np.random.seed(seed)

    ic_gen = initial_condition_generator(p, varied)

    u_init = ic_gen(p["Resolution"], key=jax.random.PRNGKey(seed))

    diffusivity = np.random.uniform(p["Hyper-Diffusivity (min)"], p["Hyper-Diffusivity (max)"])
    stepper = ex.stepper.HyperDiffusion(2, p["Domain Extent"], p["Resolution"], p["Dt"]/p["Sub Steps"], hyper_diffusivity=diffusivity)

    varied["Hyper-Diffusivity"] = diffusivity

    return p, varied, stepper, u_init


def get_burgers(is_test_set:bool, seed:int) -> tuple[dict, dict, ex.BaseStepper, jnp.ndarray]:
    p = {
        "PDE": "Burgers",
        "Dimension": 2,
        "Fields Scheme": "VV",
        "Fields": ["Velocity X", "Velocity Y"],
        "Domain Extent": 1.0,
        "Resolution": 256,
        "Time Steps": 30,
        "Dt": 0.01,
        "Sub Steps": 50,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
        "Constants": ["Viscosity"],
        "Initial Types": ["Truncated Fourier", "Gaussian Field", "Diffused Noise"],
        "Initial Frequency Cutoff (min)": 2,
        "Initial Frequency Cutoff (max)": 11,
        "Initial Powerlaw Exponent (min)": 2.3,
        "Initial Powerlaw Exponent (max)": 3.6,
        "Initial Noise Intensity (min)": 0.00005,
        "Initial Noise Intensity (max)": 0.01,
        "Viscosity (min)": 0.00005,
        "Viscosity (max)": 0.0003,
    }

    varied = {"Seed": seed}
    np.random.seed(seed)

    ic_gen = initial_condition_generator(p, varied)

    multi_channel_ic_gen = ex.ic.RandomMultiChannelICGenerator([ic_gen, ic_gen])
    u_init = multi_channel_ic_gen(p["Resolution"], key=jax.random.PRNGKey(seed))

    nu = np.random.uniform(p["Viscosity (min)"], p["Viscosity (max)"])
    stepper = ex.stepper.Burgers(2, p["Domain Extent"], p["Resolution"], p["Dt"]/p["Sub Steps"], diffusivity=nu)

    varied["Viscosity"] = nu

    return p, varied, stepper, u_init


def get_korteweg_de_vries(is_test_set:bool, seed:int) -> tuple[dict, dict, ex.BaseStepper, jnp.ndarray]:
    p = {
        "PDE": "Korteweg-de-Vries",
        "Dimension": 2,
        "Fields Scheme": "VV",
        "Fields": ["Velocity X", "Velocity Y"],
        "Resolution": 256,
        "Time Steps": 30,
        "Dt": 0.05,
        "Sub Steps": 10,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
        "Constants": ["Domain Extent", "Viscosity"],
        "Initial Types": ["Truncated Fourier", "Gaussian Field", "Diffused Noise"],
        "Initial Frequency Cutoff (min)": 2,
        "Initial Frequency Cutoff (max)": 11,
        "Initial Powerlaw Exponent (min)": 2.3,
        "Initial Powerlaw Exponent (max)": 3.6,
        "Initial Noise Intensity (min)": 0.00005,
        "Initial Noise Intensity (max)": 0.01,
        "Domain Extent (min)": 30.0,
        "Domain Extent (max)": 120.0,
        "Viscosity (min)": 0.00005,
        "Viscosity (max)": 0.001,
    }

    varied = {"Seed": seed}
    np.random.seed(seed)

    ic_gen = initial_condition_generator(p, varied)

    multi_channel_ic_gen = ex.ic.RandomMultiChannelICGenerator([ic_gen, ic_gen])
    u_init = multi_channel_ic_gen(p["Resolution"], key=jax.random.PRNGKey(seed))

    extent = np.random.uniform(p["Domain Extent (min)"], p["Domain Extent (max)"])
    nu = np.random.uniform(p["Viscosity (min)"], p["Viscosity (max)"])
    #scale = np.random.uniform(p["Convection Scale (min)"], p["Convection Scale (max)"])
    #dispersivity = np.random.uniform(p["Dispersivity (min)"], p["Dispersivity (max)"])
    stepper = ex.stepper.KortewegDeVries(2, extent, p["Resolution"], p["Dt"]/p["Sub Steps"], diffusivity=nu)

    varied["Domain Extent"] = extent
    varied["Viscosity"] = nu
    #varied["Convection Scale"] = scale
    #varied["Dispersivity"] = dispersivity

    return p, varied, stepper, u_init


def get_kuramoto_sivashinsky(is_test_set:bool, seed:int) -> tuple[dict, dict, ex.BaseStepper, jnp.ndarray]:
    p = {
        "PDE": "Kuramoto-Sivashinsky",
        "Dimension": 2,
        "Fields Scheme": "d",
        "Fields": ["Density"],
        "Resolution": 256,
        "Time Steps": 30 if not is_test_set else 200,
        "Warmup Steps": 200,
        "Dt": 0.5,
        "Sub Steps": 5,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
        "Constants": ["Domain Extent"],
        "Initial Types": ["Truncated Fourier", "Gaussian Field", "Diffused Noise"],
        "Initial Frequency Cutoff (min)": 2,
        "Initial Frequency Cutoff (max)": 11,
        "Initial Powerlaw Exponent (min)": 2.3,
        "Initial Powerlaw Exponent (max)": 3.6,
        "Initial Noise Intensity (min)": 0.00005,
        "Initial Noise Intensity (max)": 0.01,
        "Domain Extent (min)": 10,
        "Domain Extent (max)": 130,
    }

    varied = {"Seed": seed}
    np.random.seed(seed)

    ic_gen = initial_condition_generator(p, varied)

    u_init = ic_gen(p["Resolution"], key=jax.random.PRNGKey(seed))

    extent = np.random.uniform(p["Domain Extent (min)"], p["Domain Extent (max)"])
    stepper = ex.stepper.KuramotoSivashinsky(2, extent, p["Resolution"], p["Dt"]/p["Sub Steps"])

    varied["Domain Extent"] = extent

    return p, varied, stepper, u_init


def get_fisher_kpp(is_test_set:bool, seed:int) -> tuple[dict, dict, ex.BaseStepper, jnp.ndarray]:
    p = {
        "PDE": "Fisher-KPP",
        "Dimension": 2,
        "Fields Scheme": "c",
        "Fields": ["Concentration"],
        "Domain Extent": 1.0,
        "Resolution": 256,
        "Time Steps": 30,
        "Dt": 0.005,
        "Sub Steps": 1,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
        "Constants": ["Diffusivity", "Reactivity"],
        "Initial Types": ["Truncated Fourier", "Gaussian Field", "Diffused Noise"],
        "Initial Frequency Cutoff (min)": 2,
        "Initial Frequency Cutoff (max)": 11,
        "Initial Powerlaw Exponent (min)": 2.3,
        "Initial Powerlaw Exponent (max)": 3.6,
        "Initial Noise Intensity (min)": 0.00005,
        "Initial Noise Intensity (max)": 0.01,
        "Diffusivity (min)": 0.0001,
        "Diffusivity (max)": 0.02,
        "Reactivity (min)": 5.0,
        "Reactivity (max)": 15.0,
    }

    varied = {"Seed": seed}
    np.random.seed(seed)

    #ic_gen = ex.ic.ClampingICGenerator(ex.ic.RandomTruncatedFourierSeries(3), limits=(0, 1))
    ic_gen = ex.ic.ClampingICGenerator(initial_condition_generator(p, varied), limits=(0, 1))

    u_init = ic_gen(p["Resolution"], key=jax.random.PRNGKey(seed))

    nu = np.random.uniform(p["Diffusivity (min)"], p["Diffusivity (max)"])
    r = np.random.uniform(p["Reactivity (min)"], p["Reactivity (max)"])
    stepper = ex.stepper.reaction.FisherKPP(2, p["Domain Extent"], p["Resolution"], p["Dt"]/p["Sub Steps"], diffusivity=nu, reactivity=r)

    varied["Diffusivity"] = nu
    varied["Reactivity"] = r

    return p, varied, stepper, u_init


def get_gray_scott(is_test_set:bool, seed:int, variant:str) -> tuple[dict, dict, ex.BaseStepper, jnp.ndarray]:
    if variant == "alpha": # time-dependent
        p = {
            "PDE": "Gray Scott",
            "Dimension": 2,
            "Fields Scheme": "cb",
            "Fields": ["Concentration A", "Concentration B"],
            "Domain Extent": 2.5,
            "Resolution": 256,
            "Time Steps": 30 if not is_test_set else 100,
            "Warmup Steps": 75,
            "Dt": 30.0,
            "Sub Steps": 30,
            "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
            "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
            "Constants": ["Feed Rate", "Kill Rate"],
            "Initial Types": ["Gaussian Blobs"],
            "Diffusivity A": 0.00002,
            "Diffusivity B": 0.00001,
            "Dynamics Types (Feed Rate, Kill Rate)": [
                (0.008, 0.046),
            ],
        }
    elif variant == "beta": # time-dependent
        p = {
            "PDE": "Gray Scott",
            "Dimension": 2,
            "Fields Scheme": "cb",
            "Fields": ["Concentration A", "Concentration B"],
            "Domain Extent": 2.5,
            "Resolution": 256,
            "Time Steps": 30 if not is_test_set else 100,
            "Warmup Steps": 50,
            "Dt": 30.0,
            "Sub Steps": 30,
            "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
            "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
            "Constants": ["Feed Rate", "Kill Rate"],
            "Initial Types": ["Gaussian Blobs"],
            "Diffusivity A": 0.00002,
            "Diffusivity B": 0.00001,
            "Dynamics Types (Feed Rate, Kill Rate)": [
                (0.020, 0.046),
            ],
        }
    elif variant == "gamma": # steady (with time-dependent defects)
        p = {
            "PDE": "Gray Scott",
            "Dimension": 2,
            "Fields Scheme": "cb",
            "Fields": ["Concentration A", "Concentration B"],
            "Domain Extent": 2.5,
            "Resolution": 256,
            "Time Steps": 30 if not is_test_set else 100,
            "Warmup Steps": 70,
            "Dt": 75.0,
            "Sub Steps": 75,
            "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
            "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
            "Constants": ["Feed Rate", "Kill Rate"],
            "Initial Types": ["Gaussian Blobs"],
            "Diffusivity A": 0.00002,
            "Diffusivity B": 0.00001,
            "Dynamics Types (Feed Rate, Kill Rate)": [
                (0.024, 0.056),
            ],
        }
    elif variant == "delta": # steady
        p = {
            "PDE": "Gray Scott",
            "Dimension": 2,
            "Fields Scheme": "cb",
            "Fields": ["Concentration A", "Concentration B"],
            "Domain Extent": 2.5,
            "Resolution": 256,
            "Time Steps": 30,
            #"Warmup Steps": 10,
            "Dt": 130.0,
            "Sub Steps": 130,
            "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
            "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
            "Constants": ["Feed Rate", "Kill Rate"],
            "Initial Types": ["Gaussian Blobs"],
            "Diffusivity A": 0.00002,
            "Diffusivity B": 0.00001,
            "Dynamics Types (Feed Rate, Kill Rate)": [
                (0.028, 0.056),
            ],
        }
    elif variant == "epsilon": # chaotic
        p = {
            "PDE": "Gray Scott",
            "Dimension": 2,
            "Fields Scheme": "cb",
            "Fields": ["Concentration A", "Concentration B"],
            "Domain Extent": 2.5,
            "Resolution": 256,
            "Time Steps": 30 if not is_test_set else 100,
            "Warmup Steps": 300,
            "Dt": 15.0,
            "Sub Steps": 15,
            "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
            "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
            "Constants": ["Feed Rate", "Kill Rate"],
            "Initial Types": ["Gaussian Blobs"],
            "Diffusivity A": 0.00002,
            "Diffusivity B": 0.00001,
            "Dynamics Types (Feed Rate, Kill Rate)": [
                (0.020, 0.056),
            ],
        }
    elif variant == "theta": # steady
        p = {
            "PDE": "Gray Scott",
            "Dimension": 2,
            "Fields Scheme": "cb",
            "Fields": ["Concentration A", "Concentration B"],
            "Domain Extent": 2.5,
            "Resolution": 256,
            "Time Steps": 30,
            #"Warmup Steps": 10,
            "Dt": 200.0,
            "Sub Steps": 200,
            "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
            "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
            "Constants": ["Feed Rate", "Kill Rate"],
            "Initial Types": ["Gaussian Blobs"],
            "Diffusivity A": 0.00002,
            "Diffusivity B": 0.00001,
            "Dynamics Types (Feed Rate, Kill Rate)": [
                (0.040, 0.060),
            ],
        }
    elif variant == "iota": # steady
        p = {
            "PDE": "Gray Scott",
            "Dimension": 2,
            "Fields Scheme": "cb",
            "Fields": ["Concentration A", "Concentration B"],
            "Domain Extent": 2.5,
            "Resolution": 256,
            "Time Steps": 30,
            #"Warmup Steps": 10,
            "Dt": 240.0,
            "Sub Steps": 240,
            "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
            "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
            "Constants": ["Feed Rate", "Kill Rate"],
            "Initial Types": ["Gaussian Blobs"],
            "Diffusivity A": 0.00002,
            "Diffusivity B": 0.00001,
            "Dynamics Types (Feed Rate, Kill Rate)": [
                (0.050, 0.0605),
            ],
        }
    elif variant == "kappa": # steady-ish (very slow process)
        p = {
            "PDE": "Gray Scott",
            "Dimension": 2,
            "Fields Scheme": "cb",
            "Fields": ["Concentration A", "Concentration B"],
            "Domain Extent": 2.5,
            "Resolution": 256,
            "Time Steps": 30,
            "Warmup Steps": 15,
            "Dt": 300.0,
            "Sub Steps": 300,
            "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
            "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
            "Constants": ["Feed Rate", "Kill Rate"],
            "Initial Types": ["Gaussian Blobs"],
            "Diffusivity A": 0.00002,
            "Diffusivity B": 0.00001,
            "Dynamics Types (Feed Rate, Kill Rate)": [
                (0.052, 0.063),
            ],
        }
    else:
        raise ValueError(f"Unknown variant '{variant}'")

    varied = {"Seed": seed}
    np.random.seed(seed)

    ic_gen = ex.ic.RandomMultiChannelICGenerator([
        ex.ic.RandomGaussianBlobs(2, domain_extent=p["Domain Extent"],
                                  position_range=(0.4,0.6) if variant in ["kappa"] else (0.2,0.8), num_blobs=4, one_complement=True),
        ex.ic.RandomGaussianBlobs(2, domain_extent=p["Domain Extent"],
                                  position_range=(0.4,0.6) if variant in ["kappa"] else (0.2,0.8), num_blobs=4),
    ])

    u_init = ic_gen(p["Resolution"], key=jax.random.PRNGKey(seed))

    dyn = np.random.randint(len(p["Dynamics Types (Feed Rate, Kill Rate)"]))
    feed, kill = p["Dynamics Types (Feed Rate, Kill Rate)"][dyn]
    stepper = ex.stepper.reaction.GrayScott(2, p["Domain Extent"], p["Resolution"], p["Dt"]/p["Sub Steps"],
                                    diffusivity_1=p["Diffusivity A"], diffusivity_2=p["Diffusivity B"], feed_rate=feed, kill_rate=kill)

    #varied["Dynamics Type"] = dynamic
    varied["Feed Rate"] = feed
    varied["Kill Rate"] = kill

    return p, varied, stepper, u_init


def get_swift_hohenberg(is_test_set:bool, seed:int) -> tuple[dict, dict, ex.BaseStepper, jnp.ndarray]:
    p = {
        "PDE": "Swift-Hohenberg",
        "Dimension": 2,
        "Fields Scheme": "c",
        "Fields": ["Concentration"],
        "Domain Extent": 20.0 * jnp.pi,
        "Resolution": 256,
        "Time Steps": 30,
        "Dt": 0.5,
        "Sub Steps": 5,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
        "Constants": ["Reactivity", "Critical Number"],
        "Initial Types": ["Truncated Fourier", "Gaussian Field", "Diffused Noise"],
        "Initial Frequency Cutoff (min)": 2,
        "Initial Frequency Cutoff (max)": 11,
        "Initial Powerlaw Exponent (min)": 2.3,
        "Initial Powerlaw Exponent (max)": 3.6,
        "Initial Noise Intensity (min)": 0.00005,
        "Initial Noise Intensity (max)": 0.01,
        "Reactivity (min)": 0.4,
        "Reactivity (max)": 1.0,
        "Critical Number (min)": 0.8,
        "Critical Number (max)": 1.2,
    }

    varied = {"Seed": seed}
    np.random.seed(seed)

    #ic_gen = ex.ic.RandomTruncatedFourierSeries(3, max_one=True)
    ic_gen = initial_condition_generator(p, varied)

    u_init = ic_gen(p["Resolution"], key=jax.random.PRNGKey(seed))

    r = np.random.uniform(p["Reactivity (min)"], p["Reactivity (max)"])
    k = np.random.uniform(p["Critical Number (min)"], p["Critical Number (max)"])
    stepper = ex.stepper.reaction.SwiftHohenberg(2, p["Domain Extent"], p["Resolution"], p["Dt"]/p["Sub Steps"], reactivity=r, critical_number=k)

    varied["Reactivity"] = r
    varied["Critical Number"] = k

    return p, varied, stepper, u_init


def get_decaying_turbulence(is_test_set:bool, seed:int) -> tuple[dict, dict, ex.BaseStepper, jnp.ndarray]:
    p = {
        "PDE": "Navier-Stokes: Decaying Turbulence",
        "Dimension": 2,
        "Fields Scheme": "v",
        "Fields": ["Vorticity"],
        "Domain Extent": 1.0,
        "Resolution": 256,
        "Time Steps": 30 if not is_test_set else 200,
        "Dt": 3.0,
        "Sub Steps": 500,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
        "Constants": ["Viscosity"],
        "Initial Types": ["Truncated Fourier", "Gaussian Field", "Diffused Noise"],
        "Initial Frequency Cutoff (min)": 2,
        "Initial Frequency Cutoff (max)": 11,
        "Initial Powerlaw Exponent (min)": 2.3,
        "Initial Powerlaw Exponent (max)": 3.6,
        "Initial Noise Intensity (min)": 0.00005,
        "Initial Noise Intensity (max)": 0.01,
        "Viscosity (min)": 0.00001,
        "Viscosity (max)": 0.0001,
    }

    varied = {"Seed": seed}
    np.random.seed(seed)

    ic_gen = initial_condition_generator(p, varied)

    u_init = ic_gen(p["Resolution"], key=jax.random.PRNGKey(seed))

    nu = np.random.uniform(p["Viscosity (min)"], p["Viscosity (max)"])
    stepper = ex.stepper.NavierStokesVorticity(2, p["Domain Extent"], p["Resolution"], p["Dt"]/p["Sub Steps"], diffusivity=nu)

    varied["Viscosity"] = nu

    return p, varied, stepper, u_init


def get_kolmogorov_flow(is_test_set:bool, seed:int) -> tuple[dict, dict, ex.BaseStepper, jnp.ndarray]:
    p = {
        "PDE": "Navier-Stokes: Kolmogorov Flow",
        "Dimension": 2,
        "Fields Scheme": "v",
        "Fields": ["Vorticity"],
        "Domain Extent": 2 * jnp.pi,
        "Resolution": 256,
        "Time Steps": 30 if not is_test_set else 200,
        "Warmup Steps": 50,
        "Dt": 0.3,
        "Sub Steps": 1500,
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive"],
        "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic"],
        "Constants": ["Viscosity"],
        "Initial Types": ["Truncated Fourier", "Gaussian Field", "Diffused Noise"],
        "Initial Frequency Cutoff (min)": 2,
        "Initial Frequency Cutoff (max)": 11,
        "Initial Powerlaw Exponent (min)": 2.3,
        "Initial Powerlaw Exponent (max)": 3.6,
        "Initial Noise Intensity (min)": 0.00005,
        "Initial Noise Intensity (max)": 0.01,
        "Viscosity (min)": 0.0001,
        "Viscosity (max)": 0.001,
    }

    varied = {"Seed": seed}
    np.random.seed(seed)

    ic_gen = initial_condition_generator(p, varied)

    u_init = ic_gen(p["Resolution"], key=jax.random.PRNGKey(seed))

    nu = np.random.uniform(p["Viscosity (min)"], p["Viscosity (max)"])
    stepper = ex.stepper.KolmogorovFlowVorticity(2, p["Domain Extent"], p["Resolution"], p["Dt"]/p["Sub Steps"], diffusivity=nu)

    varied["Viscosity"] = nu

    return p, varied, stepper, u_init