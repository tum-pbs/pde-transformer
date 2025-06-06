import argparse
import os
import sys
import datetime
from pathlib import Path

import warnings

from pl_bolts.utils.stability import UnderReviewWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UnderReviewWarning)

from omegaconf import OmegaConf

from src.setup import main_setup
from src.utils import parse_config, generate_id

from lightning.fabric.utilities.rank_zero import rank_zero_warn, rank_zero_info

import logging
log = logging.getLogger(__name__)

DEFAULT_SEED = 0
DEFAULT_LOG_DIR = './logs'
DEFAULT_ENVIRONMENT = './env/local.yaml'
DEFAULT_CFG_NAME = 'project_config.yaml'

def get_parser(**parser_kwargs):

    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default=None,
        nargs="?",
        help="Name of the experiment. Used for logging.",
    )

    parser.add_argument(
        "--dryrun",
        action='store_true',
        help="If true, don't log anything and don't create a logdir.",
    )

    parser.add_argument(
        "-c",
        "--config",
        nargs="*",
        metavar="config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default=None,
        help="directory for logging",
    )

    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="path to environment config",
    )

    parser.add_argument(
        "--debug",
        action='store_true',
        help="debug mode enabled",
    )

    parser.add_argument(
        "--no-test",
        action='store_true',
        help="no test after training",
    )

    parser.add_argument(
        "--no-train",
        action='store_true',
        help="no training",
    )

    parser.add_argument(
        "--ema",
        action='store_true',
        help="for testing use ema weights",
    )

    parser.add_argument(
        "--no-inference",
        action='store_true',
        help="no inference",
    )

    return parser

def restore_config(config_dir):
    config_file = config_dir.joinpath(DEFAULT_CFG_NAME)
    if config_file.exists():
        return OmegaConf.load(config_file)
    else:
        raise FileNotFoundError(f'Config file not found at {config_file}')

def save_config(config, config_dir):
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir.joinpath(DEFAULT_CFG_NAME)
    OmegaConf.save(config, config_file)


def get_config(opt, unknown):

    try:
        configs = [parse_config(OmegaConf.load(cfg)) for cfg in opt.config]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)

    except Exception as e:
        rank_zero_warn('Failed to load configs')
        raise e

    rank_zero_info(f'Configs loaded from {opt.config}')

    project = config["project"]
    rank_zero_info(f'Project: {project}')

    if opt.name is None:
        if 'name' in config:
            name = config['name']
        else:
            raise ValueError("No name provided in cli and config")
    else:
        name = opt.name

    rank_zero_info(f'Experiment name: {name}')

    if opt.seed is None:
        if 'seed' in config:
            seed = config['seed']
        else:
            seed = DEFAULT_SEED
    else:
        seed = opt.seed

    rank_zero_info(f'Random seed set to: {seed}')

    env = load_environment(opt.env)

    if opt.logdir is None:
        if 'logdir' in config:
            logdir = config['logdir']
        else:
            logdir = DEFAULT_LOG_DIR
    else:
        logdir = opt.logdir

    logdir = Path(logdir).joinpath(name)
    logdir.mkdir(parents=True, exist_ok=True)

    rank_zero_info(f'Logging experiment at: {logdir}')

    config.train = not opt.no_train
    rank_zero_info(f'Training enabled: {config.train}')

    config.inference = not opt.no_inference
    rank_zero_info(f'Inference enabled: {config.inference}')

    config.debug = opt.debug
    rank_zero_info(f'Debug mode: {opt.debug}')

    config.ema = opt.ema
    rank_zero_info(f'Using EMA weights for inference: {config.ema}')

    if config.debug:
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    config.no_test = opt.no_test
    rank_zero_info(f'No tests after training: {config.no_test}')

    checkpoint_dir = logdir.joinpath('checkpoint')
    checkpoint_dir.mkdir(exist_ok=True)

    config_dir = logdir.joinpath('config')
    config_dir.mkdir(exist_ok=True)

    # check if logdir exists
    if logdir.exists():
        rank_zero_info(f"Run directory {logdir} already exists. Trying to continue run...")

        try:
            config_restored = restore_config(config_dir)
            run_id = config_restored.runtime.id
            rank_zero_info(f"Continuing run with id {run_id}")

        except Exception as e:
            rank_zero_warn(f'Could not previous config at {config_dir}')
            rank_zero_warn(f'Reason: {e}')
            run_id = generate_id()
            rank_zero_warn(f'Creating new run with id {run_id}')

    else:
        logdir.mkdir(parents=True, exist_ok=True)
        run_id = generate_id()
        rank_zero_info(f'Creating new run with id {run_id}')

    if opt.dryrun:
        logger_state = "offline"
    else:
        logger_state = "online"

    rank_zero_info(f'Syncing with cloud: {logger_state}')

    runtime_config = OmegaConf.create({'runtime': {'seed': seed,
                                                   'project': project,
                                                   'logdir': logdir.absolute().as_posix(),
                                                   'name': name, 'id': run_id,
                                                   'logger_state': logger_state,
                                                   'checkpoint_dir': checkpoint_dir.absolute().as_posix(),
                                                   'config_dir': config_dir.absolute().as_posix(),
                                                   'debug': opt.debug,
                                                   'resume': False
                                                   }})

    config = OmegaConf.merge(config, runtime_config, env)

    OmegaConf.resolve(config)

    # after resolving, delete environment entries in config
    for k, v in env.items():
        config.pop(k)

    return config


def load_environment(env):

    if env is None:
        rank_zero_info(f'Loading default environment at {DEFAULT_ENVIRONMENT}')
        env_config = OmegaConf.load(DEFAULT_ENVIRONMENT)
    else:
        try:
            env_config = OmegaConf.load(env)
        except Exception as e:
            rank_zero_warn(f'Could not load environment {env}:')
            rank_zero_warn(e)
            rank_zero_warn(f'Loading environment {DEFAULT_ENVIRONMENT} instead')
            env_config = OmegaConf.load(DEFAULT_ENVIRONMENT)

    return env_config


if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # parse config
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    config = OmegaConf.create(get_config(opt, unknown))

    save_config(config, Path(config.runtime.config_dir))
    main_setup(config)






