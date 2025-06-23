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

from pdetransformer.setup import main_setup, save_config, get_config

import logging

log = logging.getLogger(__name__)


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
        help="no inference after training",
    )

    return parser


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # parse config
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    config = OmegaConf.create(get_config(opt, unknown))

    save_config(config, Path(config.runtime.config_dir))
    main_setup(config)
