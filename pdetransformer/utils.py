import importlib
import os
from copy import copy
from typing import Sequence
from omegaconf import OmegaConf, DictConfig


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_pipeline(args: DictConfig, module):
    args = copy(args)

    if 'scheduler' in args:
        scheduler = instantiate_from_config(args['scheduler'])
        pipeline_args = {'scheduler': scheduler}
    else:
        pipeline_args = {}

    pipeline_args.update(module.get_pipeline_args())

    pipeline = {'target': args['target'],
                'params': pipeline_args}

    return instantiate_from_config(pipeline).to(module.device)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    if not "params" in config or config["params"] is None:
        return get_obj_from_str(config["target"])()
    else:
        return get_obj_from_str(config["target"])(**config.get("params", dict()))


def instantiate_from_config_(target, **params):
    return get_obj_from_str(target)(**params)


def parse_config(config):
    """
    Recursively parses a configuration dictionary and returns an OmegaConf object.
    If a configuration item is a file, it will be loaded and merged into the configuration.

    Args:
        config (dict): The configuration dictionary to parse.

    Returns:
        OmegaConf: The parsed configuration as an OmegaConf object.
    """
    conf_ = OmegaConf.create({})
    for key, value in config.items_ex(resolve=False):
        if (isinstance(value, dict) or isinstance(value, OmegaConf)
                or isinstance(value, DictConfig)):
            conf_[key] = parse_config(value)
        elif isinstance(value, str) and key == "_file":
            conf_ = OmegaConf.merge(conf_, parse_config(OmegaConf.load(value)))
        else:
            conf_[key] = value
    return conf_


import secrets
import string


# see wandb.sdk.lib.runid
def generate_id(length: int = 8) -> str:
    """Generate a random base-36 string of `length` digits."""
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def search_checkpoint(logdir):
    pass


def table_info(table: Sequence[Sequence], header: Sequence[str], return_hline=False) -> str:
    table.insert(0, header)
    col_widths = [max(len(str(item)) for item in col) for col in zip(*table)]
    h_line = "-".join("-" * width for width in col_widths)
    str_table = []
    for row in table:
        str_table.append(" | ".join(str(item).ljust(width) for item, width in zip(row, col_widths)))
    str_table.insert(1, h_line)
    str_table.insert(0, h_line)
    str_table.append(h_line)
    if return_hline:
        return os.linesep.join(str_table), h_line
    else:
        return os.linesep.join(str_table)
