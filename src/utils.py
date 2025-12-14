import importlib
from omegaconf import OmegaConf
import random
import numpy as np
import torch

# Imported from HW2 of the Foundations models and generative AI course


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    """
    print(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


def load_callable_from_path(path: str):
    """
    Given 'module.submodule.ClassOrFunc', return the actual object.
    """
    if "." not in path:
        raise ValueError(f"Invalid path '{path}', expected module.object")

    module_path, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_path)

    try:
        return getattr(module, attr)
    except AttributeError:
        raise ImportError(f"Cannot find '{attr}' in module '{module_path}'")


def build_callable(class_or_func_path: str, args_dict: dict, **extra_args):
    """
    Given a path to a class or function and a dict of arguments,
    return a callable object (class or function) initialized with the given arguments.
    """
    func_or_class = load_callable_from_path(class_or_func_path)

    if not callable(func_or_class):
        raise ValueError(f"'{class_or_func_path}' is not callable")
    return func_or_class(**args_dict, **extra_args)


def build_from_cfg(cfg: dict, **extra_args):
    """
    Given a config dict with 'class_path' and 'args', build the corresponding object.
    """
    if "class_path" not in cfg or "args" not in cfg:
        raise ValueError("Config dict must have 'class_path' and 'args' keys")

    return build_callable(cfg["class_path"], cfg["args"], **extra_args)


def get_time_str():
    """
    Return current time as a string formatted as YYYYMMDD_HHMMSS.
    """
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_cfg(cfg_path: str, base_cfg_path="src/configs/base.yaml"):
    """
    Load a configuration from a YAML file, optionally merging with a base config.
    """
    base_cfg = OmegaConf.load(base_cfg_path)
    cfg = OmegaConf.load(cfg_path)
    merged_cfg = OmegaConf.merge(base_cfg, cfg)
    return merged_cfg


def dump_cfg(cfg: dict, out_path: str):
    """
    Dump a configuration dict to a YAML file.
    """
    with open(out_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
