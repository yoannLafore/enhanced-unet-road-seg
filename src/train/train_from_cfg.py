from src.model.unet import *
from src.preprocessing.dataset import *
import os
from src.utils import *

from src.train.loop.unet_loop import train_model_on_ds
from src.train.loop.refined_unet_loop import train_improve_model_on_ds


def train_from_cfg(cfg):
    """Train a model based on the provided configuration.
    Configuration specifications can be found in src/configs/sample/train_sample.yaml.

    Args:
        cfg (dict): Configuration dictionary containing all necessary settings.
    """

    random_state = cfg.random_state
    set_seed(random_state)

    # Create the dataset
    train_transform_cfg = cfg.train.transform
    test_transform_cfg = cfg.test.transform

    train_transform = build_from_cfg(train_transform_cfg, seed=random_state)
    test_transform = build_from_cfg(test_transform_cfg, seed=random_state)

    train_out_dir = cfg.train.out_dir
    train_run_name = cfg.train.run_name

    if train_run_name is not None and cfg.train.make_unique_name:
        train_run_name += "_" + get_time_str()
        train_out_dir = os.path.join(train_out_dir, train_run_name)

    # Make sure output directory exists
    os.makedirs(train_out_dir, exist_ok=True)
    # Update the out_dir in cfg
    cfg.train.out_dir = train_out_dir

    # Run the training
    train_ds, test_ds = load_train_test(
        cfg.data.train_dir,
        train_transform,
        test_transform,
        test_size=cfg.data.test_size,
        random_state=random_state,
    )

    train_type = cfg.train.get("type", "standard")

    if train_type == "improve":
        return train_improve_model_on_ds(train_ds, test_ds, cfg)
    else:
        return train_model_on_ds(train_ds, test_ds, cfg)
