from src.model.unet import *
from src.preprocessing.dataset import *
import os
import torch
from tqdm import tqdm
import wandb
from src.utils import *
import json

from src.train.epoch.train_epoch import *
from src.train.epoch.eval_epoch import *


def append_val_json(file_path: str, data: dict):
    """Append validation data to a JSON file.

    Args:
        file_path (str): Path to the JSON file.
        data (dict): Data to append.
    """
    if os.path.exists(file_path):
        existing_data = json.load(open(file_path))
    else:
        existing_data = []

    existing_data.append(data)
    json.dump(existing_data, open(file_path, "w"), indent=4)


def get_dataloaders(
    train_ds,
    val_ds,
    cfg,
):
    collate_fn = build_from_cfg(cfg.train.collate_fn)
    batch_size = int(cfg.train.batch_size)

    generator = torch.Generator()
    generator.manual_seed(cfg.random_state)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=4,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        generator=generator,
        pin_memory=True,
        num_workers=4,
    )

    return train_loader, val_loader


def make_out_files_dirs(out_dir: str):
    checkpoint_dir = os.path.join(out_dir, "checkpoints/")
    validation_file = os.path.join(out_dir, "validation.json")
    final_stats_file = os.path.join(out_dir, "final_stats.json")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir, validation_file, final_stats_file


def save_config_file(out_dir, cfg):
    dump_cfg(cfg, os.path.join(out_dir, "config.yaml"))


def load_pretrained_if_needed(model, train_cfg, device):
    # Use pretrained weights if specified
    pretrained_weights = train_cfg.get("pretrained_weights", None)
    if pretrained_weights is not None:
        print(f"Loading pretrained weights from: {pretrained_weights}")
        model.load_state_dict(torch.load(pretrained_weights, map_location=device))
