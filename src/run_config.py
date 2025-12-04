import argparse
from src.utils import *
from src.train.kfolds import perform_kfolds
from src.train.unet_train import train_from_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run configuration for training or evaluation."
    )

    parser.add_argument(
        "-c",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Config file: {args.c}")
    cfg = load_cfg(args.c)
    print("Configuration loaded.")

    task_type = cfg.task_type.lower()

    if task_type == "train" or task_type == "improve_model":
        # Perform training
        train_from_cfg(cfg)
    elif task_type == "kfold":
        # Perform k-fold cross-validation
        perform_kfolds(cfg)
    else:
        raise ValueError(f"Unknown task type '{task_type}'")


if __name__ == "__main__":
    main()
