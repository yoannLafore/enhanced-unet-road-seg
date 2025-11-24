import json
from src.model.unet import *
from src.preprocessing.dataset import *
from src.utils import *
import os
from src.train.unet_train import train_model_on_ds
import pandas as pd


def perform_kfolds(cfg: dict):
    random_state = cfg.random_state

    # Get the k-folds
    k = int(cfg.kfold.k)
    train_transform_cfg = cfg.train.transform
    test_transform_cfg = cfg.test.transform

    train_transform = build_from_cfg(train_transform_cfg)
    test_transform = build_from_cfg(test_transform_cfg)

    kfold = load_k_fold_datasets(
        cfg.data.train_dir, train_transform, test_transform, k, random_state
    )

    # Create the output directory for k-folds
    kfold_dir = cfg.kfold.out_dir
    run_name = cfg.kfold.run_name

    if cfg.kfold.make_unique_name:
        run_name += "_" + get_time_str()

    out_dir = os.path.join(kfold_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Save the config file
    dump_cfg(cfg, os.path.join(out_dir, "config.yaml"))

    fold_run_names = ["fold_" + str(i + 1) for i in range(k)]

    for i, (train_fold, test_fold) in enumerate(kfold):
        print(f"Training fold {i+1}/{k}...")

        cfg.train.run_name = fold_run_names[i]
        cfg.train.out_dir = os.path.join(out_dir, fold_run_names[i])

        train_model_on_ds(train_fold, test_fold, cfg)

    # Compute final statistics across folds
    raw_final_stats = []

    for i in range(k):
        stats_file = os.path.join(out_dir, fold_run_names[i], "final_stats.json")
        stats = json.load(open(stats_file))
        stats["fold"] = i + 1
        raw_final_stats.append(stats)

    # Create pandas DataFrame
    df = pd.DataFrame(raw_final_stats)

    # For each, compute mean and std
    summary = {}
    for column in df.columns:
        # Skip fold column
        if column == "fold":
            continue

        mean = df[column].mean()
        std = df[column].std()
        summary[column + "_mean"] = mean
        summary[column + "_std"] = std

    # Save summary to JSON
    summary_file = os.path.join(out_dir, "kfolds_final_stats.json")
    json.dump(summary, open(summary_file, "w"), indent=4)
    print(f"K-Folds final statistics saved to {summary_file}")

    # Also save raw stats
    raw_stats_file = os.path.join(out_dir, "kfolds_raw_stats.json")
    json.dump(raw_final_stats, open(raw_stats_file, "w"), indent=4)
    print(f"K-Folds raw statistics saved to {raw_stats_file}")
