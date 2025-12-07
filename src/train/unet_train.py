from src.model.unet import *
from src.preprocessing.dataset import *
import os
import torch
from tqdm import tqdm
import wandb
from src.utils import *
import json

from src.train.train_utils import *

# TODO :
# - Why are the validation metrics doing zigzag ?
# - Do we apply softmax and stuff,
# - Do a correct float tensor conversion in dataset
# - Try with other losses (Dice, Focal ... )
# - Finetune threshold
# - LR scheduler : Try CosineAnnealing with warm restarts
# TODO: VERY IMPORTANT : COMPUTE VALIDATION METRICS OVER THE FULL DATASET not per BATCH !!!
# TODO: For now we use cross entropy with loagits, but the mask values are 0 and 1, might be better to apply a sigmoid first and use BCE loss ?


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


def train_model_on_ds(train_ds, test_ds, cfg):
    print(f"Using out dir: {cfg.train.out_dir}")

    train_cfg = cfg.train
    device = cfg.device

    model_cfg = cfg.model
    model = build_from_cfg(model_cfg).to(device)

    collate_fn = build_from_cfg(cfg.train.collate_fn)
    batch_size = int(train_cfg.batch_size)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=4,
    )

    val_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=4,
    )

    # Make sure output directory exists
    out_dir = train_cfg.out_dir
    checkpoint_dir = os.path.join(out_dir, "checkpoints/")
    validation_file = os.path.join(out_dir, "validation.json")
    final_stats_file = os.path.join(out_dir, "final_stats.json")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config file
    dump_cfg(cfg, os.path.join(out_dir, "config.yaml"))

    log_wandb = bool(train_cfg.log_wandb)
    nb_epochs = int(train_cfg.epochs)
    validate_every = int(train_cfg.validate_every)
    checkpoint_every = int(train_cfg.checkpoint_every)

    optimizer = build_from_cfg(train_cfg.optimizer, params=model.parameters())
    lr_scheduler = build_from_cfg(train_cfg.lr_scheduler, optimizer=optimizer)
    criterion = build_from_cfg(train_cfg.criterion)

    if log_wandb:
        wandb.init(
            project="road-segmentation",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    progress = tqdm(range(nb_epochs), desc="Training Epochs")

    for epoch in progress:
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, log_wandb
        )
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(train_loss)
        else:
            lr_scheduler.step()
        progress.set_postfix({"train_loss": f"{train_loss:.4f}"})

        if log_wandb:
            wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})

        if (epoch + 1) % validate_every == 0:
            val_stats = evaluate_epoch(
                model, val_loader, criterion, device, log_wandb=log_wandb
            )
            val_stats["epoch"] = epoch + 1
            # Save validation stats
            append_val_json(validation_file, val_stats)

        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"unet_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)

    # Evaluate one last time on validation set
    final_stats = evaluate_epoch(
        model, val_loader, criterion, device, log_wandb=log_wandb
    )
    # Save final stats
    json.dump(final_stats, open(final_stats_file, "w"), indent=4)

    if log_wandb:
        wandb.finish()

    # Save final model
    final_model_path = os.path.join(out_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)

    print(f"Saved final model: {final_model_path}")

    return final_stats


def train_improve_model_on_ds(train_ds, test_ds, cfg):
    print(f"Using out dir: {cfg.train.out_dir}")

    train_cfg = cfg.train
    device = cfg.device

    base_model_cfg = cfg.base_model
    base_model = build_from_cfg(base_model_cfg).to(device)

    improved_model_cfg = cfg.improved_model
    improved_model = build_from_cfg(improved_model_cfg).to(device)

    collate_fn = build_from_cfg(cfg.train.collate_fn)
    batch_size = int(train_cfg.batch_size)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=4,
    )

    val_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=4,
    )

    train_jointly = bool(train_cfg.train_jointly)
    forward_features = bool(train_cfg.get("forward_features", False))

    # Make sure output directory exists
    out_dir = train_cfg.out_dir
    checkpoint_dir = os.path.join(out_dir, "checkpoints/")
    validation_file = os.path.join(out_dir, "validation.json")
    final_stats_file = os.path.join(out_dir, "final_stats.json")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config file
    dump_cfg(cfg, os.path.join(out_dir, "config.yaml"))

    log_wandb = bool(train_cfg.log_wandb)
    nb_epochs = int(train_cfg.epochs)
    validate_every = int(train_cfg.validate_every)
    checkpoint_every = int(train_cfg.checkpoint_every)

    if train_jointly:
        params = list(base_model.parameters()) + list(improved_model.parameters())
        optimizer = build_from_cfg(train_cfg.optimizer, params=params)
        criterion = build_from_cfg(train_cfg.criterion)
        lr_scheduler = build_from_cfg(train_cfg.lr_scheduler, optimizer=optimizer)
    else:
        base_optimizer = build_from_cfg(
            train_cfg.base_optimizer, params=base_model.parameters()
        )
        improved_optimizer = build_from_cfg(
            train_cfg.improved_optimizer, params=improved_model.parameters()
        )
        base_criterion = build_from_cfg(train_cfg.base_criterion)
        improved_criterion = build_from_cfg(train_cfg.improved_criterion)
        base_lr_scheduler = build_from_cfg(
            train_cfg.base_lr_scheduler, optimizer=base_optimizer
        )
        improved_lr_scheduler = build_from_cfg(
            train_cfg.improved_lr_scheduler, optimizer=improved_optimizer
        )

    if log_wandb:
        wandb.init(
            project="road-segmentation",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    progress = tqdm(range(nb_epochs), desc="Training Epochs")

    for epoch in progress:
        if train_jointly:
            train_loss = train_epoch_improve_module_joint(
                base_model,
                improved_model,
                optimizer,
                train_loader,
                criterion,
                device,
                log_wandb=log_wandb,
                forward_features=forward_features,
            )
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(train_loss)
            else:
                lr_scheduler.step()
            progress.set_postfix({"train_loss": f"{train_loss:.4f}"})
        else:
            train_loss_base, train_loss_improved = train_epoch_improve_module_sep(
                base_model,
                improved_model,
                base_optimizer,
                improved_optimizer,
                train_loader,
                base_criterion,
                improved_criterion,
                device,
                log_wandb=log_wandb,
                forward_features=forward_features,
            )
            if isinstance(
                base_lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                base_lr_scheduler.step(train_loss_base)
            else:
                base_lr_scheduler.step()
            if isinstance(
                improved_lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                improved_lr_scheduler.step(train_loss_improved)
            else:
                improved_lr_scheduler.step()
            progress.set_postfix(
                {
                    "train_loss": f"{train_loss_improved:.4f}",
                }
            )

        if log_wandb:
            if train_jointly:
                wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})
            else:
                wandb.log({"base_learning_rate": base_optimizer.param_groups[0]["lr"]})
                wandb.log(
                    {"improved_learning_rate": improved_optimizer.param_groups[0]["lr"]}
                )

        if (epoch + 1) % validate_every == 0:
            val_criterion = criterion if train_jointly else improved_criterion

            val_stats_base, val_stats_improved = evaluate_epoch_improve_module(
                base_model,
                improved_model,
                val_loader,
                val_criterion,
                device,
                log_wandb=log_wandb,
                forward_features=forward_features,
            )

            val_stats_improved["epoch"] = epoch + 1
            # Save validation stats
            append_val_json(validation_file, val_stats_improved)

        if (epoch + 1) % checkpoint_every == 0:
            base_checkpoint_path = os.path.join(
                checkpoint_dir, f"base_unet_epoch_{epoch+1}.pth"
            )
            improved_checkpoint_path = os.path.join(
                checkpoint_dir, f"improved_unet_epoch_{epoch+1}.pth"
            )
            torch.save(base_model.state_dict(), base_checkpoint_path)
            torch.save(improved_model.state_dict(), improved_checkpoint_path)

    # Evaluate one last time on validation set
    val_criterion = criterion if train_jointly else improved_criterion
    _, final_stats = evaluate_epoch_improve_module(
        base_model,
        improved_model,
        val_loader,
        val_criterion,
        device,
        log_wandb=log_wandb,
        forward_features=forward_features,
    )
    # Save final stats
    json.dump(final_stats, open(final_stats_file, "w"), indent=4)

    if log_wandb:
        wandb.finish()

    # Save final model
    base_final_model_path = os.path.join(out_dir, "base_final_model.pth")
    torch.save(base_model.state_dict(), base_final_model_path)

    improved_final_model_path = os.path.join(out_dir, "improved_final_model.pth")
    torch.save(improved_model.state_dict(), improved_final_model_path)

    print(f"Saved base final model: {base_final_model_path}")
    print(f"Saved improved final model: {improved_final_model_path}")

    return final_stats


def train_from_cfg(cfg):
    random_state = cfg.random_state

    # Create the dataset
    train_transform_cfg = cfg.train.transform
    test_transform_cfg = cfg.test.transform

    train_transform = build_from_cfg(train_transform_cfg)
    test_transform = build_from_cfg(test_transform_cfg)

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
