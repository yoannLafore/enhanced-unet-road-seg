from src.train.loop.loop_utils import *


def train_model_on_ds(train_ds, test_ds, cfg):
    """Train a (Resnet)UNet model on the given datasets.

    Args:
        train_ds (Dataset): Training dataset.
        test_ds (Dataset): Testing/validation dataset.
        cfg (DictConfig): Configuration for the run.

    Returns:
        dict: Final statistics after training.
    """

    print(f"Using out dir: {cfg.train.out_dir}")

    train_cfg = cfg.train
    device = cfg.device

    model_cfg = cfg.model
    model = build_from_cfg(model_cfg).to(device)

    # Use pretrained weights if specified
    load_pretrained_if_needed(model, train_cfg, device)

    # Load data loaders
    train_loader, val_loader = get_dataloaders(train_ds, test_ds, cfg)

    # Make sure output directory exists
    out_dir = train_cfg.out_dir
    checkpoint_dir, validation_file, final_stats_file = make_out_files_dirs(out_dir)

    # Save config file
    save_config_file(out_dir, cfg)

    # Get hyperparameters
    log_wandb = bool(train_cfg.log_wandb)
    nb_epochs = int(train_cfg.epochs)
    validate_every = int(train_cfg.validate_every)
    checkpoint_every = int(train_cfg.checkpoint_every)

    # Initialize optimizer, lr_scheduler, criterion
    optimizer = build_from_cfg(train_cfg.optimizer, params=model.parameters())
    lr_scheduler = build_from_cfg(train_cfg.lr_scheduler, optimizer=optimizer)
    criterion = build_from_cfg(train_cfg.criterion)

    # Initialize wandb if needed
    if log_wandb:
        wandb.init(
            project="road-segmentation",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    progress = tqdm(range(nb_epochs), desc="Training Epochs")

    # Save best model based on validation F1 score
    save_best_f1_model = bool(train_cfg.get("save_best_f1_model", False))
    best_val_f1 = 0.0

    for epoch in progress:
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, log_wandb
        )
        lr_scheduler.step()
        progress.set_postfix({"train_loss": f"{train_loss:.4f}"})

        if log_wandb:
            wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})

        if (epoch + 1) % validate_every == 0:
            val_stats = evaluate_epoch(
                model, val_loader, criterion, device, log_wandb=log_wandb
            )

            if val_stats["val_f1_score"] > best_val_f1 and save_best_f1_model:
                best_val_f1 = val_stats["val_f1_score"]
                best_model_path = os.path.join(
                    out_dir, f"best_model_epoch_{epoch+1}.pth"
                )
                torch.save(model.state_dict(), best_model_path)
                print(
                    f"New best model saved at epoch {epoch+1} with F1: {best_val_f1:.4f}"
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
