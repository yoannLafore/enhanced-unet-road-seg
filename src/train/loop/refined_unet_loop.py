from src.train.loop.loop_utils import *


def train_improve_model_on_ds(train_ds, test_ds, cfg):
    print(f"Using out dir: {cfg.train.out_dir}")

    train_cfg = cfg.train
    device = cfg.device

    base_model_cfg = cfg.base_model
    base_model = build_from_cfg(base_model_cfg).to(device)

    improved_model_cfg = cfg.improved_model
    improved_model = build_from_cfg(improved_model_cfg).to(device)

    train_loader, val_loader = get_dataloaders(train_ds, test_ds, cfg)

    train_jointly = bool(train_cfg.train_jointly)
    forward_features = bool(train_cfg.get("forward_features", False))

    # Make sure output directory exists
    out_dir = train_cfg.out_dir
    checkpoint_dir, validation_file, final_stats_file = make_out_files_dirs(out_dir)

    # Save config file
    save_config_file(out_dir, cfg)

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
        # Training step
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
            base_lr_scheduler.step()
            improved_lr_scheduler.step()
            progress.set_postfix(
                {
                    "train_loss": f"{train_loss_improved:.4f}",
                }
            )

        # WanDB logging
        if log_wandb:
            if train_jointly:
                wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})
            else:
                wandb.log({"base_learning_rate": base_optimizer.param_groups[0]["lr"]})
                wandb.log(
                    {"improved_learning_rate": improved_optimizer.param_groups[0]["lr"]}
                )

        # Validation step
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

        # Checkpoint step
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
