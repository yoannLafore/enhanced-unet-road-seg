import torch
import wandb


# Train the basic model for one epoch
def train_epoch(
    model, dataloader, optimizer, criterion, device, log_wandb=True, use_preds=False
):
    model.train()
    running_loss = 0.0

    for images, masks in dataloader:
        # Move to device
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()

        # Forward pass
        logits, preds = model(images)

        # Compute loss and backpropagate
        if use_preds:
            loss = criterion(preds, masks)
        else:
            loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        # Log to wandb
        if log_wandb:
            wandb.log({"train_loss": loss.item()})

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def train_epoch_improve_module_sep(
    base_model,
    improvement_model,
    base_opt,
    improvement_opt,
    dataloader,
    base_criterion,
    improvement_criterion,
    device,
    log_wandb=True,
    forward_features=False,
):
    base_model.train()
    improvement_model.train()

    running_loss_base = 0.0
    running_loss_improved = 0.0

    for images, labels in dataloader:
        images, masks = images.to(device), labels.to(device)

        base_opt.zero_grad()
        improvement_opt.zero_grad()

        if forward_features:
            base_logits, base_preds, features = base_model(images, get_features=True)

            # Use features as additional input to improvement model
            improvement_model_input = torch.cat(
                [base_preds.detach(), features.detach()], dim=1
            )
            improved_logits, improved_preds = improvement_model(improvement_model_input)
        else:
            base_logits, base_preds = base_model(images)
            # /!\ Use detached predictions as input to improvement model
            improved_logits, improved_preds = improvement_model(base_preds.detach())

        # Compute losses
        loss_base = base_criterion(base_logits, masks)
        loss_improved = improvement_criterion(improved_logits, masks)

        # Back propagate
        loss_base.backward()
        loss_improved.backward()

        # Step
        base_opt.step()
        improvement_opt.step()

        running_loss_base += loss_base.item() * images.size(0)
        running_loss_improved += loss_improved.item() * images.size(0)

        if log_wandb:
            wandb.log({"base": loss_base.item(), "improved": loss_improved.item()})

    epoch_loss_base = running_loss_base / len(dataloader.dataset)
    epoch_loss_improved = running_loss_improved / len(dataloader.dataset)

    return epoch_loss_base, epoch_loss_improved


def train_epoch_improve_module_joint(
    base_model,
    improvement_model,
    optimizer,
    dataloader,
    criterion,
    device,
    log_wandb=True,
    forward_features=False,
):
    base_model.train()
    improvement_model.train()

    running_loss = 0.0

    for images, labels in dataloader:
        images, masks = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if forward_features:
            base_logits, base_preds, features = base_model(images, get_features=True)

            # Use features as additional input to improvement model
            improvement_model_input = torch.cat([base_preds, features], dim=1)
            improved_logits, improved_preds = improvement_model(improvement_model_input)
        else:
            base_logits, base_preds = base_model(images)
            improved_logits, improved_preds = improvement_model(base_preds)

        # Compute losses
        loss = criterion(improved_logits, masks)

        # Back propagate
        loss.backward()

        # Step
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        if log_wandb:
            wandb.log({"loss": loss.item()})

    epoch_loss = running_loss / len(dataloader.dataset)

    return epoch_loss
