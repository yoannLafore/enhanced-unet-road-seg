import torch
import wandb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from src.preprocessing.transform import denormalize_tensor


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


def _evaluate_epoch(
    pred_get_fn,
    dataloader,
    criterion,
    device,
    threshold=0.5,
    log_wandb=True,
    prefix="",
    log_blacklist=None,
):
    # Compute accuracy, precision, recall, F1-score on validation set
    # Also generate the mask, compare them to ground truth and save to wandb
    running_loss = 0.0

    with torch.no_grad():
        preds_list = []
        masks_list = []
        imgs_list = []

        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = pred_get_fn(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)

            # Compute accuracy, precision, recall, F1-score
            preds_flat = (outputs.view(-1) > threshold).long()
            masks_flat = (masks.view(-1) > threshold).long()

            # Store for overall metrics
            denormalized_imgs = denormalize_tensor(images[:, :3, :, :].cpu()).clamp(
                0, 1
            )

            preds_list.append(preds_flat)
            masks_list.append(masks_flat)
            imgs_list.append(denormalized_imgs)

        # Concatenate all tensors
        preds_flat = torch.cat(preds_list)
        masks_flat = torch.cat(masks_list)
        imgs = torch.cat(imgs_list)

        # Unflatten for wandb logging
        preds = preds_flat.view(-1, 1, imgs.size(2), imgs.size(3))
        masks = masks_flat.view(-1, 1, imgs.size(2), imgs.size(3))

        # Compute metrics
        accuracy = accuracy_score(masks_flat.cpu(), preds_flat.cpu())
        precision = precision_score(masks_flat.cpu(), preds_flat.cpu())
        recall = recall_score(masks_flat.cpu(), preds_flat.cpu())
        f1 = f1_score(masks_flat.cpu(), preds_flat.cpu())

        epoch_loss = running_loss / len(dataloader.dataset)

        stats = {
            prefix + "val_loss": epoch_loss,
            prefix + "val_accuracy": accuracy,
            prefix + "val_precision": precision,
            prefix + "val_recall": recall,
            prefix + "val_f1_score": f1,
        }

        # Log to wandb
        if log_wandb:
            if log_blacklist is not None:
                for key in log_blacklist:
                    if key in stats:
                        del stats[key]

            wandb.log(stats)

            if log_blacklist is None or "images" not in log_blacklist:
                wandb.log(
                    {
                        prefix
                        + "images": [
                            wandb.Image(img, caption="Input Image") for img in imgs
                        ],
                    }
                )

            if log_blacklist is None or "predicted_masks" not in log_blacklist:
                wandb.log(
                    {
                        prefix
                        + "predicted_masks": [
                            wandb.Image(img, caption="Predicted Mask") for img in preds
                        ],
                    }
                )
            if log_blacklist is None or "ground_truth_masks" not in log_blacklist:
                wandb.log(
                    {
                        prefix
                        + "ground_truth_masks": [
                            wandb.Image(img, caption="Ground Truth Mask")
                            for img in masks
                        ],
                    }
                )

    return stats


def evaluate_epoch(model, dataloader, criterion, device, threshold=0.5, log_wandb=True):
    model.eval()

    def pred_get_fn(images):
        logits, preds = model(images)
        return preds

    stats = _evaluate_epoch(
        pred_get_fn,
        dataloader,
        criterion,
        device,
        threshold,
        log_wandb,
    )

    return stats


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
):
    base_model.train()
    improvement_model.train()

    running_loss_base = 0.0
    running_loss_improved = 0.0

    for images, labels in dataloader:
        images, masks = images.to(device), labels.to(device)

        base_opt.zero_grad()
        improvement_opt.zero_grad()

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
):
    base_model.train()
    improvement_model.train()

    running_loss = 0.0

    for images, labels in dataloader:
        images, masks = images.to(device), labels.to(device)

        optimizer.zero_grad()

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


def evaluate_epoch_improve_module(
    base_model,
    improvement_model,
    dataloader,
    criterion,
    device,
    threshold=0.5,
    log_wandb=True,
):
    base_model.eval()
    improvement_model.eval()

    def pred_improved_get_fn(images):
        base_logits, base_preds = base_model(images)
        improved_logits, improved_preds = improvement_model(base_preds)
        return improved_preds

    def base_pred_get_fn(images):
        base_logits, base_preds = base_model(images)
        return base_preds

    stats_base = _evaluate_epoch(
        base_pred_get_fn,
        dataloader,
        criterion,
        device,
        threshold,
        log_wandb,
        "base_",
        ["images", "ground_truth_masks"],
    )

    stats_improved = _evaluate_epoch(
        pred_improved_get_fn,
        dataloader,
        criterion,
        device,
        threshold,
        log_wandb,
        "improved_",
        None,
    )

    return stats_base, stats_improved
