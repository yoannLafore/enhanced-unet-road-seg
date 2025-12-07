import torch
import wandb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from src.preprocessing.transform import denormalize_tensor


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


def evaluate_epoch_improve_module(
    base_model,
    improvement_model,
    dataloader,
    criterion,
    device,
    threshold=0.5,
    log_wandb=True,
    forward_features=False,
):
    base_model.eval()
    improvement_model.eval()

    def pred_improved_get_fn(images):
        if forward_features:
            base_logits, base_preds, features = base_model(images, get_features=True)

            # Use features as additional input to improvement model
            improvement_model_input = torch.cat([base_preds, features], dim=1)
            improved_logits, improved_preds = improvement_model(improvement_model_input)
            return improved_preds
        else:
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
        "",
        None,
    )

    return stats_base, stats_improved
