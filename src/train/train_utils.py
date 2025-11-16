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
def train_epoch(model, dataloader, optimizer, criterion, device, log_wandb=True):
    model.train()
    running_loss = 0.0

    for images, masks in dataloader:
        # Move to device
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()

        # Forward pass
        _, outputs = model(images)

        # Compute loss and backpropagate
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        # Log to wandb
        if log_wandb:
            wandb.log({"train_loss": loss.item()})

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate_epoch(model, dataloader, criterion, device, threshold=0.5, log_wandb=True):
    # Compute accuracy, precision, recall, F1-score on validation set
    # Also generate the mask, compare them to ground truth and save to wandb
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        preds_list = []
        masks_list = []
        imgs_list = []

        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            _, outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)

            # Compute accuracy, precision, recall, F1-score
            # Flatten tensors
            preds_flat = (outputs.view(-1) > threshold).long()
            masks_flat = (masks.view(-1) > threshold).long()

            # Store for overall metrics
            denormalized_imgs = denormalize_tensor(images.cpu()).clamp(0, 1)

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
            "val_loss": epoch_loss,
            "val_accuracy": accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1_score": f1,
        }

        # Log to wandb
        if log_wandb:
            wandb.log(stats)
            wandb.log(
                {
                    "images": [wandb.Image(img, caption="Input Image") for img in imgs],
                    "predicted_masks": [
                        wandb.Image(img, caption="Predicted Mask") for img in preds
                    ],
                    "ground_truth_masks": [
                        wandb.Image(img, caption="Ground Truth Mask") for img in masks
                    ],
                }
            )

    return stats
