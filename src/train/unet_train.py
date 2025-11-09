from src.model.unet import *
from src.preprocessing.dataset import *
import os
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from tqdm import tqdm
import wandb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# TODO :
# - Why are the validation metrics doing zigzag ?
# - Do we apply softmax and stuff,
# - Do a correct float tensor conversion in dataset
# - Try with other losses (Dice, Focal ... )
# - Finetune threshold
# - LR scheduler : Try CosineAnnealing with warm restarts
# TODO: VERY IMPORTANT : COMPUTE VALIDATION METRICS OVER THE FULL DATASET not per BATCH !!!
# TODO: For now we use cross entropy with loagits, but the mask values are 0 and 1, might be better to apply a sigmoid first and use BCE loss ?


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        masks = masks.unsqueeze(1)  # Add channel dimension
        optimizer.zero_grad()

        _, outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        wandb.log({"train_loss": loss.item()})

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate_epoch(model, dataloader, criterion, device):
    # Compute accuracy, precision, recall, F1-score on validation set
    # Also generate the mask, compare them to ground truth and save to wandb
    model.eval()
    running_loss = 0.0

    with torch.no_grad():

        preds_list = []
        masks_list = []

        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            masks = masks.unsqueeze(1)  # Add channel dimension

            _, outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)

            # Generate predicted masks
            # TODO : Shouldn't we apply a softmax here ?
            preds = (outputs > 0.5).float()

            wandb.log(
                {"images": [wandb.Image(img, caption="Input Image") for img in images]}
            )
            wandb.log(
                {
                    "predicted_masks": [
                        wandb.Image(img, caption="Predicted Mask") for img in preds
                    ]
                }
            )
            wandb.log(
                {
                    "ground_truth_masks": [
                        wandb.Image(img, caption="Ground Truth Mask") for img in masks
                    ]
                }
            )

            # Compute accuracy, precision, recall, F1-score
            # Flatten tensors
            preds_flat = (preds.view(-1) > 0.5).long()
            masks_flat = (masks.view(-1) > 0.5).long()

            preds_list.append(preds_flat)
            masks_list.append(masks_flat)

        preds_flat = torch.cat(preds_list)
        masks_flat = torch.cat(masks_list)

        accuracy = accuracy_score(masks_flat.cpu(), preds_flat.cpu())
        precision = precision_score(masks_flat.cpu(), preds_flat.cpu())
        recall = recall_score(masks_flat.cpu(), preds_flat.cpu())
        f1 = f1_score(masks_flat.cpu(), preds_flat.cpu())

        epoch_loss = running_loss / len(dataloader.dataset)

        wandb.log(
            {
                "val_loss": epoch_loss,
                "val_accuracy": accuracy,
                "val_precision": precision,
                "val_recall": recall,
                "val_f1_score": f1,
            }
        )

    return epoch_loss


def main():
    # Create checkpoint directory
    checkpoint_dir = "./checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the dataset
    train_dir = "/home/yoann/Desktop/project-2-roadseg_nsy/data/training/"
    img_dir = os.path.join(train_dir, "images/")
    mask_dir = os.path.join(train_dir, "groundtruth/")

    all_imgs = sorted(
        os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")
    )
    all_masks = sorted(
        os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")
    )

    all_imgs_train, all_imgs_val, all_masks_train, all_masks_val = train_test_split(
        all_imgs, all_masks, test_size=0.2, random_state=42
    )

    print(f"Total number of training samples: {len(all_imgs_train)}")
    print(f"Total number of validation samples: {len(all_imgs_val)}")

    normalize_step = A.Normalize(
        mean=(0.33298134, 0.33009373, 0.29579783),
        std=(0.18409964, 0.17780256, 0.17631003),
    )

    # Transform for datasets
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=180, p=0.9
            ),
            A.RandomBrightnessContrast(p=0.8),
            normalize_step,
            ToTensorV2(),
        ]
    )

    default_transform = A.Compose(
        [
            normalize_step,
            ToTensorV2(),
        ]
    )

    # Create dataset objects
    train_dataset = RoadSegDataset(all_imgs_train, all_masks_train, transform=transform)
    val_dataset = RoadSegDataset(
        all_imgs_val, all_masks_val, transform=default_transform
    )

    # Collate function to handle batching
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        masks = torch.stack([item[1] for item in batch])
        return images, masks

    batch_size = 8

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Init the model
    model = Unet().to(device)

    lr = 1e-4
    weight_decay = 1e-3

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    nb_epochs = 100
    steps_per_epochs = len(train_loader)
    total_steps = nb_epochs * steps_per_epochs
    validate_every = 5
    checkpoint_every = 10

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    # TODO: Are we really working with logits here ... ?
    criterion = nn.BCELoss()

    wandb.init(
        project="road-segmentation",
        config={
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "epochs": nb_epochs,
            "batch_size": 16,
        },
    )

    for epoch in tqdm(range(nb_epochs), desc="Training Epochs"):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        if (epoch + 1) % validate_every == 0:
            val_loss = evaluate_epoch(model, val_loader, criterion, device)
            lr_scheduler.step(val_loss)

            print(
                f"Epoch [{epoch+1}/{nb_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
        else:
            print(f"Epoch [{epoch+1}/{nb_epochs}], Train Loss: {train_loss:.4f}")

        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"unet_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "unet_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model: {final_model_path}")

    return


if __name__ == "__main__":

    main()
