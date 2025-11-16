from src.preprocessing.transform import augmented_transform, default_transform
from src.model.unet import *
from src.preprocessing.dataset import *
import os
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
import wandb


from src.train.train_utils import evaluate_epoch, train_epoch

# TODO :
# - Why are the validation metrics doing zigzag ?
# - Do we apply softmax and stuff,
# - Do a correct float tensor conversion in dataset
# - Try with other losses (Dice, Focal ... )
# - Finetune threshold
# - LR scheduler : Try CosineAnnealing with warm restarts
# TODO: VERY IMPORTANT : COMPUTE VALIDATION METRICS OVER THE FULL DATASET not per BATCH !!!
# TODO: For now we use cross entropy with loagits, but the mask values are 0 and 1, might be better to apply a sigmoid first and use BCE loss ?


def main():
    # Create checkpoint directory
    checkpoint_dir = "./checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the dataset
    train_dir = "/home/yoann/Desktop/project-2-roadseg_nsy/data/training/"
    train_dataset, val_dataset = load_train_test(
        train_dir, train_transform=augmented_transform()
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
            "lr_scheduler": lr_scheduler.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "weight_decay": weight_decay,
            "epochs": nb_epochs,
            "batch_size": batch_size,
        },
    )

    progress = tqdm(range(nb_epochs), desc="Training Epochs")

    for epoch in progress:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        lr_scheduler.step(train_loss)
        progress.set_postfix({"train_loss": f"{train_loss:.4f}"})
        wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})

        if (epoch + 1) % validate_every == 0:
            evaluate_epoch(model, val_loader, criterion, device)

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
