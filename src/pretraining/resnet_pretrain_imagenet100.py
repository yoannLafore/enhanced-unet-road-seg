import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from tqdm import tqdm

torch.backends.cudnn.benchmark = True


def build_dataloaders(data_root, batch_size=128, num_workers=8):
    """
    Expects:
        data_root/
            train/
                class_a/
                class_b/
                ...
            val/
                class_a/
                class_b/
                ...
    """
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    train_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_ds = torchvision.datasets.ImageFolder(
        root=train_dir,
        transform=train_tf,
    )

    val_ds = torchvision.datasets.ImageFolder(
        root=val_dir,
        transform=val_tf,
    )

    num_classes = len(train_ds.classes)
    print(f"Found {num_classes} classes in {train_dir}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, num_classes


def build_resnet(num_classes, device, resnet_type="resnet50"):
    """
    ResNet from scratch (NO pretrained weights), adapted to num_classes.
    """
    if resnet_type == "resnet18":
        model = torchvision.models.resnet18(weights=None)
    elif resnet_type == "resnet34":
        model = torchvision.models.resnet34(weights=None)
    elif resnet_type == "resnet50":
        model = torchvision.models.resnet50(weights=None)
    elif resnet_type == "resnet101":
        model = torchvision.models.resnet101(weights=None)
    else:
        raise ValueError(f"Unsupported resnet_type '{resnet_type}'")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.to(device)
    return model


def run_epoch(model, loader, optimizer=None, device="cuda"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for imgs, labels in tqdm(loader, desc="Training" if is_train else "Validating"):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/imagenet100",
        help="Path to extracted Kaggle ImageNet-100 root (containing train/ and val/).",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--resnet_type", type=str, default="resnet50")
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Data root: {args.data_root}")

    save_path = args.save_path
    if save_path is None:
        save_path = f"{args.resnet_type}_imagenet100.pth"

    train_loader, val_loader, num_classes = build_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_resnet(num_classes, device, resnet_type=args.resnet_type)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer=optimizer, device=device
        )
        val_loss, val_acc = run_epoch(model, val_loader, optimizer=None, device=device)

        dt = time.time() - t0
        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"- {dt:.1f}s "
            f"- train loss: {train_loss:.4f}, acc: {train_acc*100:.2f}% "
            f"- val loss: {val_loss:.4f}, acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  🔹 New best checkpoint saved to {save_path}")

    total_time = (time.time() - start_time) / 60
    print(f"\nDone in {total_time:.1f} minutes.")
    print(f"Best val accuracy: {best_val_acc*100:.2f}%")
    print(f"Final weights saved at: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    main()
