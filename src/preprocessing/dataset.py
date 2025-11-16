from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
import numpy as np
from src.preprocessing.transform import default_transform
import os


class RoadSegDataset(Dataset):
    def __init__(self, img_paths: list[str], mask_paths: list[str], transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = np.array(image)
        mask = np.array(mask)

        # Map mask values to 0 and 1
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            mask = mask.unsqueeze(0)  # Add channel dimension
            # TODO : How do we make sure both are float tensors?

        return image, mask


def load_train_test(
    data_dir: str,
    train_transform,
    test_transform=default_transform,
    test_size: float = 0.2,
    random_state: int = 42,
):
    # Load the paths
    img_dir = os.path.join(data_dir, "images/")
    mask_dir = os.path.join(data_dir, "groundtruth/")

    all_imgs = sorted(
        os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")
    )
    all_masks = sorted(
        os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")
    )

    # Split into train and test
    all_imgs_train, all_imgs_test, all_masks_train, all_masks_test = train_test_split(
        all_imgs, all_masks, test_size=test_size, random_state=random_state
    )

    # Create dataset objects
    train_dataset = RoadSegDataset(
        all_imgs_train, all_masks_train, transform=train_transform
    )
    test_dataset = RoadSegDataset(
        all_imgs_test, all_masks_test, transform=test_transform
    )

    return train_dataset, test_dataset
