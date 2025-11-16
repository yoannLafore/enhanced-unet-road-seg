from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
import numpy as np


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
