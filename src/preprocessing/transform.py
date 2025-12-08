import albumentations as A
import torch
import cv2

DATASET_MEAN = (0.33298134, 0.33009373, 0.29579783)
DATASET_STD = (0.18409964, 0.17780256, 0.17631003)


def augmented_transform(
    width: int = 400, height: int = 400, seed: int = 42
) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(height, width),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.7,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent=(-0.03, 0.03),
                scale=(0.95, 1.05),
                rotate=(-30, 30),
                mode=cv2.BORDER_REFLECT_101,
                p=0.7,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.8,
            ),
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.5,
            ),
            A.GaussNoise(p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.ImageCompression(p=0.3),
            A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
            A.ToTensorV2(),
        ],
        seed=seed,
    )


def default_transform(seed: int = 42) -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
            A.ToTensorV2(),
        ],
        seed=seed,
    )


def denormalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize a tensor using the dataset mean and std.

    Args:
        tensor (torch.Tensor): Normalized tensor of shape (C, H, W) or (N, C, H, W).

    Returns:
        torch.Tensor: Denormalized tensor.
    """
    mean = torch.tensor(DATASET_MEAN).view(-1, 1, 1)
    std = torch.tensor(DATASET_STD).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    denormalized_tensor = tensor * std + mean
    return denormalized_tensor
