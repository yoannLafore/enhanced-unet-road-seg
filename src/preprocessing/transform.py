import albumentations as A
import torch
import cv2


def augmented_transform(
    width: int = 400, height: int = 400, seed: int = 42
) -> A.Compose:
    """Create an augmented transformation pipeline for image preprocessing.

    Args:
        width (int): Target width of the images after transformation.
        height (int): Target height of the images after transformation.
        seed (int): Random seed for reproducibility.
    Returns:
        A.Compose: The composed augmentation pipeline.
    """
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(height, width),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.5,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent=(-0.03, 0.03),
                scale=(0.95, 1.05),
                rotate=(-15, 15),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.4,
            ),
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.25,
            ),
            A.ToFloat(max_value=255.0),
            A.ToTensorV2(),
        ],
        seed=seed,
    )


def default_transform(seed: int = 42) -> A.Compose:
    """Create a default transformation pipeline for image preprocessing.

    Args:
        seed (int): Random seed for reproducibility. Although no randomness is used here, it's included for call consistency.
    Returns:
        A.Compose: The composed augmentation pipeline.
    """
    return A.Compose(
        [
            A.ToFloat(max_value=255.0),
            A.ToTensorV2(),
        ],
        seed=seed,
    )
