import albumentations as A
import torch

DATASET_MEAN = (0.33298134, 0.33009373, 0.29579783)
DATASET_STD = (0.18409964, 0.17780256, 0.17631003)


def augmented_transform(width: int = 400, height: int = 400) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(width, height), scale=(0.7, 1.0), ratio=(0.9, 1.1), p=1.0
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=180, p=0.9
            ),
            A.RandomBrightnessContrast(p=0.8),
            A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
            A.ToTensorV2(),
        ]
    )


def default_transform() -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
            A.ToTensorV2(),
        ]
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
