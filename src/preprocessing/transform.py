import albumentations as A

DATASET_MEAN = (0.33298134, 0.33009373, 0.29579783)
DATASET_STD = (0.18409964, 0.17780256, 0.17631003)


def augmented_transform(width: int = 400, height: int = 400) -> A.Compose:
    return [
        A.RandomResizedCrop(size=(400, 400), scale=(0.7, 1.0), ratio=(0.9, 1.1), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=180, p=0.9),
        A.RandomBrightnessContrast(p=0.8),
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        A.ToTensorV2(),
    ]


default_transform = A.Compose(
    [
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        A.ToTensorV2(),
    ]
)
