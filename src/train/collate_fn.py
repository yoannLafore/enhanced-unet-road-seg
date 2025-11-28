from src.model.unet import *
from src.preprocessing.dataset import *
import torch
from src.utils import *


def get_base_collate_fn():
    return base_collate_fn


def base_collate_fn(batch):
    """
    Base collate function for stacking images and masks.
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack([item[1] for item in batch], dim=0)
    return images, masks
