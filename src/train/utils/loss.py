import torch.nn as nn
import torch


def get_bce_with_logits_loss(pos_weight=None):
    """Get Binary Cross Entropy with Logits Loss.

    Args:
        pos_weight (float, optional): A weight of positive examples. Defaults to None.

    Returns:
        nn.Module: BCE with Logits Loss module.
    """
    if pos_weight is not None:
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    else:
        return nn.BCEWithLogitsLoss()
