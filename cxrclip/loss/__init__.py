from typing import Dict

from .classification import Classification
from .combined_loss import CombinedLoss
from .cxr_clip import CXRClip


def build_loss(all_loss_config: Dict) -> CombinedLoss:
    loss_list = []

    for loss_config in all_loss_config:
        cfg = all_loss_config[loss_config]
        if cfg["loss_ratio"] == 0.0:
            continue
        if loss_config == "classification":
            loss = Classification(**cfg)
        elif loss_config == "cxr_clip":
            loss = CXRClip(**cfg)
        else:
            raise KeyError(f"Unknown loss: {loss_config}")

        loss_list.append(loss)

    total_loss = CombinedLoss(loss_list)
    return total_loss
