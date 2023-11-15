from typing import Dict

import torch

from .warmup_cosine import LinearWarmupCosineAnnealingLR


def build_scheduler(optimizer, lr_config: Dict):
    schedule_name = lr_config["name"].lower()
    if schedule_name == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, **lr_config["config"])
    elif schedule_name == "cosine":
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, **lr_config["config"])
    else:
        raise NotImplementedError(f"got not implemented scheduler : {schedule_name}")
    return scheduler
