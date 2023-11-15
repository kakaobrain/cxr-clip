import logging
import random

import numpy as np
import torch
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def seed_everything(seed: int):
    log.info("Global seed set to %d", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return seed


def convert_dictconfig_to_dict(cfg):
    if isinstance(cfg, DictConfig):
        return {k: convert_dictconfig_to_dict(v) for k, v in cfg.items()}
    else:
        return cfg
