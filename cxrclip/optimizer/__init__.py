import logging
from typing import Dict

import torch
from torch import nn

log = logging.getLogger(__name__)


def build_optimizer(model: nn.Module, optim_config: Dict):

    no_decay = [name.lower() for name in getattr(optim_config, "no_decay", [])]
    if no_decay:
        wd = optim_config["config"]["weight_decay"]
        params = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)], "weight_decay": wd},
            {"params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)], "weight_decay": 0.0},
        ]
        log.info("seperated no decay params (#params:%d no-decay #params:%d)", len(params[0]["params"]), len(params[1]["params"]))
    else:
        params = model.parameters()

    optim_name = optim_config["name"].lower()
    if optim_name == "sgd":
        optimizer = torch.optim.SGD(params, **optim_config["config"])
    elif optim_name == "adamw":
        optimizer = torch.optim.AdamW(params, **optim_config["config"])
    else:
        raise NotImplementedError(f"Not implemented optimizer : {optim_name}")
    return optimizer
