import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from cxrclip import convert_dictconfig_to_dict, run, seed_everything

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):

    OmegaConf.resolve(cfg)

    if "LOCAL_RANK" in os.environ:
        # for ddp
        # passed by torchrun or torch.distributed.launch
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # for debugging
        local_rank = -1

    if local_rank < 1:
        log.info(f"Configurations:\n{OmegaConf.to_yaml(cfg)}")

    seed_everything(cfg.base.seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    cfg = convert_dictconfig_to_dict(cfg)
    run(local_rank, cfg)


if __name__ == "__main__":
    main()
