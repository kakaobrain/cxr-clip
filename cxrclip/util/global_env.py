import collections
import os

import torch
import torch.distributed as dist


class SummaryWriter:
    def __init__(self):
        self.train = None
        self.valid = None
        self.global_step = 0


class GlobalEnv:
    _instance = None

    @staticmethod
    def get():
        if GlobalEnv._instance is None:
            GlobalEnv()
        return GlobalEnv._instance

    def __init__(self):
        if GlobalEnv._instance is not None:
            raise Exception("This class is a singleton")

        Tuples = collections.namedtuple("DistEnv", ["world_size", "world_rank", "local_rank", "num_gpus", "master", "summary_writer"])
        if dist.is_initialized():
            GlobalEnv._instance = Tuples(
                dist.get_world_size(), dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0)), 1, dist.get_rank() == 0, SummaryWriter()
            )
        else:
            GlobalEnv._instance = Tuples(1, 0, 0, torch.cuda.device_count(), True, SummaryWriter())
