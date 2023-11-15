from torch.utils.tensorboard import SummaryWriter

from .global_env import GlobalEnv


def master_only_decorator(f):
    def process(*args, **kwargs):
        if GlobalEnv.get().master:
            f(*args, **kwargs)

    return process


def decorator_all_methods(decorator):
    def class_decorator(cls):
        for attr_name in dir(cls):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(cls, attr_name)
            if callable(attr_value):
                setattr(cls, attr_name, decorator(attr_value))
        return cls

    return class_decorator


@decorator_all_methods(master_only_decorator)
class DistSummaryWriter(SummaryWriter):
    def __init__(self, *args, **kwargs):
        if GlobalEnv.get().master:
            super().__init__(*args, **kwargs)
