import logging
from typing import Dict

from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from cxrclip import util

from .data_utils import load_tokenizer
from .datasets import load_dataset

log = logging.getLogger(__name__)


class DataModule:
    def __init__(
        self,
        data_config: Dict,
        dataloader_config: Dict = None,
        tokenizer_config: Dict = None,
        loss_config: Dict = None,
        transform_config: Dict = None,
    ):
        self.data_config = data_config
        self.dataloader_config = dataloader_config
        self.tokenizer_config = tokenizer_config
        self.loss_config = loss_config
        self.tokenizer = load_tokenizer(**self.tokenizer_config) if self.tokenizer_config is not None else None

        self.datasets = {"train": [], "valid": [], "test": []}

        self.train_loader = None
        self.valid_loader_dict = None
        self.test_loader = None

        for split in data_config:
            dataset_split_config = self.data_config[split]
            for name in dataset_split_config:
                dataset_config = dataset_split_config[name]
                dataset = load_dataset(
                    split=split, tokenizer=self.tokenizer, transform_config=transform_config, loss_config=self.loss_config, **dataset_config
                )
                self.datasets[split].append(dataset)

                log.info(f"Dataset loaded: {dataset_split_config[name]['name']} for {split}")

    def train_dataloader(self, distributed):
        assert self.dataloader_config is not None

        if self.train_loader is None:
            dataset = ConcatDataset(self.datasets["train"])
            shuffle = self.dataloader_config["train"]["shuffle"]
            if distributed:
                self.dataloader_config["train"]["shuffle"] = False
                if self.dataloader_config["train"]["batch_size"] % util.GlobalEnv.get().world_size != 0:
                    raise Exception(
                        f'train.batch_size({self.dataloader_config["train"]["batch_size"]}) \
                            is must be a multiple of world_size({util.GlobalEnv.get().world_size})'
                    )
                self.dataloader_config["train"]["batch_size"] = (
                    self.dataloader_config["train"]["batch_size"] // util.GlobalEnv.get().world_size
                )

            self.train_sampler = DistributedSampler(dataset=dataset, shuffle=shuffle) if distributed else None
            self.train_loader = DataLoader(
                dataset,
                collate_fn=getattr(self.datasets["train"][0], "collate_fn", None),
                sampler=self.train_sampler,
                **self.dataloader_config["train"],
            )

        return self.train_loader, self.train_sampler

    def valid_dataloader(self, distributed=False):
        assert self.dataloader_config is not None
        if self.valid_loader_dict is None:
            self.valid_loader_dict = dict()
            if self.dataloader_config["valid"]["batch_size"] % util.GlobalEnv.get().world_size != 0:
                raise Exception(
                    f'valid.batch_size({self.dataloader_config["valid"]["batch_size"]}) \
                        is must be a multiple of world_size({util.GlobalEnv.get().world_size})'
                )
            self.dataloader_config["valid"]["batch_size"] = self.dataloader_config["valid"]["batch_size"] // util.GlobalEnv.get().world_size

            for val_dataset in self.datasets["valid"]:
                sampler = DistributedSampler(dataset=val_dataset, shuffle=False) if distributed else None
                if sampler is not None:
                    sampler.set_epoch(0)
                dataloader = DataLoader(
                    val_dataset, collate_fn=getattr(val_dataset, "collate_fn", None), sampler=sampler, **self.dataloader_config["valid"]
                )
                self.valid_loader_dict[val_dataset.name] = dataloader

        return self.valid_loader_dict

    def test_dataloader(self):
        assert self.dataloader_config is not None
        if self.test_loader is None:
            self.test_loader = {
                test_dataset.name: DataLoader(
                    test_dataset, collate_fn=getattr(test_dataset, "collate_fn", None), **self.dataloader_config["test"]
                )
                for test_dataset in self.datasets["test"]
            }
        return self.test_loader
