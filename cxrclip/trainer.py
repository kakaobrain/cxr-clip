import logging
import math
import os
import shutil
from typing import Dict

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from . import util
from .data import DataModule
from .loss import build_loss
from .model import build_model
from .optimizer import build_optimizer
from .scheduler import build_scheduler

log = logging.getLogger(__name__)


def run(local_rank, cfg: Dict):
    if "tokenizer" in cfg:
        assert (
            cfg["tokenizer"]["pretrained_model_name_or_path"] == cfg["model"]["text_encoder"]["name"]
        ), "tokenizer should be same to text_encoder"

    distributed = local_rank != -1
    if distributed:
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"DistEnv: {util.GlobalEnv.get()}")

    log.info(f"{device}: Load datasets")
    data_config = {}
    if "data_train" in cfg:
        data_config["train"] = cfg["data_train"]
    if "data_valid" in cfg:
        data_config["valid"] = cfg["data_valid"]
    if "data_test" in cfg:
        data_config["test"] = cfg["data_test"]

    if cfg["model"]["image_encoder"]["name"] == "resnet":
        for _split in data_config:
            for _dataset in data_config[_split]:
                data_config[_split][_dataset]["normalize"] = "imagenet"

    datamodule = DataModule(
        data_config=data_config,
        dataloader_config=cfg["dataloader"],
        tokenizer_config=cfg["tokenizer"] if "tokenizer" in cfg else None,
        loss_config=cfg["loss"],
        transform_config=cfg["transform"],
    )
    train_dataloader, train_sampler = datamodule.train_dataloader(distributed=distributed)
    valid_dataloaders = datamodule.valid_dataloader(distributed=distributed)

    log.info(f"{device}: Build the model")
    model = build_model(cfg["model"], cfg["loss"], datamodule.tokenizer)
    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[device], find_unused_parameters=True)
    if util.GlobalEnv.get().master:
        log.info(f"{device}: Model info:\n{model}")

    log.info(f"{device}: Build the loss function")
    loss_func = build_loss(cfg["loss"])

    log.info(f"{device}: Build the optimizer")
    optimizer = build_optimizer(model, cfg["optimizer"])

    log.info(f"{device}: Build the LR scheulder")
    if "total_epochs" in cfg["scheduler"]["config"]:
        # with open_dict(cfg):
        cfg["scheduler"]["config"]["total_steps"] = len(train_dataloader) * cfg["scheduler"]["config"]["total_epochs"]
    if "warmup_epochs" in cfg["scheduler"]["config"]:
        # with open_dict(cfg):
        if isinstance(cfg["scheduler"]["config"]["warmup_epochs"], int):
            cfg["scheduler"]["config"]["warmup_steps"] = len(train_dataloader) * cfg["scheduler"]["config"]["warmup_epochs"]
        elif isinstance(cfg["scheduler"]["config"]["warmup_epochs"], float):
            cfg["scheduler"]["config"]["warmup_steps"] = cfg["scheduler"]["config"]["warmup_epochs"]

    scheduler = build_scheduler(optimizer, cfg["scheduler"])
    scaler = torch.cuda.amp.GradScaler() if cfg["base"]["amp"] else None

    if local_rank < 1:
        import nltk

        log.info("Download nltk module")
        nltk.download("punkt")

    # train
    log.info(f"{device}: Train the model")
    if "total_epoch" in cfg["scheduler"]:
        total_epochs = cfg["scheduler"]["total_epoch"]
        cfg["scheduler"]["config"]["total_steps"] = total_epochs * len(train_dataloader)
    else:
        total_epochs = math.ceil(cfg["scheduler"]["config"]["total_steps"] / len(train_dataloader))

    # tensorboard
    util.GlobalEnv.get().summary_writer.train = util.DistSummaryWriter(cfg["base"]["output"]["tensorboard"] + "/train")
    util.GlobalEnv.get().summary_writer.valid = util.DistSummaryWriter(cfg["base"]["output"]["tensorboard"] + "/valid")
    util.GlobalEnv.get().summary_writer.global_step = 0
    util.GlobalEnv.get().summary_writer.train.add_text(
        "hyperparams/config", "\n".join(["\t" + line for line in OmegaConf.to_yaml(cfg).splitlines()]), 0
    )
    if util.GlobalEnv.get().master:
        os.makedirs(cfg["base"]["output"]["checkpoint"], exist_ok=True)

    # training
    # best_loss = ckpt['train_loss'] if cfg.test.resume else 9e9
    # epoch_resume = ckpt['epoch'] if cfg.test.resume else 0
    best_loss = 9e9
    epoch_resume = 0
    for epoch in range(epoch_resume, total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss_dict = train(
            model,
            device,
            loss_func,
            optimizer,
            scheduler,
            train_dataloader,
            epoch,
            total_epochs,
            scaler,
            cfg["scheduler"]["config"]["total_steps"],
        )

        val_loss_dict_per_dataset = validate(
            model, device, loss_func, valid_dataloaders, epoch, total_epochs, local_rank, cfg["base"]["amp"]
        )

        # tensorboard
        for k, v in train_loss_dict.items():
            util.GlobalEnv.get().summary_writer.train.add_scalar(f"loss_per_epoch/{k}", v, epoch + 1)

        avg_val_loss_per_loss = {"total": 0.0}
        for loss_key in loss_func.loss_list:
            avg_val_loss_per_loss[loss_key.name] = 0.0

        for data_name, loss_dict in val_loss_dict_per_dataset.items():
            for loss_key, v in loss_dict.items():
                util.GlobalEnv.get().summary_writer.valid.add_scalar(f"loss_per_epoch/{loss_key}/{data_name}", v, epoch + 1)
                avg_val_loss_per_loss[loss_key] += v

        for loss_key in avg_val_loss_per_loss:
            avg_val_loss_per_loss[loss_key] /= len(valid_dataloaders)
            util.GlobalEnv.get().summary_writer.valid.add_scalar(f"loss_per_epoch/{loss_key}", avg_val_loss_per_loss[loss_key], epoch + 1)

        if util.GlobalEnv.get().master:
            # checkpoint
            filename = os.path.join(cfg["base"]["output"]["checkpoint"], "model")
            checkpoint = f"{filename}-last.tar"
            model_state_dict = model.state_dict() if local_rank == -1 else model.module.state_dict()
            torch.save(
                {
                    "model": model_state_dict,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": cfg,
                    "epoch": epoch + 1,
                    "train_loss": train_loss_dict["total"],
                },
                checkpoint,
            )
            log.info(f"Epoch {epoch}, last-model saved")

            # best model
            if avg_val_loss_per_loss[cfg["base"]["loss_best"]] < best_loss:
                shutil.copyfile(checkpoint, f"{filename}-best.tar")
                log.info(f"{filename}-best.tar saved")
                best_loss = avg_val_loss_per_loss[cfg["base"]["loss_best"]]

    util.GlobalEnv.get().summary_writer.train.close()
    util.GlobalEnv.get().summary_writer.valid.close()
    log.info(f"{device}: Training has been completed")


def train(model, device, loss_func, optimizer, scheduler, dataloader, epoch, total_epochs, scaler, total_step, print_step=30):
    model.train()
    if util.GlobalEnv.get().local_rank < 1:
        progress_iter = tqdm(enumerate(dataloader), desc=f"[{epoch:03d}/{total_epochs:03d} epoch train]", total=len(dataloader))
    else:
        progress_iter = enumerate(dataloader)

    avg_loss_dict = {"total": 0.0}
    for k in loss_func.loss_list:
        avg_loss_dict[k.name] = 0.0

    for idx, batch in progress_iter:
        optimizer.zero_grad(set_to_none=True)

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(batch, device)
                loss_dict = loss_func(**outputs, is_train=True)
            total_loss = loss_dict["total"]
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch, device)
            loss_dict = loss_func(**outputs, is_train=True)
            total_loss = loss_dict["total"]
            total_loss.backward()
            optimizer.step()

        scheduler.step()
        util.GlobalEnv.get().summary_writer.global_step = scheduler._step_count

        for k in loss_dict:
            avg_loss_dict[k] += loss_dict[k].item()

        if idx % print_step == 0 and util.GlobalEnv.get().local_rank < 1:
            for k, lr in enumerate(scheduler.get_last_lr()):
                util.GlobalEnv.get().summary_writer.train.add_scalar(f"hyperparam/lr-{k}", lr, scheduler._step_count)
            util.GlobalEnv.get().summary_writer.train.add_scalar("loss", total_loss, scheduler._step_count)

            for k in loss_dict:
                util.GlobalEnv.get().summary_writer.train.add_scalar(f"loss/{k}", loss_dict[k], scheduler._step_count)

            progress_iter.set_postfix(
                {
                    "lr": [f"{v:.8f}" for v in scheduler.get_last_lr()],
                    "loss": f"{total_loss:.6f}",
                    "CUDA-Mem": f"{torch.cuda.memory_usage(device)}%",
                    "CUDA-Util": f"{torch.cuda.utilization(device)}%",
                }
            )
        if total_step == scheduler._step_count:
            break

    for k in avg_loss_dict:
        avg_loss_dict[k] = avg_loss_dict[k] / len(dataloader)

    return avg_loss_dict


def validate(model, device, loss_func, dataloader_dict, epoch, total_epochs, local_rank, amp, print_step=10):
    model.eval()
    loss_dict_per_dataset = dict()
    with torch.no_grad():
        for data_name, dataloader in dataloader_dict.items():
            avg_loss_dict = {"total": 0.0}
            for loss_key in loss_func.loss_list:
                avg_loss_dict[loss_key.name] = 0.0

            if util.GlobalEnv.get().local_rank < 1:
                progress_iter = tqdm(enumerate(dataloader), desc=f"[{epoch:03d}/{total_epochs:03d} epoch valid]", total=len(dataloader))
            else:
                progress_iter = enumerate(dataloader)

            for idx, batch in progress_iter:
                if amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch, device)
                        loss_dict = loss_func(**outputs, is_train=False)
                else:
                    outputs = model(batch, device)
                    loss_dict = loss_func(**outputs, is_train=False)

                if util.GlobalEnv.get().world_size > 1:
                    for loss_key in loss_dict:
                        dist.all_reduce(loss_dict[loss_key], dist.ReduceOp.SUM)
                        loss_dict[loss_key] = loss_dict[loss_key] / util.GlobalEnv.get().world_size

                for loss_key in loss_dict:
                    avg_loss_dict[loss_key] += loss_dict[loss_key].item()

                if (idx % print_step == 0 or idx == len(dataloader) - 1) and local_rank < 1:
                    progress_iter.set_postfix(
                        {
                            "loss": f'{avg_loss_dict["total"]:.6f}',
                            "CUDA-Mem(%)": torch.cuda.memory_usage(device),
                            "CUDA-Util(%)": torch.cuda.utilization(device),
                        }
                    )

            for loss_key in avg_loss_dict:
                avg_loss_dict[loss_key] = avg_loss_dict[loss_key] / len(dataloader)

            loss_dict_per_dataset[data_name] = avg_loss_dict
    return loss_dict_per_dataset
