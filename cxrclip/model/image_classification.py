import logging
import os
from typing import Dict, TypeVar

import torch
from torch import nn
from torch.nn import Module

from .modules import load_image_classifier, load_image_encoder

T = TypeVar("T", bound="Module")
log = logging.getLogger(__name__)


class CXRClassification(nn.Module):
    def __init__(self, model_config: Dict, model_type: str = "vit"):
        super().__init__()
        self.model_type = model_type
        self.model_config = model_config
        if model_config["load_backbone_weights"] is None:
            self.image_encoder = load_image_encoder(model_config["image_encoder"])
        else:
            log.info("    loading pre-trained image encoder for fine-tuning")
            if not os.path.isfile(model_config["load_backbone_weights"]):
                raise ValueError(f"Cannot find a weight file: {model_config['load_backbone_weights']}")
            ckpt = torch.load(model_config["load_backbone_weights"], map_location="cpu")
            print(ckpt["config"]["model"]["image_encoder"])
            self.image_encoder = load_image_encoder(ckpt["config"]["model"]["image_encoder"])
            image_encoder_weights = {}
            for k in ckpt["model"].keys():
                if k.startswith("image_encoder."):
                    image_encoder_weights[".".join(k.split(".")[1:])] = ckpt["model"][k]
            self.image_encoder.load_state_dict(image_encoder_weights, strict=True)

        if model_config["freeze_backbone_weights"]:
            log.info("    freezing image encoder to not be trained")
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        self.classifier = load_image_classifier(model_config["classifier"]["config"], self.image_encoder.out_dim)

    def encode_image(self, image):
        image_features = self.image_encoder(image)

        if self.model_config["image_encoder"]["name"] == "resnet":
            return image_features
        else:
            # get [CLS] token for global representation (only for vision transformer)
            global_features = image_features[:, 0]
            return global_features

    def train(self: T, mode: bool = True) -> T:
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")

        if mode:
            self.image_encoder.eval()
            self.classifier.train()
        else:
            self.image_encoder.eval()
            self.classifier.eval()

        return self

    def forward(self, batch, device=None):
        device = batch["images"].device if device is None else device

        # get image features and predict
        image_feature = self.encode_image(batch["images"].to(device))
        cls_pred = self.classifier(image_feature)

        out = {"cls_pred": cls_pred, "target_class": batch["labels"].to(device)}
        return out
