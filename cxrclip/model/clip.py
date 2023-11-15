import logging
from typing import Dict

import numpy as np
import torch
from torch import nn
from transformers.tokenization_utils import PreTrainedTokenizer

from .modules import load_image_encoder, load_projection_head, load_text_encoder

log = logging.getLogger(__name__)


class CXRClip(nn.Module):
    def __init__(self, model_config: Dict, all_loss_config: Dict, tokenizer: PreTrainedTokenizer = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.image_encoder = load_image_encoder(model_config["image_encoder"])
        self.text_encoder = load_text_encoder(model_config["text_encoder"], vocab_size=tokenizer.vocab_size)
        self.text_pooling = model_config["text_encoder"]["pooling"]

        self.model_config = model_config
        self.loss_config = {k: v for k, v in all_loss_config.items()}

        self.projection = "projection_head" in model_config

        if self.projection:
            self.image_projection = load_projection_head(
                embedding_dim=self.image_encoder.out_dim, config_projection_head=model_config["projection_head"]
            )
            self.text_projection = load_projection_head(
                embedding_dim=self.text_encoder.out_dim, config_projection_head=model_config["projection_head"]
            )
        else:
            assert (
                self.image_encoder.out_dim == self.text_encoder.out_dim
            ), "Without 'projection_head', embedding_dim of the image and text encoder must be the same."

        self.temperature = model_config["temperature"] if "temperature" in model_config else None
        if self.temperature:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature))
        else:
            self.logit_scale = torch.tensor(1, dtype=torch.float32)
            log.warning("[CXRCLIP] missing temperature scaling factor")

    def encode_image(self, image):
        image_features = self.image_encoder(image)

        if self.model_config["image_encoder"]["name"] == "resnet":
            return image_features
        else:
            # get [CLS] token for global representation (only for vision transformer)
            global_features = image_features[:, 0]
            return global_features

    def encode_text(self, text_tokens):
        text_features = self.text_encoder(text_tokens)

        if self.text_pooling == "eos":
            # take features from the eot embedding (eos_token is the highest number in each sequence)
            eos_token_indices = text_tokens["attention_mask"].sum(dim=-1) - 1
            text_features = text_features[torch.arange(text_features.shape[0]), eos_token_indices]
        elif self.text_pooling == "bos":
            text_features = text_features[:, 0]
        elif self.text_pooling == "mean":
            input_mask_expanded = text_tokens["attention_mask"].unsqueeze(axis=-1).expand(text_features.size()).float()
            text_features = torch.sum(text_features * input_mask_expanded, axis=1) / torch.clamp(input_mask_expanded.sum(axis=1), min=1e-9)
        else:
            raise NotImplementedError("Not supported pooling method : %s", self.text_pooling)

        return text_features

    def forward(self, batch, device=None):
        device = batch["images"].device if device is None else device
        # get image and text features
        image_features_g = self.encode_image(batch["images"].to(device))
        text_features_g = self.encode_text(batch["text_tokens"].to(device))

        image_embeddings = self.image_projection(image_features_g) if self.projection else image_features_g
        text_embeddings = self.text_projection(text_features_g) if self.projection else text_features_g

        # normalize features
        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        # labels
        labels = torch.arange(image_embeddings.shape[0], device=device)

        out = {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
            "labels": labels,
            "logit_scale": self.logit_scale.exp(),
        }

        text_features_g2 = self.encode_text(batch["text_tokens2"].to(device))
        text_embeddings2 = self.text_projection(text_features_g2) if self.projection else text_features_g
        text_embeddings2 = text_embeddings2 / text_embeddings2.norm(dim=1, keepdim=True)
        out["text_embeddings2"] = text_embeddings2

        image_view_encode = self.encode_image(batch["image_views"].to(device))
        image_view_embeddings = self.image_projection(image_view_encode) if self.projection else image_view_encode
        image_view_embeddings = image_view_embeddings / image_view_embeddings.norm(dim=1, keepdim=True)
        out["image_view_embeddings"] = image_view_embeddings

        return out
