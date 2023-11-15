from typing import Dict

from torch import nn
from transformers.tokenization_utils import PreTrainedTokenizer

from .clip import CXRClip
from .image_classification import CXRClassification


def build_model(model_config: Dict, loss_config: Dict, tokenizer: PreTrainedTokenizer = None) -> nn.Module:
    if model_config["name"].lower() == "clip_custom":
        model = CXRClip(model_config, loss_config, tokenizer)
    elif model_config["name"].lower() == "finetune_classification":
        model_type = model_config["image_encoder"]["model_type"] if "model_type" in model_config["image_encoder"] else "vit"
        model = CXRClassification(model_config, model_type)
    else:
        raise KeyError(f"Not supported model: {model_config['name']}")
    return model
