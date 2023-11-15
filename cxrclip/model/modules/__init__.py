import os
from typing import Dict

from .image_classifier import LinearClassifier
from .image_encoder import HuggingfaceImageEncoder, ResNet50
from .projection import LinearProjectionHead, MLPProjectionHead
from .text_encoder import HuggingfaceTextEncoder


def load_image_encoder(config_image_encoder: Dict):
    if config_image_encoder["source"].lower() == "huggingface":
        cache_dir = config_image_encoder["cache_dir"] if "cache_dir" in config_image_encoder else "~/.cache/huggingface/hub"
        gradient_checkpointing = (
            config_image_encoder["gradient_checkpointing"] if "gradient_checkpointing" in config_image_encoder else False
        )
        model_type = config_image_encoder["model_type"] if "model_type" in config_image_encoder else "vit"
        _image_encoder = HuggingfaceImageEncoder(
            name=config_image_encoder["name"],
            pretrained=config_image_encoder["pretrained"],
            gradient_checkpointing=gradient_checkpointing,
            cache_dir=cache_dir,
            model_type=model_type,
            local_files_only=os.path.exists(os.path.join(cache_dir, f'models--{config_image_encoder["name"].replace("/", "--")}')),
        )
    elif config_image_encoder["name"] == "resnet":
        _image_encoder = ResNet50()

    else:
        raise KeyError(f"Not supported image encoder: {config_image_encoder}")
    return _image_encoder


def load_text_encoder(config_text_encoder: Dict, vocab_size: int):
    if config_text_encoder["source"].lower() == "huggingface":
        cache_dir = config_text_encoder["cache_dir"]
        gradient_checkpointing = config_text_encoder["gradient_checkpointing"]
        _text_encoder = HuggingfaceTextEncoder(
            name=config_text_encoder["name"],
            vocab_size=vocab_size,
            pretrained=config_text_encoder["pretrained"],
            gradient_checkpointing=gradient_checkpointing,
            cache_dir=cache_dir,
            local_files_only=os.path.exists(os.path.join(cache_dir, f'models--{config_text_encoder["name"].replace("/", "--")}')),
            trust_remote_code=config_text_encoder["trust_remote_code"],
        )
    else:
        raise KeyError(f"Not supported text encoder: {config_text_encoder}")
    return _text_encoder


def load_projection_head(embedding_dim: int, config_projection_head: Dict):
    if config_projection_head["name"].lower() == "mlp":
        projection_head = MLPProjectionHead(
            embedding_dim=embedding_dim, projection_dim=config_projection_head["proj_dim"], dropout=config_projection_head["dropout"]
        )
    elif config_projection_head["name"].lower() == "linear":
        projection_head = LinearProjectionHead(embedding_dim=embedding_dim, projection_dim=config_projection_head["proj_dim"])
    else:
        raise KeyError(f"Not supported text encoder: {config_projection_head}")
    return projection_head


def load_image_classifier(config_image_classifier: Dict, feature_dim: int):
    if config_image_classifier["name"].lower() == "linear":
        _image_classifier = LinearClassifier(feature_dim=feature_dim, num_class=config_image_classifier["n_class"])
    else:
        raise KeyError(f"Not supported image classifier: {config_image_classifier}")

    return _image_classifier
