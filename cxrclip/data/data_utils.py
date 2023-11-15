import os
from typing import Dict, Union

import albumentations
import albumentations.pytorch.transforms
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer


def load_tokenizer(source, pretrained_model_name_or_path, cache_dir, **kwargs):
    if source == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=os.path.exists(os.path.join(cache_dir, f'models--{pretrained_model_name_or_path.replace("/", "--")}')),
            **kwargs,
        )
        if tokenizer.bos_token_id is None:
            tokenizer.bos_token_id = tokenizer.cls_token_id
    else:
        raise KeyError(f"Not supported tokenizer source: {source}")

    return tokenizer


def load_transform(split: str = "train", transform_config: Dict = None):
    assert split in {"train", "valid", "test", "aug"}

    config = []
    if transform_config:
        if split in transform_config:
            config = transform_config[split]
    image_transforms = []

    for name in config:
        if hasattr(transforms, name):
            tr_ = getattr(transforms, name)
        else:
            tr_ = getattr(albumentations, name)
        tr = tr_(**config[name])
        image_transforms.append(tr)

    return image_transforms


def transform_image(image_transforms, image: Union[Image.Image, np.ndarray], normalize="huggingface"):
    for tr in image_transforms:
        if isinstance(tr, albumentations.BasicTransform):
            image = np.array(image) if not isinstance(image, np.ndarray) else image
            image = tr(image=image)["image"]
        else:
            image = transforms.ToPILImage()(image) if not isinstance(image, Image.Image) else image
            image = tr(image)

    if normalize == "huggingface":
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)(image)

    elif normalize == "imagenet":
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

    else:
        raise KeyError(f"Not supported Normalize: {normalize}")

    return image
