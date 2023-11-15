import ast
import json
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from nltk import tokenize
from PIL import Image
from torch.utils.data.dataset import Dataset

from cxrclip.data.data_utils import load_transform, transform_image
from cxrclip.prompt.prompts import generate_report_from_labels


class ImageTextDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        name: str,
        data_path: str,
        split: str,
        text_max_length: int = 256,
        text_sampling: str = "random",
        loss_config: Dict = None,
        transform_config: Dict = None,
        prompt_from_json: bool = False,
        data_frac: float = 1.0,
        num_negs: int = 0,
        normalize: str = "huggingface",
        **kwargs
    ):
        super().__init__()
        self.name = name
        self.split = split
        self.text_max_length = text_max_length
        self.text_sampling = text_sampling
        self.data_frac = data_frac
        self.num_negs = num_negs
        self.normalize = normalize

        self.tokenizer = tokenizer
        self.image_transforms = load_transform(split=split, transform_config=transform_config)

        if prompt_from_json:
            with open("datasets/train_prompts_all.json") as f:
                self.prompt_json = json.load(f)
        else:
            self.prompt_json = False

        assert data_path.endswith(".csv")

        self.df = pd.read_csv(data_path)
        if data_frac < 1.0:
            self.df = self.df.sample(frac=self.data_frac, random_state=1, ignore_index=True)

        self.loss_config = {k: v for k, v in loss_config.items()}

        self.image_view_aug = True
        self.image_aug_other_image = True
        self.image_aug_transforms = self.image_transforms
        self.has_backtranslated = hasattr(self.df, "text_augment")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if hasattr(self.df, "AP"):  # AP / PA / Lateral
            try:
                view_list = ast.literal_eval(self.df["view"][index])
            except Exception:
                view_list = [self.df["view"][index]]

            if len(view_list) > 2:
                view_list = np.random.choice(view_list, size=2, replace=False)
                image_path_list = []
                for view in view_list:
                    try:
                        image_path_list = ast.literal_eval(self.df[view][index])
                    except Exception:
                        image_path_list = [self.df[view][index]]

                    image_path = np.random.choice(image_path_list, size=1)[0]
                    image_path_list.append(image_path)

            else:
                if len(view_list) == 1:
                    tag = view_list[0]
                else:
                    tag = "image"

                try:
                    image_path_list = ast.literal_eval(self.df[tag][index])
                except Exception:
                    image_path_list = [self.df[tag][index]]

                if self.split == "train":
                    if self.image_aug_other_image and len(image_path_list) > 1:
                        image_path_list = np.random.choice(image_path_list, size=2, replace=False)
                    else:
                        image_path_list = np.random.choice(image_path_list, size=1)
        else:
            try:
                image_path_list = ast.literal_eval(self.df["image"][index])
            except Exception:
                image_path_list = [self.df["image"][index]]

        image_original = Image.open(image_path_list[0]).convert("RGB")
        image = transform_image(self.image_transforms, image_original, normalize=self.normalize)

        if self.image_view_aug:
            if len(image_path_list) > 1:
                image_original = Image.open(image_path_list[1]).convert("RGB")

            image_view = transform_image(self.image_aug_transforms, image_original, normalize=self.normalize)

        # Get Text or Prompt
        if hasattr(self.df, "text"):
            try:
                text_list = ast.literal_eval(self.df["text"][index])
            except Exception:
                text_list = self.df["text"][index]

            if self.has_backtranslated:
                try:
                    text_aug_list = ast.literal_eval(self.df["text_augment"][index])
                except Exception:
                    text_aug_list = self.df["text_augment"][index]

            if len(text_list) >= 2:
                indexes = np.random.randint(len(text_list), size=2)  # Multiple section
                text = text_aug_list[indexes[0]] if random.random() < 0.5 and self.has_backtranslated else text_list[indexes[0]]
                text2 = text_aug_list[indexes[1]] if random.random() < 0.5 and self.has_backtranslated else text_list[indexes[1]]

            else:
                if random.random() < 0.5:
                    text = text_list[0]
                    text2 = text_aug_list[0] if self.has_backtranslated else text_list[0]
                else:
                    text = text_aug_list[0] if self.has_backtranslated else text_list[0]
                    text2 = text_list[0]

            if self.split == "train":  # Text shuffle augment
                for _text in [text, text2]:
                    _text_list = tokenize.sent_tokenize(_text, language="english")
                    random.shuffle(_text_list)
                    _text = " ".join(_text_list)

        # Get Two Prompts per sample.
        elif hasattr(self.df, "text_label"):
            labels = ast.literal_eval(self.df["text_label"][index])
            text = generate_report_from_labels(
                labels, self.prompt_json, deterministic=(self.split != "train"), num_negs=self.num_negs, name=self.name
            )
            text2 = generate_report_from_labels(
                labels, self.prompt_json, deterministic=(self.split != "train"), num_negs=self.num_negs, name=self.name
            )
        else:
            raise AttributeError("There is no report column in DataFrame.")

        out = {"image": image, "image_view": image_view, "text": text, "text2": text2}

        return out

    def collate_fn(self, instances: List):
        images = torch.stack([ins["image"] for ins in instances], dim=0)
        texts = list([ins["text"] for ins in instances])
        text_tokens = self.tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt", max_length=self.text_max_length)

        texts2 = list([ins["text2"] for ins in instances])
        text_tokens2 = self.tokenizer(texts2, padding="max_length", truncation=True, return_tensors="pt", max_length=self.text_max_length)
        image_views = torch.stack([ins["image_view"] for ins in instances], dim=0)

        batch = {
            "images": images,
            "image_views": image_views,
            "texts": texts,
            "texts2": texts2,
            "text_tokens": text_tokens,
            "text_tokens2": text_tokens2,
        }

        return batch
