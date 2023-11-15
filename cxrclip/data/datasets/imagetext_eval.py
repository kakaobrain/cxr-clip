import ast
from typing import Dict, List

import pandas as pd
from PIL import Image
from torch.utils.data import default_collate
from torch.utils.data.dataset import Dataset

from cxrclip.data.data_utils import load_transform, transform_image
from cxrclip.prompt.constants import CHEXPERT_CLASS_PROMPTS


class ImageTextEvalDataset(Dataset):
    def __init__(
        self,
        name: str,
        data_path: str,
        split: str,
        data_frac: float = 1.0,
        tokenizer=None,
        text_max_length: int = 256,
        transform_config: Dict = None,
        normalize: str = "huggingface",
        **kwargs
    ):
        super().__init__()
        self.name = name
        self.split = split
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.data_frac = data_frac
        self.normalize = normalize

        if self.name == "chexpert5x200":
            self.label_list = list(CHEXPERT_CLASS_PROMPTS.keys())
        else:
            self.label_list = []

        self.idx2label = {idx: self.label_list[idx] for idx in range(len(self.label_list))}
        self.label2idx = {v: k for k, v in self.idx2label.items()}

        self.image_transforms = load_transform(split="test", transform_config=transform_config)
        self.df = pd.read_csv(data_path)
        if data_frac < 1.0:
            self.df = self.df.sample(frac=self.data_frac, random_state=1, ignore_index=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.name == "chexpert5x200":
            image_path = self.df["Path"][index]
        else:
            image_path = self.df["image"][index]

        if image_path.startswith("["):
            image_path = ast.literal_eval(image_path)[0]  # not random sampling
        image = Image.open(image_path).convert("RGB")
        image = transform_image(self.image_transforms, image, normalize=self.normalize)

        if self.name == "chexpert5x200":
            text = self.df["Report Impression"][index]
        else:
            text = self.df["text"][index]

        sample = {"images": image, "text": text}

        if self.name in {"chexpert5x200"}:
            for label_candidate in self.label_list:
                if self.df[label_candidate][index] == 1.0:
                    label = label_candidate
            label_idx = self.label2idx[label]
            sample["label_names"] = label
            sample["label_indices"] = label_idx

        return sample

    def collate_fn(self, instances: List):
        collate = default_collate(instances)
        text_tokens = self.tokenizer(
            collate["text"], padding="longest", truncation=True, return_tensors="pt", max_length=self.text_max_length
        )
        collate["text_tokens"] = text_tokens
        texts = list([ins["text"] for ins in instances])
        collate["texts"] = texts

        return collate
