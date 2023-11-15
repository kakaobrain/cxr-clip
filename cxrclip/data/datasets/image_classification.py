import ast
from typing import Dict, List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from cxrclip.data.data_utils import load_transform, transform_image


class ImageClassificationDataset(Dataset):
    def __init__(
        self,
        name: str,
        data_path: str,
        split: str,
        data_frac: float = 1.0,
        transform_config: Dict = None,
        normalize: str = "huggingface",
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.split = split
        self.data_frac = data_frac
        self.normalize = normalize

        self.image_transforms = load_transform(split=split, transform_config=transform_config)
        self.df = pd.read_csv(data_path)
        if data_frac < 1.0:
            self.df = self.df.sample(frac=self.data_frac, random_state=1, ignore_index=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = self.df["image"][index]
        if image_path.startswith("["):
            image_path = ast.literal_eval(image_path)[0]
        image = Image.open(image_path).convert("RGB")
        image = transform_image(self.image_transforms, image, normalize=self.normalize)

        if "label" in self.df:
            label = self.df["label"][index]
            if type(label) is str:
                label = ast.literal_eval(label)
            else:
                label = [label]
            if self.name == "vindr_cxr":
                label.pop(-5)  # other lesion
                label.pop(-1)  # other disease
            label = torch.Tensor(label)
        else:
            raise AttributeError("Cannot read the column for label")

        if "class" in self.df:
            label_name = self.df["class"][index]
            if label_name.startswith("["):
                label_name = ast.literal_eval(label_name)
        else:
            raise AttributeError("Cannot read the column for label_name")

        return {"image": image, "label": label, "label_name": label_name}

    def collate_fn(self, instances: List):
        images = torch.stack([ins["image"] for ins in instances], dim=0)
        labels = torch.stack([ins["label"] for ins in instances], dim=0)
        label_names = list([ins["label_name"] for ins in instances])
        return {"images": images, "labels": labels, "label_names": label_names}
