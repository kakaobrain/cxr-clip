from .image_classification import ImageClassificationDataset
from .imagetext import ImageTextDataset
from .imagetext_eval import ImageTextEvalDataset


def load_dataset(data_type: str, loss_config=None, transform_config=None, **kwargs):
    if data_type == "imagetext":
        dataset = ImageTextDataset(loss_config=loss_config, transform_config=transform_config, **kwargs)
    elif data_type == "image_classification":
        dataset = ImageClassificationDataset(transform_config=transform_config, **kwargs)
    elif data_type == "imagetext_eval":
        dataset = ImageTextEvalDataset(transform_config=transform_config, **kwargs)
    else:
        raise KeyError(f"Not supported data type: {data_type}")
    return dataset
