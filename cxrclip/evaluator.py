import logging
from collections import Counter
from typing import Dict, List, Union

import numpy as np
import torch
from scipy.special import softmax
from sklearn import metrics
from tqdm import tqdm

from cxrclip.data import DataModule
from cxrclip.model import build_model
from cxrclip.prompt import constants
from cxrclip.prompt.prompts import generate_chexpert_class_prompts

log = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config: Dict, ckpt_paths):
        super().__init__()

        assert "test" in config and "checkpoint" in config["test"], "Evaluation needs model checkpoint."

        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # load ckpt config
        ckpt = torch.load(ckpt_paths[0], map_location="cpu")
        self.ckpt_config = ckpt["config"]
        print(self.ckpt_config)

        # load dataset
        data_config = {"test": self.config["data_test"]}

        if self.ckpt_config["model"]["image_encoder"]["name"] == "resnet":
            for _split in data_config:
                for _dataset in data_config[_split]:
                    data_config[_split][_dataset]["normalize"] = "imagenet"

        self.datamodule = DataModule(
            data_config=data_config,
            dataloader_config=self.config["dataloader"],
            tokenizer_config=self.ckpt_config["tokenizer"] if "tokenizer" in self.ckpt_config else None,
            transform_config=self.config["transform"] if "transform" in self.config else self.ckpt_config["transform"],
        )

        self.test_dataloader_dict = self.datamodule.test_dataloader()
        assert len(self.test_dataloader_dict) > 0

        # load model
        self.model = build_model(
            model_config=self.ckpt_config["model"], loss_config=self.ckpt_config["loss"], tokenizer=self.datamodule.tokenizer
        )
        self.model = self.model.to(self.device)

    def evaluate_clip(self, checkpoint, test_dataset_name):
        log.info(f"Load model {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(ckpt["model"], strict=False)
        self.model.eval()

        dataloader = self.test_dataloader_dict[test_dataset_name]

        image_embeddings = []
        text_embeddings = []
        texts = []
        label_names = []
        label_indices = []
        for batch in tqdm(dataloader):
            img_emb = self.encode_image(batch["images"])
            image_embeddings.append(img_emb)
            if "texts" in batch:
                texts += batch["texts"]
            if "text_tokens" in batch:
                text_emb = self.encode_text(batch["text_tokens"])
                text_embeddings.append(text_emb)
            if "label_names" in batch:
                label_names.extend(batch["label_names"])
            if "label_indices" in batch:
                label_indices.extend(batch["label_indices"].numpy())

        image_embeddings = np.concatenate(image_embeddings, axis=0)
        if len(text_embeddings) > 0:
            text_embeddings = np.concatenate(text_embeddings, axis=0)
        if hasattr(constants, test_dataset_name.upper()):
            class_list = getattr(constants, test_dataset_name.upper())

        results = {}
        if test_dataset_name in {"chexpert5x200"}:
            results["zeroshot_gloria"] = self.zeroshot_gloria(image_embeddings, label_names, class_list, 1000000, "mean")

        if test_dataset_name in {"siim_pneumothorax", "vindr_cxr"}:
            results["zeroshot_binary"] = self.zeroshot_binary(image_embeddings, label_names, class_list)

        if test_dataset_name in {"rsna_pneumonia"}:
            results["zeroshot_binary"] = self.zeroshot_binary_BioVIL(image_embeddings, label_names, class_list)

        if test_dataset_name in {"chexpert5x200", "mimic_cxr", "openi"}:
            results["retrieval_i2t"] = retrieval_image_text(image_embeddings, text_embeddings, texts)
        return results

    def evaluate_classifier(self, checkpoint, test_dataset_name):
        log.info(f"Load model {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(ckpt["model"], strict=False)
        self.model.eval()

        dataloader = self.test_dataloader_dict[test_dataset_name]

        preds = []
        labels = []
        for batch in tqdm(dataloader):
            with torch.no_grad():
                out = self.model(batch, device=self.device)
                preds.append(torch.sigmoid(out["cls_pred"]).detach().cpu().numpy())
                labels.append(batch["labels"].numpy())
        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        class_list = getattr(constants, test_dataset_name.upper())

        results = {}
        if test_dataset_name in {"chexpert5x200", "rsna_pneumonia", "siim_pneumothorax", "vindr_cxr"}:
            results["multilabel_classification"] = multilabel_classification(preds, labels, class_list)
        if test_dataset_name in {"chexpert5x200"}:
            results["multiclass_classification"] = multiclass_classification(preds, labels, class_list)
        return results

    def encode_image(self, image: torch.Tensor):
        with torch.no_grad():
            img_emb = self.model.encode_image(image.to(self.device))
            img_emb = self.model.image_projection(img_emb) if self.model.projection else img_emb
            img_emb = img_emb / torch.norm(img_emb, dim=1, keepdim=True)
        return img_emb.detach().cpu().numpy()

    def encode_text(self, text_token: Union[str, List[str], Dict, torch.Tensor]):
        if isinstance(text_token, str) or isinstance(text_token, list):
            text_token = self.datamodule.tokenizer(
                text_token, padding="longest", truncation=True, return_tensors="pt", max_length=self.ckpt_config["base"]["text_max_length"]
            )

        with torch.no_grad():
            text_emb = self.model.encode_text(text_token.to(self.device))
            text_emb = self.model.text_projection(text_emb) if self.model.projection else text_emb
            text_emb = text_emb / torch.norm(text_emb, dim=1, keepdim=True)
        return text_emb.detach().cpu().numpy()

    def zeroshot_gloria(
        self,
        image_embeddings: np.ndarray,
        label_names: list,
        class_list: list,
        num_prompt: Union[int, List[int]] = 1000000,  # [5, 10, 10000]
        label_select_strategy: Union[str, List[str]] = "mean",
    ):  # ["mean", "max"]
        log.info("evaluate zero-shot classification")
        if not isinstance(num_prompt, list):
            num_prompt = [num_prompt]
        if not isinstance(label_select_strategy, list):
            label_select_strategy = [label_select_strategy]

        # Atelectasis 210
        # Cardiomegaly 15
        # Consolidation 192
        # Edema 18
        # Pleural Effusion 54
        prompt_generator_fn = generate_chexpert_class_prompts
        prompt_dict = prompt_generator_fn(n=1000000)  # get all prompts
        similarities_all = {}
        for k, v in prompt_dict.items():
            text_embeddings = self.encode_text(v)
            similarities = metrics.pairwise.cosine_similarity(image_embeddings, text_embeddings)
            similarities_all[k] = similarities

        class_dict = {name: {"total_num": 0, "correct_num": 0} for name in class_list}
        for class_name, total_num in Counter(label_names).items():
            class_dict[class_name]["total_num"] = total_num

        result_zeroshot = {}
        for n_prom in num_prompt:
            for label_pooling in label_select_strategy:
                for idx, label in enumerate(label_names):
                    similarities = {k: np.random.permutation(v[idx])[:n_prom] for k, v in similarities_all.items()}

                    if label_pooling == "mean":
                        similarities = {k: np.mean(v) for k, v in similarities.items()}
                    elif label_pooling == "max":
                        similarities = {k: np.max(v) for k, v in similarities.items()}
                    else:
                        raise ValueError("Unknown label_select_strategy, got %s" % label_pooling)

                    if label == max(similarities, key=similarities.get):
                        class_dict[label]["correct_num"] += 1

                total_num = len(label_names)
                correct_num = sum([v["correct_num"] for v in class_dict.values()])

                result = {k: v["correct_num"] / v["total_num"] for k, v in class_dict.items()}
                result["Accuracy(Macro)"] = np.mean(list(result.values()))
                result["Accuracy(Micro)"] = correct_num / total_num  # same with macro due to same total_num
                result_zeroshot[f"{label_pooling}/{n_prom}"] = result

                s = " / ".join([f"{c}: {v:.3f}" for c, v in result.items()])
                log.info(s)

        return result_zeroshot

    def zeroshot_binary(self, image_embeddings: np.ndarray, label_names: list, class_list: list):
        log.info("evaluate zero-shot binary classification")
        if type(label_names[0]) is not list:
            label_names = [[label] for label in label_names]

        result = {}
        for class_name in class_list:
            prompts = ["No " + class_name, class_name]
            text_embeddings = self.encode_text(prompts)
            similarities = metrics.pairwise.cosine_similarity(image_embeddings, text_embeddings)
            similarities = softmax(similarities, axis=1)

            y_true = [1 if class_name in label else 0 for label in label_names]

            result[class_name] = {}
            fpr, tpr, thresholds = metrics.roc_curve(y_true, similarities[:, 1])
            result[class_name]["AUROC"] = metrics.auc(fpr, tpr)
            result[class_name]["Accuracy"] = metrics.accuracy_score(y_true, np.argmax(similarities, axis=1))
            result[class_name]["F1"] = metrics.f1_score(y_true, np.argmax(similarities, axis=1))

        return classification_score(result)

    def zeroshot_binary_BioVIL(self, image_embeddings: np.ndarray, label_names: list, class_list: list):
        log.info("evaluate zero-shot binary classification")
        if type(label_names[0]) is not list:
            label_names = [[label] for label in label_names]

        result = {}
        for class_name in class_list:
            prompts = ["No evidence of pneumonia", "Findings suggesting pneumonia."]
            text_embeddings = self.encode_text(prompts)
            similarities = metrics.pairwise.cosine_similarity(image_embeddings, text_embeddings)
            similarities = softmax(similarities, axis=1)

            y_true = [1 if class_name in label else 0 for label in label_names]

            result[class_name] = {}
            fpr, tpr, thresholds = metrics.roc_curve(y_true, similarities[:, 1])
            result[class_name]["AUROC"] = metrics.auc(fpr, tpr)
            result[class_name]["Accuracy"] = metrics.accuracy_score(y_true, np.argmax(similarities, axis=1))
            result[class_name]["F1"] = metrics.f1_score(y_true, np.argmax(similarities, axis=1))

        return classification_score(result)


def multilabel_classification(preds: np.ndarray, labels: np.ndarray, class_list: list):
    log.info("evaluate multi-label classification")

    result = {}
    for idx, class_name in enumerate(class_list):
        result[class_name] = {}
        fpr, tpr, thresholds = metrics.roc_curve(labels[:, idx], preds[:, idx])
        result[class_name]["AUROC"] = metrics.auc(fpr, tpr)
        result[class_name]["Accuracy"] = metrics.accuracy_score(labels[:, idx], preds[:, idx] > 0.5)
        result[class_name]["F1"] = metrics.f1_score(labels[:, idx], preds[:, idx] > 0.5)

    return classification_score(result)


def classification_score(result: dict):
    auroc = np.mean([value["AUROC"] for value in result.values()])
    f1 = np.mean([value["F1"] for value in result.values()])
    acc = np.mean([value["Accuracy"] for value in result.values()])

    result["AUROC(Avg)"] = auroc
    result["F1(Avg)"] = f1
    result["Accuracy(Avg)"] = acc

    s = "\n".join(f"{k}: {v}" for k, v in result.items())
    log.info(s)

    return result


def multiclass_classification(preds: np.ndarray, labels: np.ndarray, class_list: list):
    log.info("evaluate multi-class classification")
    preds_args = np.argmax(preds, axis=1)

    class_dict = {class_name: {"total_num": 0, "correct_num": 0} for class_name in class_list}
    for idx, class_name in enumerate(class_list):
        class_dict[class_name]["total_num"] = labels[:, idx].sum()
        class_dict[class_name]["correct_num"] = (labels[:, idx] * (preds_args == idx)).sum()

    total_num = len(labels)
    correct_num = sum([v["correct_num"] for v in class_dict.values()])

    result = {k: v["correct_num"] / v["total_num"] for k, v in class_dict.items()}
    result["Accuracy(Macro)"] = np.mean(list(result.values()))
    result["Accuracy(Micro)"] = correct_num / total_num  # same with macro due to same total_num
    s = " / ".join([f"{c}: {v:.3f}" for c, v in result.items()])
    log.info(s)

    return result


def retrieval_image_text(image_embeddings: np.ndarray, text_embeddings: np.ndarray, text_list: list = []):
    log.info("evaluate image text retrieval")

    identical_text_set = []

    idx2label = {}
    identical_indexes = []
    for i, text in enumerate(text_list):
        if text not in identical_text_set:
            identical_text_set.append(text)
            identical_indexes.append(i)
            idx2label[i] = len(identical_text_set) - 1
        else:
            idx2label[i] = identical_text_set.index(text)

    identical_text_embedding = text_embeddings[identical_indexes]

    num_samples = image_embeddings.shape[0]
    n_text = len(identical_text_set)

    similarities = metrics.pairwise.cosine_similarity(image_embeddings, identical_text_embedding)  # n x m
    recall_dict = {1: 0, 5: 0, 10: 0}
    mean_rank = 0
    for idx in range(num_samples):
        label = idx2label[idx]
        similarity = similarities[idx]
        similarity_args = similarity.argsort()

        # rank of the paired text
        rank = n_text - np.argwhere(similarity_args == label).ravel()[0]
        mean_rank += rank

        for k in recall_dict:
            if rank <= k:
                recall_dict[k] += 1

    # results
    log.info(
        "\n".join([f"Recall@{k}: {v / num_samples:.3f}" for k, v in recall_dict.items()]) + f"\nmean rank: {mean_rank / num_samples:.3f}"
    )
    result = {}
    result.update({f"Recall@{k}": v / num_samples for k, v in recall_dict.items()})
    result.update({"MeanRank": mean_rank / num_samples})
    return result
