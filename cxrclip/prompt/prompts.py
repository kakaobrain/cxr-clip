"""drawn from MedCLIP github: https://github.com/RyanWangZf/MedCLIP"""

import random

from . import constants


def generate_chexpert_class_prompts(n=None):
    """Generate text prompts for each CheXpert classification task
    Parameters
    ----------
    n:  int
        number of prompts per class
    Returns
    -------
    class prompts : dict
        dictionary of class to prompts
    """

    prompts = {}
    for k, v in constants.CHEXPERT_CLASS_PROMPTS.items():
        cls_prompts = []
        keys = list(v.keys())

        # severity
        for k0 in v[keys[0]]:
            # subtype
            for k1 in v[keys[1]]:
                # location
                for k2 in v[keys[2]]:
                    cls_prompts.append(f"{k0} {k1} {k2}".strip())

        # randomly sample n prompts for zero-shot classification
        # TODO: we shall make use all the candidate prompts for autoprompt tuning
        if n is not None and n < len(cls_prompts):
            prompts[k] = random.sample(cls_prompts, n)
        else:
            prompts[k] = cls_prompts
        # print(f'sample {len(prompts[k])} num of prompts for {k} from total {len(cls_prompts)}')
    return prompts


def generate_report_from_labels(labels, prompt_json, deterministic=False, num_negs=0, name="chexpert"):
    if name == "chexpert":
        positive, negative, uncertain = labels

    elif name == "chest14":
        positive = labels
        if num_negs:
            negative = random.sample(list(set(constants.CHEST14_TASKS) - set(positive)), k=num_negs)
            if "Effusion" in negative:
                negative = [neg.replace("Effusion", "Pleural Effusion") for neg in negative]
        else:
            negative = []
        uncertain = []

        if "Effusion" in positive:
            positive = [pos.replace("Effusion", "Pleural Effusion") for pos in positive]

    # validation loss control
    if deterministic:
        if not positive:
            positive = ["No Finding"]
        negative, uncertain = [], []

    report = []
    if prompt_json:
        for pos in positive:
            cand = prompt_json[pos]["pos"]
            sentence = cand[0] if deterministic else random.choice(cand)
            if len(sentence) > 0:
                report.append(sentence)

        for neg in negative:
            cand = prompt_json[neg]["neg"]
            sentence = cand[0] if deterministic else random.choice(cand)
            if len(sentence) > 0:
                report.append(sentence)

        for unc in uncertain:
            cand = prompt_json[unc]["unc"]
            sentence = cand[0] if deterministic else random.choice(cand)
            if len(sentence) > 0:
                report.append(sentence)

    if not deterministic:
        random.shuffle(report)

    report = " ".join(report)
    return report
