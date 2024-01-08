[![KakaoBrain](https://img.shields.io/badge/kakao-brain-ffcd00.svg)](http://kakaobrain.com/)

# CXR-CLIP
This is an official Pytorch Implementation of **"CXR-CLIP: Toward Large Scale Chest X-ray Language-Image Pre-training"** [[arxiv]](https://arxiv.org/abs/2310.13292)

## Environment setup
We have experimented the implementation on the following enviornment.
- Pytorch 1.12
- CUDA 11
```bash
pip install -r requirements.txt
```

## Prepare dataset
Datasets we used are as follows:

|           Dataset |                                                                            Download |              Comment |
|:-----------------:|:-----------------------------------------------------------------------------------:|----------------------|
| MIMIC-CXR         | [Link](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)                          | official split       |
| CheXpert          | [Link](https://stanfordmlgroup.github.io/competitions/chexpert/)                    | official split for train and val, and `chexpert_5x200` from [GLoRIA](https://stanfordmedicine.app.box.com/s/j5h7q99f3pfi7enc0dom73m4nsm6yzvh) for test |
| ChestX-ray14      | [Link](https://nihcc.app.box.com/v/ChestXray-NIHCC)                                 | not used for test    |
| VinDr-CXR         | [Link](https://physionet.org/content/vindr-cxr/1.0.0/)                              | official split for test, and random split for train and val |
| RSNA-Pneumonia    | [Link](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data) | same split as [GLoRIA](https://github.com/marshuang80/gloria/blob/416466af1036294301a872e4da169fefc137a192/gloria/datasets/preprocess_datasets.py#L49-L50) |
| SIIM-Pneumothorax | [Link](https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation/data) | same split as [GLoRIA](https://github.com/marshuang80/gloria/blob/416466af1036294301a872e4da169fefc137a192/gloria/datasets/preprocess_datasets.py#L90-L91) |
| OpenI | [Link](https://openi.nlm.nih.gov/faq#collection) | all frontal images are used for evaluation |

For more details, please refer to [data preparation](datasets/README.md).

## Pre-trained model checkpoint
We trained Resnet50 and SwinTiny models with three dataset compositions.  
MIMIC-CXR (**M**), MIMIC-CXR + CheXpert (**M,C**), MIMIC-CXR + CheXpert + ChestX-ray14 (**M,C,C14**)

| model / dataset |  M  | M,C | M,C,C14 | 
|---------------|--------------------|------------------------|-|
| ResNet50      |  [Link](https://twg.kakaocdn.net/brainrepo/models/cxr-clip/f982386ef38aa3ecd6ce1f8f928e8aa8/r50_m.tar)   |   [Link](https://twg.kakaocdn.net/brainrepo/models/cxr-clip/f7ebbe4ad815868905d0820dbbde3662/r50_mc.tar)  | [Link](https://twg.kakaocdn.net/brainrepo/models/cxr-clip/de4b5e4ae2047c1fb7960ddcd8d861cb/r50_mcc.tar) |
| SwinTiny      |  [Link](https://twg.kakaocdn.net/brainrepo/models/cxr-clip/a21ef120894e072ae77434daf6b98b72/swint_m.tar)   |   [Link](https://twg.kakaocdn.net/brainrepo/models/cxr-clip/97cbbdfb347d22ea44e95a70c7b0520a/swint_mc.tar)   | [Link](https://twg.kakaocdn.net/brainrepo/models/cxr-clip/a25ce760064591c899f67c97ed7790de/swint_mcc.tar) |

## Pre-Train model
### command line
* single gpu
    ```bash
    python train.py {--config-name default_config}
    ```
* multi gpu
    ```bash
    torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=45678 train.py {--config-name default_config}
    ```

## Evaluation
### Zero-shot Evaluation
* Zero-shot classification
  * perform zero-shot and image-text retrieval evaluation
    on ( vindr_cxr, rsna_pneumonia, siim_pneumothorax, chexpert5x200, mimic_cxr, openi )
  ```bash
  python evaluate_clip.py test.checkpoint=${CKPT_PATH/model-best.tar}
  ```

### Fine-tuned Classifier (linear probing)
* on rsna_pneumonia
```bash
# train
python finetune.py --config-name finetune_10 hydra.run.dir=${SAVE_DIR} data_train=rsna_pneumonia data_valid=rsna_pneumonia model.load_backbone_weights=${CKPT_PATH/model-best.tar} # 10%
python finetune.py hydra.run.dir=${SAVE_DIR} data_train=rsna_pneumonia data_valid=rsna_pneumonia model.load_backbone_weights=${CKPT_PATH/model-best.tar} # 100%
# evaluate
python evaluate_finetune.py data_test=rsna_pneumonia test.checkpoint=${FINETUNED_CKPT_PATH/model-best.tar}
```
* on siim_pneumothorax
```bash
# train
python finetune.py --config-name finetune_10 hydra.run.dir=${SAVE_DIR} data_train=siim_pneumothorax data_valid=siim_pneumothorax model.load_backbone_weights=${CKPT_PATH/model-best.tar} # 10%
python finetune.py hydra.run.dir=${SAVE_DIR} data_train=siim_pneumothorax data_valid=siim_pneumothorax model.load_backbone_weights=${CKPT_PATH/model-best.tar} # 100%
# evaluate
python evaluate_finetune.py data_test=siim_pneumothorax test.checkpoint=${FINETUNED_CKPT_PATH/model-best.tar}
```
* on vindr_cxr
```bash
# train
python finetune.py --config-name finetune_10 hydra.run.dir=${SAVE_DIR} data_train=vindr_cxr data_valid=vindr_cxr model.load_backbone_weights=${CKPT_PATH/model-best.tar} # 10%
python finetune.py hydra.run.dir=${SAVE_DIR} data_train=vindr_cxr data_valid=vindr_cxr model.load_backbone_weights=${CKPT_PATH/model-best.tar} # 100%
# evaluate
python evaluate_finetune.py data_test=vindr_cxr test.checkpoint=${FINETUNED_CKPT_PATH/model-best.tar}
```
## Citation
```
@incollection{You_2023,
	doi = {10.1007/978-3-031-43895-0_10},
	url = {https://doi.org/10.1007%2F978-3-031-43895-0_10},
	year = 2023,
	publisher = {Springer Nature Switzerland},
	pages = {101--111},
	author = {Kihyun You and Jawook Gu and Jiyeon Ham and Beomhee Park and Jiho Kim and Eun K. Hong and Woonhyuk Baek and Byungseok Roh},
	title="CXR-CLIP: Toward Large Scale Chest X-ray Language-Image Pre-training",
	booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
}
```
## License
CXR-CLIP: Toward Large Scale Chest X-ray Language-Image Pre-training © 2023 is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1)

## Contact for Issues
Kihyun You, [ukihyun@kakaobrain.com](ukihyun@kakaobrain.com)  
Jawook Gu, [jawook.gu@kakaobrain.com](jawook.gu@kakaobrain.com)
