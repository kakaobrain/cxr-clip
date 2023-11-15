This folder contains examples of processed dataset and propsed prompt (train_prompts_all.json)
- We resized the image ensuring that the shorter side measures 512 pixels with the same aspect ratio

## Pre-training datasets

### MIMIC-CXR
1. Download **mimic-cxr-jpg** in [Physionet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).
2. Extract "findings" and "impression" sections as list of texts.
3. Split train/val/test set with **mimic-cxr-2.0.0-split.csv.gz**.
4. For utilizing multi-view, get view position with **mimic-cxr-2.0.0-metadata.csv.gz**.
5. Text Augmentation with `text_augmentation/back_translation.py`.

The final csv file contains following columns.

| index | image              | view          | AP                    | PA                    | Lateral                    | text                           | text_augment                      |
|-------|--------------------|---------------|-----------------------|-----------------------|----------------------------|--------------------------------|-----------------------------------|
| 0     | List of image_path | List of views | List of AP image_path | List of PA image_path | List of Lateral image_path | List of [findings, impression] | Result of backtranslation of text |


### CheXpert
1. Download the dataset in [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/).
2. Get **chexpert_5x200.csv** from [GLoRIA stroage](https://stanfordmedicine.app.box.com/s/j5h7q99f3pfi7enc0dom73m4nsm6yzvh).
3. Exclude the patients in **chexpert_5x200.csv** from training set as follows.
   <details><summary>click to expand</summary><div markdown="1">
   
   ```
   import os
   import pandas as pd
   
   WORK_DIR = "/path/to/load/chexpert"
   df_train = pd.read_csv(os.path.join(WORK_DIR, "CheXpert-v1.0", "train.csv"))
   df_test = pd.read_csv(os.path.join(WORK_DIR, "chexpert_5x200.csv"))
   
   df_train["patient_id"] = df_train["Path"].apply(lambda x: x.split("/")[-3])
   df_test["patient_id"] = df_test["Path"].apply(lambda x: x.split("/")[-3])
   print(f"# image : {len(df_train)}, # patient : {df_train['patient_id'].nunique()}")  # # image : 223414, # patient : 64540
   print(f"# image : {len(df_test)}, # patient : {df_test['patient_id'].nunique()}")  # # image : 1000, # patient : 966
   
   df_train = df_train[~df_train["patient_id"].isin(df_test["patient_id"].tolist())]
   print(f"# image : {len(df_train)}, # patient : {df_train['patient_id'].nunique()}")  # # image : 216478, # patient : 63574
   ```
   
   </div></details>
4. Get text_label as triplet [[positives], [negatives], [uncertain]].
5. For evalutating chexpert_5x200, change **Path** to fit your datapath.

The final csv file contains following columns.

| index | image              | view          | AP                    | PA                    | Lateral                    | text_label                                                 |
|-------|--------------------|---------------|-----------------------|-----------------------|----------------------------|------------------------------------------------------------|
| 0     | List of image_path | List of views | List of AP image_path | List of PA image_path | List of Lateral image_path | [[positive labels], [negative labels], [uncertain labels]] |

### Chest-Xray14
1. Download the dataset in [ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC).
2. Get image label with **Data_Entry_2017_v2020.csv**.
3. Split train/valid as follows.
   <details><summary>click to expand</summary><div markdown="1">
   
   ```
   import os
   import pandas as pd
   
   from sklearn.model_selection import train_test_split
   
   WORK_DIR = "/path/to/load/chest14"
   df = pd.read_csv(os.path.join(WORK_DIR, "Data_Entry_2017_v2020.csv"))
   df.set_index("Image Index", inplace=True)
   
   files_train = []
   with open(os.path.join(WORK_DIR, "train_val_list.txt"), "r") as file:
       for line in file.readlines():
           filename = line.replace("\n", "")
           files_train.append(filename)
   df_train = df.loc[files_train, :]
   df_train.reset_index(drop=False, inplace=True)
   
   unique_ids = df_train["Patient ID"].unique()
   train_ids, valid_ids = train_test_split(unique_ids, test_size=0.2, random_state=0)
   
   df_train = df_train.set_index("Patient ID")
   df_train, df_valid = df_train.loc[train_ids, :], df_train.loc[valid_ids, :]
   df_train.reset_index(drop=False, inplace=True)
   df_valid.reset_index(drop=False, inplace=True)
   
   df_train.to_csv(os.path.join(WORK_DIR, "chest14_train.csv"))
   df_valid.to_csv(os.path.join(WORK_DIR, "chest14_valid.csv"))
   ```
   
   </div></details>

The final csv file contains following columns.

| index | image              | text_label        |
|-------|--------------------|-------------------|
| 0     | List of image_path | [positive labels] |

## Downstream datasets

### VinDr-CXR
1. Download **VinDr-CXR** in [Physionet](https://physionet.org/content/vindr-cxr/1.0.0/).
2. dcm2png
   <details><summary>click to expand</summary><div markdown="1">
   
   ```
   import numpy as np
   
   import cv2
   import pydicom
   from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
   
   
   def resize_and_save(load_path, save_path):  # load_path=/path/to/load/*.dicom, save_path=/path/to/save/*.jpg
       ds = pydicom.dcmread(load_path, force=True)
       img = ds.pixel_array
       img = apply_modality_lut(img, ds)  # rescaleSlope & intercept
       img = apply_voi_lut(img, ds)  # windowing
       if hasattr(ds, "PhotometricInterpretation"):
           if ds.PhotometricInterpretation.lower().strip() == "monochrome1":
               img = img.max() - img  # invert
       
       h, w = img.shape
       ratio = 512 / min(h, w)
       target_size = (int(w * ratio), int(h * ratio))
       img = cv2.resize(img, target_size, cv2.INTER_LANCZOS4)
      
       # normalize
       img = (img - img.min()) / (img.max() - img.min()) * np.iinfo(np.uint8).max
       img = img.astype(np.uint8)
       cv2.imwrite(save_path, img)
   ```
   
   </div></details>
3. Get Label with **image_labels_{train/test}.csv**
4. Get valid set from training set
   <details><summary>click to expand</summary><div markdown="1">
   
   ```
   import os
   import pandas as pd
   
   from sklearn.model_selection import train_test_split
   
   WORK_DIR = "/path/to/load/vindr-cxr"
   
   df = pd.read_csv(os.path.join(WORK_DIR, "1.0.0", "annotations", "image_labels_train.csv"))
   
   df = df.groupby("image_id").agg(sum)
   df.loc[:, "Aortic enlargement":"No finding"] = (df.loc[:, "Aortic enlargement":"No finding"] > 0).astype(int)
   df.reset_index(drop=False, inplace=True)
   
   df_train, df_valid = train_test_split(df, test_size=0.2, random_state=0)
   
   df_train.to_csv(os.path.join(WORK_DIR, "vindr_train.csv"))
   df_valid.to_csv(os.path.join(WORK_DIR, "vindr_valid.csv"))
   ```
   
   </div></details>

The final csv file contains following columns.

| index | image      | label                        | class                      |
|-------|------------|------------------------------|----------------------------|
| 0     | image_path | List of label value (0 or 1) | List of positive classname |

### RSNA Pneumonia
1. Download RSNA dataset in [Kaggle](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data)
2. dcm2png - Details in Vindr-CXR
3. get label from **stage_2_train_labels.csv** 
4. get class from **stage_2_detailed_class_info.csv** if "class" is "Lung Opacity", class is "Pneumonia" else "Normal"
5. Split train/valid/test following [GLoRIA preprocess code](https://github.com/marshuang80/gloria/blob/416466af1036294301a872e4da169fefc137a192/gloria/datasets/preprocess_datasets.py#L49-L50).

The final csv file contains following columns.
| index | image      | label                | class               |
|-------|------------|----------------------|---------------------|
| 0     | image_path | label value (0 or 1) | {Pneumonia/ Normal} |

### SIIM Pneumothorax
1. Download SIIM dataset in [Kaggle](https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation/data).
2. dcm2png - Details in Vindr-CXR
3. with "EncodedPixels" in **stage_2_train.csv**, get label.
4. Split train/valid/test following [GLoRIA preprocess code](https://github.com/marshuang80/gloria/blob/416466af1036294301a872e4da169fefc137a192/gloria/datasets/preprocess_datasets.py#L90-L91).

| index | image      | label                | class               |
|-------|------------|----------------------|---------------------|
| 0     | image_path | label value (0 or 1) | {Pneumothorax/ No Pneumothorax} |

## Image-text evaluation dataset

### OpenI
1. Download dcm files and reports on [OpenI](https://openi.nlm.nih.gov/faq#collection)
2. dcm2png - Details in Vindr-CXR, We only use frontal images (3,955 image-text pairs)
3. From .xml file extract a corresponding report.
   <details><summary>click to expand</summary><div markdown="1">
   
   ```
   import xmltodict

   def extract_report_from_xml(load_path):  # load_path=/path/to/load/*.xml, return report
        with open(load_path) as fd:
            data = xmltodict.parse(fd.read())

        abstract = data["eCitation"]["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
        comparison, indication, finding, impression = abstract

        report = max(finding["#text"], impression["#text"])
        return report
   ```
   
   </div></details>

The final csv file contains following columns.

| index | image      | text                | 
|-------|------------|----------------------|
| 0     | image_path | max(finding, impression) | 