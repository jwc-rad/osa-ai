# Hybrid CNN-Transformer for OSA diagnosis on PSG

This repository contains the source code of the following paper:

[A novel deep learning model for obstructive sleep apnea diagnosis: hybrid CNN-Transformer approach for radar-based detection of apnea-hypopnea events]()<br>
<i>UNDER REVIEW</i>

## Usage

### Dataset format

The base dataset directory `DATASET_DIR` contains label (e.g. Event) and input directories (e.g. Radar) with `{PATIENT_ID}.npy` files, and a dataset split file.

```
DATASET_DIR/
├── Event
│   ├── PSG_0001.npy
│   ├── PSG_0002.npy
│   ├── ...
├── Radar
│   ├── PSG_0001.npy
│   ├── PSG_0002.npy
│   ├── ...
├── split.json
```

A sample code for generating split file for 5-fold CV based on `sklearn.model_selection.StratifiedKFold` on OSA classes:

```python
# suppose you already have list of cases and OSA classes for dev and test sets

kf = StratifiedKFold(n_splits=5, random_state=12345, shuffle=True)

splits = {}

for i, (train_idx, valid_idx) in enumerate(kf.split(cases_trainvalid, osa_trainvalid)):
    splits[i] = {
        'train': [x for j, x in enumerate(cases_trainvalid) if j in train_idx],
        'valid': [x for j, x in enumerate(cases_trainvalid) if j in valid_idx],
        'test': cases_test,
    }

ds_json = {
    'split': splits,
}

with open(os.path.join(DATASET_DIR, 'split.json'), 'w') as f:
    json.dump(ds_json, f)

```

### Run training

```
python train.py --multirun experiment=psgradar_paper data.dataset.cv_fold=0,1,2,3,4 data.dataset.image_totlen=4800 model.inferer.overlap=0.95
```

For different options, look at [`config/experiment/psgradar_paper.yaml`](config/experiment/psgradar_paper.yaml) and [`config/data/dataset/2311_clsHA_2ch_v0.yaml`](config/data/dataset/2311_clsHA_2ch_v0.yaml) for details.

### Inference

Refer to [`inference.ipynb`](inference.ipynb).

## Citation

If you find this repo useful for your work, please consider citing:

If you find our [preliminary study](https://www.mdpi.com/1424-8220/22/19/7177) useful, please also consider citing:

    @Article{s22197177,
        AUTHOR = {Choi, Jae Won and Kim, Dong Hyun and Koo, Dae Lim and Park, Yangmi and Nam, Hyunwoo and Lee, Ji Hyun and Kim, Hyo Jin and Hong, Seung-No and Jang, Gwangsoo and Lim, Sungmook and Kim, Baekhyun},
        TITLE = {Automated Detection of Sleep Apnea-Hypopnea Events Based on 60 GHz Frequency-Modulated Continuous-Wave Radar Using Convolutional Recurrent Neural Networks: A Preliminary Report of a Prospective Cohort Study},
        JOURNAL = {Sensors},
        VOLUME = {22},
        YEAR = {2022},
        NUMBER = {19},
        ARTICLE-NUMBER = {7177},
        URL = {https://www.mdpi.com/1424-8220/22/19/7177},
        PubMedID = {36236274},
        ISSN = {1424-8220},
        DOI = {10.3390/s22197177}
    }
