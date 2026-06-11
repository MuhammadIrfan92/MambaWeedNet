# MambaWeedNet

Official implementation for crop and weed semantic segmentation experiments on WeedsGalore, PotatoWeed, and SorghumWeed.

## Download Datasets:
    Datasets used in this study can be downloaded form the following links: \
    [WeedsGalore]()\
    [Sorghum Weed]()\
    [Potato Weed]()


## Experimental Environment Setup:
    Use the [requirements.txt](https://github.com/MuhammadIrfan92/MambaWeedNet/blob/main/requirements.txt) to install required packages.

## Directory map

```text
MambaWeedNet/
├── configs/
│   ├── weedsgalore.yaml
│   ├── potatoweed.yaml
│   └── sorghumweed.yaml
├── src/
│   ├── datasets.py
│   ├── losses.py
│   ├── metrics.py
│   ├── seed.py
│   ├── utils.py
│   └── model/
│       ├── __init__.py
│       └── encoder.py
│       └── decoder.py
│       └── proposed_model.py
│       └── modules/
│           └── modules.py
│           └── submodules.py
├── train.py
├── evaluate.py
├── predict.py
├── scripts/
│   ├── train_all.sh
│   └── evaluate_all.sh
├── weights/
│   └── README.md
├── results/
│   └── README.md
├── notebooks/
│   ├── Experimentation_WeedsGalore.ipynb
│   ├── Experimentation_PotatoWeed.ipynb
│   └── Experimentation_SorghumWeed.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
conda create -n mambaweednet python=3.10 -y
conda activate mambaweednet
pip install -r requirements.txt
```

## Train

```bash
python train.py --config configs/weedsgalore.yaml
python train.py --config configs/potatoweed.yaml
python train.py --config configs/sorghumweed.yaml
```

Or run all:

```bash
bash scripts/train_all.sh
```

## Evaluate

```bash
python evaluate.py --config configs/weedsgalore.yaml --checkpoint results/weedsgalore/MambaWeed_Net_best.pth --save-masks
python evaluate.py --config configs/potatoweed.yaml --checkpoint results/potatoweed/MambaWeed_Net_best.pth --save-masks
python evaluate.py --config configs/sorghumweed.yaml --checkpoint results/sorghumweed/MambaWeed_Net_best.pth --save-masks
```

## Predict selected samples

```bash
python predict.py --config configs/potatoweed.yaml --checkpoint results/potatoweed/MambaWeed_Net_best.pth --indices 11
```

## Outputs

Each dataset writes outputs to its own folder under `results/`:

```text
results/<dataset>/
├── MambaWeed_Net_best.pth
├── MambaWeed_Net_last.pth
├── history.pkl
├── training_plot.jpg
├── training_summary.json
├── evaluation_results.json
└── predictions/
    ├── pred_masks.pkl
    └── true_masks.pkl
```

## Dataset paths

Dataset paths are stored in the YAML files under `configs/`. Update the paths according to your local machine before training or evaluation.
