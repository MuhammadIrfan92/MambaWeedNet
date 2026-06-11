# MambaWeedNet

Official implementation for crop and weed semantic segmentation experiments on WeedsGalore, PotatoWeed, and SorghumWeed.

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
│   └── models/
│       ├── __init__.py
│       └── mambaweed_net.py
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

## Important model-code note

The scripts expect `MambaWeed_Net` in `src/models/mambaweed_net.py`.
Currently, that file imports your original `Utils.MambaLLM.MambaWeed_Net` if available. Before public release, copy the clean final model implementation into `src/models/mambaweed_net.py` so the repository runs without private notebook dependencies.

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
