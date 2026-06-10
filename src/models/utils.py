import json
from pathlib import Path
import yaml
import torch
import matplotlib.pyplot as plt


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_output_dir(cfg):
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    return out


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable, "non_trainable_parameters": total - trainable}


def save_history_plot(history, path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_iou"], label="Train IoU")
    plt.plot(history["val_iou"], label="Val IoU")
    plt.title("IoU Score")
    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
