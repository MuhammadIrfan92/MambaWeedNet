import pickle
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import random

from src.datasets import build_train_val_datasets
from src.losses import build_loss
from src.metrics import compute_iou_batch
from src.models import MambaWeed_Net
from src.seed import set_seed
from src.utils import load_yaml, make_output_dir, save_history_plot, save_json, count_parameters


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            self.early_stop = self.counter >= self.patience


def forward_logits(model, images):
    outputs = model(images)
    return outputs[0] if isinstance(outputs, tuple) else outputs


def main(config_path):
    cfg = load_yaml(config_path)
    seed = random.randint(0, 2**32 - 1)
    print(f"Using seed: {seed}")
    set_seed(seed)
    out_dir = make_output_dir(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset = build_train_val_datasets(cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 0))
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 0))

    model = MambaWeed_Net(in_channels=cfg["num_channels"], out_channels=cfg["num_classes"], kernels=cfg["kernels"]).to(device)
    criterion = build_loss(cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg.get("weight_decay", 0))
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.get("lr_tmax", max(1, cfg["epochs"] // 5)))
    early_stopper = EarlyStopping(patience=cfg.get("early_stopping_patience", max(10, (cfg["epochs"] // 5) * 2)))

    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}
    best_val_loss = float("inf")
    start_time = datetime.now()

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss, train_iou = 0.0, 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['epochs']} - Training"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = forward_logits(model, images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_iou += compute_iou_batch(torch.argmax(logits, dim=1).cpu(), masks.cpu(), cfg["num_classes"])

        model.eval()
        val_loss, val_iou = 0.0, 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{cfg['epochs']} - Validation"):
                images, masks = images.to(device), masks.to(device)
                logits = forward_logits(model, images)
                loss = criterion(logits, masks)
                val_loss += loss.item()
                val_iou += compute_iou_batch(torch.argmax(logits, dim=1).cpu(), masks.cpu(), cfg["num_classes"])

        train_loss /= len(train_loader); val_loss /= len(val_loader)
        train_iou /= len(train_loader); val_iou /= len(val_loader)
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou); history["val_iou"].append(val_iou)
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_iou={train_iou:.4f}, val_iou={val_iou:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "MambaWeed_Net_best.pth")
            print(f"Saved best model: {out_dir / 'MambaWeed_Net_best.pth'}")

        scheduler.step()
        early_stopper.step(val_loss)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    torch.save(model.state_dict(), out_dir / "MambaWeed_Net_last.pth")
    with open(out_dir / "history.pkl", "wb") as f:
        pickle.dump(history, f)
    save_history_plot(history, out_dir / "training_plot.jpg")
    summary = {**cfg, **count_parameters(model), "training_time": str(datetime.now() - start_time), "best_val_loss": best_val_loss}
    save_json(summary, out_dir / "training_summary.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
