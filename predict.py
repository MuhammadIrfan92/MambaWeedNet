import pickle
import torch
from torch.utils.data import DataLoader, Subset
from src.datasets import build_test_dataset
from src.model.proposed_model import MambaWeed_Net
from src.utils import load_yaml, make_output_dir


def main(config_path, checkpoint=None, indices=None):
    cfg = load_yaml(config_path)
    out_dir = make_output_dir(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_test_dataset(cfg)
    if indices:
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = MambaWeed_Net(in_channels=cfg["num_channels"], out_channels=cfg["num_classes"], kernels=cfg["kernels"])
    ckpt = checkpoint or str(out_dir / "MambaWeed_Net.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device).eval()

    preds, labels = [], []
    with torch.no_grad():
        for images, masks in loader:
            logits = model(images.to(device))
            if isinstance(logits, tuple):
                logits = logits[0]
            preds.append(torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu())
            labels.append(masks.cpu())
    pred_masks = torch.cat(preds, dim=0)
    true_masks = torch.cat(labels, dim=0)
    with open(out_dir / "predictions" / "selected_pred_masks.pkl", "wb") as f:
        pickle.dump(pred_masks, f)
    with open(out_dir / "predictions" / "selected_true_masks.pkl", "wb") as f:
        pickle.dump(true_masks, f)
    print(f"Saved predictions to {out_dir / 'predictions'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.indices)
