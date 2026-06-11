import pickle
import torch
from torch.utils.data import DataLoader
import random

from src.datasets import build_test_dataset
from src.metrics import evaluate_model
from src.models import MambaWeed_Net
from src.seed import set_seed
from src.utils import load_yaml, make_output_dir, save_json, count_parameters


def main(config_path, checkpoint=None, save_masks=False):
    cfg = load_yaml(config_path)
    seed = random.randint(0, 2**32 - 1)
    print(f"Using seed: {seed}")
    set_seed(seed)
    out_dir = make_output_dir(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = build_test_dataset(cfg)
    test_loader = DataLoader(test_dataset, batch_size=cfg.get("test_batch_size", cfg["batch_size"]), shuffle=False, num_workers=cfg.get("num_workers", 0))

    model = MambaWeed_Net(in_channels=cfg["num_channels"], out_channels=cfg["num_classes"], kernels=cfg["kernels"])
    ckpt = checkpoint or str(out_dir / "MambaWeedNet.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device)

    results = evaluate_model(model, test_loader, device, cfg["num_classes"], return_masks=save_masks)
    pred_masks = results.pop("pred_masks", None)
    true_masks = results.pop("true_masks", None)
    results = {**results, **count_parameters(model), "checkpoint": ckpt, "test_image_size": cfg.get("test_image_size", cfg["image_size"])}
    save_json(results, out_dir / "evaluation_results.json")

    if save_masks:
        with open(out_dir / "predictions" / "pred_masks.pkl", "wb") as f:
            pickle.dump(pred_masks, f)
        with open(out_dir / "predictions" / "true_masks.pkl", "wb") as f:
            pickle.dump(true_masks, f)
    print(results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save-masks", action="store_true")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.save_masks)
