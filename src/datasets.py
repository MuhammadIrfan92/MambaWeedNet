import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, image_mode="auto", mask_mode="pil"):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_mode = image_mode
        self.mask_mode = mask_mode
        self.image_files = sorted(os.listdir(self.image_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))
        assert len(self.image_files) == len(self.mask_files), "Image and mask counts are different."

    def __len__(self):
        return len(self.image_files)

    def _load_image(self, path):
        suffix = path.suffix.lower()
        if self.image_mode == "npy" or suffix == ".npy":
            return np.load(path, allow_pickle=True).astype(np.float32)
        return np.array(Image.open(path).convert("RGB")).astype(np.float32)

    def _load_mask(self, path):
        suffix = path.suffix.lower()
        if self.mask_mode == "npy" or suffix == ".npy":
            return np.load(path).astype(np.uint8)
        return np.array(Image.open(path).convert("L"), dtype=np.uint8)

    def __getitem__(self, idx):
        image = self._load_image(self.image_dir / self.image_files[idx])
        mask = self._load_mask(self.mask_dir / self.mask_files[idx])

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.long)
        return image, mask.long()


def build_transform(image_size, flip=None, l2_norm=False):
    ops = []
    if flip == "h":
        ops.append(A.HorizontalFlip(p=1.0))
    if flip == "v":
        ops.append(A.VerticalFlip(p=1.0))
    ops.append(A.Resize(height=image_size, width=image_size))
    ops.append(ToTensorV2(transpose_mask=True))
    return A.Compose(ops)


def build_train_val_datasets(cfg):
    train_t = build_transform(cfg["image_size"])
    train_h = build_transform(cfg["image_size"], flip="h")
    train_v = build_transform(cfg["image_size"], flip="v")
    val_t = build_transform(cfg["image_size"])

    image_mode = cfg.get("image_mode", "auto")
    mask_mode = cfg.get("mask_mode", "pil")
    train_o = SegmentationDataset(cfg["train_image_dir"], cfg["train_mask_dir"], train_t, image_mode, mask_mode)
    train_h = SegmentationDataset(cfg["train_image_dir"], cfg["train_mask_dir"], train_h, image_mode, mask_mode)
    train_v = SegmentationDataset(cfg["train_image_dir"], cfg["train_mask_dir"], train_v, image_mode, mask_mode)
    train_dataset = ConcatDataset([train_o, train_h, train_v]) if cfg.get("use_flip_augmentation", True) else train_o
    val_dataset = SegmentationDataset(cfg["val_image_dir"], cfg["val_mask_dir"], val_t, image_mode, mask_mode)
    return train_dataset, val_dataset


def build_test_dataset(cfg):
    test_t = build_transform(cfg.get("test_image_size", cfg["image_size"]), l2_norm=cfg.get("test_l2_norm", False))
    return SegmentationDataset(cfg["test_image_dir"], cfg["test_mask_dir"], test_t, cfg.get("test_image_mode", cfg.get("image_mode", "auto")), cfg.get("mask_mode", "pil"))
