"""
Dataset loading and augmentation for face recognition training.

Expected layout (ImageFolder-style, one directory per identity):

    data_root/
        person_0001/
            img_001.jpg
            img_002.jpg
            ...
        person_0002/
            ...

This works for CASIA-WebFace, VGGFace2, MS1M, and the existing DigiFace-1M
subset on the user's Desktop. See `download_datasets.py` for fetching them.

Augmentations (applied at train time only) are tuned for the assistive-glasses
deployment scenario:
    - random horizontal flip                (pose mirror)
    - random affine: small rotation + shear (head tilt)
    - color jitter                          (lighting, white balance)
    - random Gaussian blur                  (motion / focus blur)
    - random JPEG / brightness changes
    - random erasing                        (partial occlusion)

ArcFace normalization: x = (x - 127.5) / 128 (range ~[-1, 1]).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

import config


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

# ArcFace mean/std in [0,1] coordinates (because ToTensor() divides by 255):
#   ((x*255 - 127.5) / 128) == ((x - 0.5) / 0.5019607)
_NORM_MEAN = (0.5, 0.5, 0.5)
_NORM_STD = (0.5019607, 0.5019607, 0.5019607)


def build_train_transform(input_size: int = config.INPUT_SIZE) -> Callable:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5)
        ], p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(_NORM_MEAN, _NORM_STD),
        # Random erasing simulates glasses, masks, scarves — partial occlusion.
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
    ])


def build_eval_transform(input_size: int = config.INPUT_SIZE) -> Callable:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(_NORM_MEAN, _NORM_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FaceFolderDataset(Dataset):
    """ImageFolder variant that:
       - Builds the (path, class_idx) index once, sorted, deterministic.
       - Skips identities that have fewer than `min_samples_per_class` images
         (helps with VGGFace2 noise and CASIA singletons).
       - Returns (tensor, class_idx); class_idx ranges over 0..num_classes-1.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        min_samples_per_class: int = 2,
    ):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        self.transform = transform

        identity_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        kept_dirs = []
        for d in identity_dirs:
            n = sum(1 for f in d.iterdir() if f.suffix.lower() in _VALID_EXTS)
            if n >= min_samples_per_class:
                kept_dirs.append(d)

        self.classes: list[str] = [d.name for d in kept_dirs]
        self.class_to_idx: dict[str, int] = {name: i for i, name in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        for d in kept_dirs:
            cls_idx = self.class_to_idx[d.name]
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in _VALID_EXTS:
                    self.samples.append((f, cls_idx))

        if not self.samples:
            raise RuntimeError(
                f"No images found under {self.root}. Expected one folder per identity."
            )

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Train / val split + DataLoader builders
# ---------------------------------------------------------------------------

def split_train_val(
    dataset: Dataset,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Random split. For face recognition, identity overlap between train/val
    is fine (val measures how well the loss/embedding generalizes to unseen
    *images* of seen identities). Cross-identity verification accuracy is
    measured separately on LFW (see `evaluate.py`)."""
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    return random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))


def build_dataloaders(
    data_root: str | Path,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    val_fraction: float = 0.05,
    min_samples_per_class: int = 2,
) -> tuple[DataLoader, DataLoader, int]:
    """Build train and val loaders. Returns (train_loader, val_loader, num_classes)."""
    train_tf = build_train_transform()
    eval_tf = build_eval_transform()

    full_train = FaceFolderDataset(
        data_root, transform=train_tf, min_samples_per_class=min_samples_per_class
    )
    train_set, _ = split_train_val(full_train, val_fraction=val_fraction)

    # The val set must use the eval transform, so re-wrap the same indices on a
    # copy of the dataset that has the eval transform applied.
    val_full = FaceFolderDataset(
        data_root, transform=eval_tf, min_samples_per_class=min_samples_per_class
    )
    n_val = len(full_train) - len(train_set)
    _, val_set = split_train_val(val_full, val_fraction=val_fraction)
    assert len(val_set) == n_val, "train/val split sizes inconsistent"

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, full_train.num_classes
