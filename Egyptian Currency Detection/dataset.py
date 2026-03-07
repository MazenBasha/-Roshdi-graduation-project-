"""
Dataset loading and augmentation for Egyptian Currency Classification.

Supports folder-based image classification datasets with structure:
    data/
        train/
            class_name/
                image1.jpg
                ...
        valid/
            ...
        test/
            ...
"""

import os
import random
from typing import Tuple, List, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageFilter

import config


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_dataset(data_dir: str, class_names: List[str]) -> Dict[str, int]:
    """
    Validate dataset integrity: check folders exist and contain images.
    Returns dict of class_name -> image_count.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    stats = {}
    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    for cls_name in class_names:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(
                f"Class directory not found: {cls_dir}\n"
                f"Expected classes: {class_names}"
            )
        images = [
            f for f in os.listdir(cls_dir)
            if os.path.splitext(f)[1].lower() in supported_ext
        ]
        if len(images) == 0:
            raise ValueError(f"No images found in class directory: {cls_dir}")
        stats[cls_name] = len(images)

    return stats


# ─── Custom Augmentations ───────────────────────────────────────────────────

class RandomGaussianBlur:
    """Apply Gaussian blur with a given probability (simulates motion blur)."""

    def __init__(self, p: float = 0.2, radius_range: Tuple[float, float] = (0.5, 2.0)):
        self.p = p
        self.radius_range = radius_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


# ─── Transforms ──────────────────────────────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    """
    Training augmentations targeting currency detection robustness:
    - RandomResizedCrop: scale variation, partial occlusion
    - Flips: orientation invariance
    - Rotation: rotated notes
    - ColorJitter: lighting variation, worn currency
    - Perspective: perspective distortion
    - GaussianBlur: motion blur / out-of-focus
    - RandomErasing: partial occlusion, folded notes
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            config.IMG_SIZE,
            scale=config.AUG_RANDOM_CROP_SCALE,
            ratio=config.AUG_RANDOM_CROP_RATIO,
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=config.AUG_HORIZONTAL_FLIP_P),
        transforms.RandomVerticalFlip(p=config.AUG_VERTICAL_FLIP_P),
        transforms.RandomRotation(
            degrees=config.AUG_ROTATION_DEGREES,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        ),
        transforms.ColorJitter(
            brightness=config.AUG_COLOR_JITTER_BRIGHTNESS,
            contrast=config.AUG_COLOR_JITTER_CONTRAST,
            saturation=config.AUG_COLOR_JITTER_SATURATION,
            hue=config.AUG_COLOR_JITTER_HUE,
        ),
        transforms.RandomPerspective(
            distortion_scale=config.AUG_PERSPECTIVE_DISTORTION,
            p=config.AUG_PERSPECTIVE_P,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        ),
        RandomGaussianBlur(p=config.AUG_GAUSSIAN_BLUR_P),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        transforms.RandomErasing(
            p=config.AUG_ERASING_P,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
        ),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Deterministic transforms for validation/test/inference."""
    return transforms.Compose([
        transforms.Resize(int(config.IMG_SIZE * 1.15), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
    ])


# ─── Dataset ────────────────────────────────────────────────────────────────

class CurrencyDataset(Dataset):
    """
    Image classification dataset for currency notes.
    Loads images from class subdirectories.
    """

    SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(
        self,
        root_dir: str,
        class_names: List[str],
        transform: Optional[transforms.Compose] = None,
    ):
        self.root_dir = root_dir
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        for cls_name in class_names:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if os.path.splitext(fname)[1].lower() in self.SUPPORTED_EXT:
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls_name])
                    )

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        # Open and immediately copy image data, then close the file handle.
        # Without explicit close, PIL leaks file descriptors and associated
        # memory buffers across thousands of images per epoch.
        with Image.open(img_path) as img_file:
            img = img_file.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency weights for handling class imbalance."""
        class_counts = [0] * len(self.class_names)
        for _, label in self.samples:
            class_counts[label] += 1
        total = sum(class_counts)
        weights = [total / (len(self.class_names) * c) for c in class_counts]
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> List[float]:
        """Per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        return [class_weights[label].item() for _, label in self.samples]


# ─── DataLoaders ─────────────────────────────────────────────────────────────

def create_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    use_weighted_sampling: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        batch_size: Batch size for all loaders.
        num_workers: Number of data loading workers.
        use_weighted_sampling: Use WeightedRandomSampler for class imbalance.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_dataset = CurrencyDataset(
        config.TRAIN_DIR, config.CLASS_NAMES, get_train_transforms()
    )
    val_dataset = CurrencyDataset(
        config.VALID_DIR, config.CLASS_NAMES, get_eval_transforms()
    )
    test_dataset = CurrencyDataset(
        config.TEST_DIR, config.CLASS_NAMES, get_eval_transforms()
    )

    # Weighted sampling to handle class imbalance
    sampler = None
    shuffle = True
    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False  # Sampler and shuffle are mutually exclusive

    # Only use pin_memory when a CUDA device is available;
    # on CPU-only systems it wastes page-locked memory for no benefit.
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick dataset sanity check
    print("Validating dataset...")
    for split_name, split_dir in [
        ("Train", config.TRAIN_DIR),
        ("Valid", config.VALID_DIR),
        ("Test", config.TEST_DIR),
    ]:
        stats = validate_dataset(split_dir, config.CLASS_NAMES)
        total = sum(stats.values())
        print(f"\n{split_name} ({total} images):")
        for cls, count in stats.items():
            print(f"  {cls}: {count}")

    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=4)
    batch_imgs, batch_labels = next(iter(train_loader))
    print(f"Batch shape: {batch_imgs.shape}")
    print(f"Labels: {batch_labels.tolist()}")
    print(f"Label names: {[config.CLASS_NAMES[l] for l in batch_labels.tolist()]}")
    print("\nDataset validation passed!")
