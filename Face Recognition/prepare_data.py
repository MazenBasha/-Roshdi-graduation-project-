#!/usr/bin/env python3
"""
Split dataset into train/val directories for training.
"""
import os
import shutil
from pathlib import Path

source_dir = Path("/Users/jilan/Downloads/subjects_0-1999_72_imgs")
train_dir = Path("/Users/jilan/Documents/face recognition from scratch/data_subset/train")
val_dir = Path("/Users/jilan/Documents/face recognition from scratch/data_subset/val")

# Get all identity folders (sorted numerically)
identities = sorted([d for d in source_dir.iterdir() if d.is_dir()], 
                   key=lambda x: int(x.name))

print(f"Found {len(identities)} identities")

# Split: 80% train, 20% val
split_idx = int(len(identities) * 0.8)
train_identities = identities[:split_idx]
val_identities = identities[split_idx:]

print(f"Train: {len(train_identities)} identities")
print(f"Val: {len(val_identities)} identities")

# Copy train data
print("\nCopying training data...")
for i, identity_dir in enumerate(train_identities):
    dest = train_dir / identity_dir.name
    if not dest.exists():
        shutil.copytree(identity_dir, dest)
    if (i + 1) % 200 == 0:
        print(f"  Copied {i + 1}/{len(train_identities)}")

print(f"✓ Copied {len(train_identities)} training identities")

# Copy val data
print("\nCopying validation data...")
for i, identity_dir in enumerate(val_identities):
    dest = val_dir / identity_dir.name
    if not dest.exists():
        shutil.copytree(identity_dir, dest)
    if (i + 1) % 100 == 0:
        print(f"  Copied {i + 1}/{len(val_identities)}")

print(f"✓ Copied {len(val_identities)} validation identities")

# Verify
train_count = len(list(train_dir.iterdir()))
val_count = len(list(val_dir.iterdir()))
print(f"\n✓ Complete!")
print(f"  Train: {train_count} identities")
print(f"  Val: {val_count} identities")
