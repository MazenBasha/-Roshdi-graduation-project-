"""
Configuration for Egyptian Currency Detection Model.
All hyperparameters and paths are centralized here.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VALID_DIR = os.path.join(DATA_ROOT, "valid")
TEST_DIR = os.path.join(DATA_ROOT, "test")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
EXPORT_PATH = os.path.join(OUTPUT_DIR, "model.ptl")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# ─── Class Mapping ───────────────────────────────────────────────────────────
# Folder names → class indices (sorted for reproducibility)
CLASS_NAMES = ["1", "5", "10", "10 (new)", "20", "20 (new)", "50", "100", "200"]
NUM_CLASSES = len(CLASS_NAMES)

# ─── Image Settings ──────────────────────────────────────────────────────────
IMG_SIZE = 224          # Resize all images to IMG_SIZE x IMG_SIZE
IMG_CHANNELS = 3
# ImageNet-like normalization (generic, not pretrained-dependent)
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# ─── Model Settings ─────────────────────────────────────────────────────────
MODEL_WIDTH_MULT = 1.0  # Width multiplier for MobileNet-style model
DROPOUT_RATE = 0.2

# ─── Training Settings ──────────────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 32
# CRITICAL: On Windows, num_workers>0 uses 'spawn' which duplicates the
# entire process per worker, causing massive memory growth. Use 0 on Windows.
NUM_WORKERS = 0
NUM_EPOCHS = 150
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

# ─── Learning Rate Scheduler ────────────────────────────────────────────────
LR_SCHEDULER = "cosine"  # "cosine" or "step"
LR_STEP_SIZE = 30
LR_GAMMA = 0.1
LR_MIN = 1e-6

# ─── Early Stopping ─────────────────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 1e-4

# ─── Mixed Precision ────────────────────────────────────────────────────────
USE_AMP = True  # Automatic Mixed Precision

# ─── Checkpoint ──────────────────────────────────────────────────────────────
SAVE_EVERY_N_EPOCHS = 5

# ─── Data Augmentation ──────────────────────────────────────────────────────
AUG_RANDOM_CROP_SCALE = (0.5, 1.0)   # wider range: note can be far from camera
AUG_RANDOM_CROP_RATIO = (0.6, 1.67)  # wider ratio: rotated/tilted notes
AUG_HORIZONTAL_FLIP_P = 0.5
AUG_VERTICAL_FLIP_P = 0.2
AUG_ROTATION_DEGREES = 45             # more rotation for hand-held notes
AUG_COLOR_JITTER_BRIGHTNESS = 0.5     # stronger: camera lighting varies a lot
AUG_COLOR_JITTER_CONTRAST = 0.4
AUG_COLOR_JITTER_SATURATION = 0.3
AUG_COLOR_JITTER_HUE = 0.1
AUG_GAUSSIAN_BLUR_P = 0.2
AUG_PERSPECTIVE_DISTORTION = 0.3
AUG_PERSPECTIVE_P = 0.3
AUG_ERASING_P = 0.2
