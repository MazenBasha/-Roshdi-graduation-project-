"""
Central configuration for the Egyptian Currency Detection project.

Edit paths or hyperparameters here — all scripts (train, detect, evaluate)
import from this single file.
"""

import os

# ─── Project paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data_yolo")
DATA_YAML = os.path.join(DATA_DIR, "data.yaml")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")
DETECT_OUT_DIR = os.path.join(OUTPUT_DIR, "detections")

# Default weights resolution: prefer the active entry in the model registry
# (so `--weights` can be flipped without editing this file), and fall back to
# the legacy hard-coded path when the registry is empty or unavailable.
_FALLBACK_WEIGHTS = os.path.join(RUNS_DIR, "train_finetune", "weights", "best.pt")

def _resolve_default_weights() -> str:
    try:
        from model_registry import ModelRegistry  # local import to avoid cycles
        active = ModelRegistry.get_active_model()
        if active and os.path.exists(active):
            return active
    except Exception:
        pass
    return _FALLBACK_WEIGHTS

DEFAULT_WEIGHTS = _resolve_default_weights()

# ─── Class names ─────────────────────────────────────────────────────────────
# Order MUST match label IDs in your YOLO .txt files.
CLASS_NAMES = [
    "1_EGP",
    "5_EGP",
    "10_EGP",
    "20_EGP",
    "50_EGP",
    "100_EGP",
    "200_EGP",
]

# Numeric value of each class (Egyptian pounds), used for the running total.
DENOMINATIONS = {
    "1_EGP": 1,
    "5_EGP": 5,
    "10_EGP": 10,
    "20_EGP": 20,
    "50_EGP": 50,
    "100_EGP": 100,
    "200_EGP": 200,
}

NUM_CLASSES = len(CLASS_NAMES)

# ─── Training defaults ───────────────────────────────────────────────────────
BASE_MODEL = "yolov8n.pt"     # nano: smallest + fastest, good for grad project
IMG_SIZE = 640
EPOCHS = 100
BATCH_SIZE = 16
PATIENCE = 20                  # early-stopping patience on val mAP
SEED = 42

# ─── Detection defaults ──────────────────────────────────────────────────────
CONF_THRESHOLD = 0.45          # min box confidence to keep
IOU_THRESHOLD = 0.5            # NMS IoU threshold
MAX_DETECTIONS = 50            # safety cap per frame
# Class-agnostic NMS: suppress overlapping boxes even when they carry different
# class labels. Critical here because the model can place two boxes on a single
# note with conflicting denominations; without this, both survive NMS and
# inflate the count/total.
AGNOSTIC_NMS = True
