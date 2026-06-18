"""
Central configuration for the Face Recognition Gilan project.

All paths are resolved relative to this file so the project is
runnable from any working directory.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
FACE_DB_DIR = PROJECT_ROOT / "face_db"
LOG_DIR = PROJECT_ROOT / "logs"
EXAMPLES_DIR = PROJECT_ROOT / "examples"

for _d in (DATA_DIR, CHECKPOINT_DIR, FACE_DB_DIR, LOG_DIR, EXAMPLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FACE_DB_PATH = FACE_DB_DIR / "faces.pkl"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "arcface_iresnet50.pt"

# --- Model -------------------------------------------------------------------
EMBEDDING_DIM = 512        # iResNet output, ArcFace standard
INPUT_SIZE = 112           # 112x112 RGB, ArcFace standard
BACKBONE = "iresnet50"     # iresnet18 | iresnet50 | iresnet100 | facenet_vggface2

# --- Training ----------------------------------------------------------------
BATCH_SIZE = 128
NUM_WORKERS = 4
EPOCHS = 30
LR = 0.1                   # SGD with cosine schedule, ArcFace recipe
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 1

# ArcFace head
ARC_SCALE = 64.0
ARC_MARGIN = 0.5

# --- Inference ---------------------------------------------------------------
# Cosine-similarity threshold for "same person".
# 0.40 is conservative (fewer false accepts), 0.30 is permissive.
MATCH_THRESHOLD = 0.40

# How many embeddings to store per enrolled person (rolling window).
MAX_EMBEDDINGS_PER_PERSON = 10

# Minimum face detection confidence (MTCNN)
DETECTION_CONFIDENCE = 0.90

# Minimum face size in pixels — anything smaller is ignored.
MIN_FACE_SIZE = 40
