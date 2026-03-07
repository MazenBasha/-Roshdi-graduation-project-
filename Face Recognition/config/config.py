"""
Configuration module for face recognition system.

This module contains all hyperparameters, paths, and settings used throughout
the face recognition pipeline.
"""

from pathlib import Path
from typing import Dict, Any


class Config:
    """Central configuration class for the face recognition system."""

    # ==================== Project Paths ====================
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data_subset"
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "val"
    TEMPLATES_DIR = PROJECT_ROOT / "templates"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    MODELS_DIR = PROJECT_ROOT / "models_saved"
    LOGS_DIR = PROJECT_ROOT / "logs"

    # ==================== Image Settings ====================
    INPUT_SIZE = 112  # Height and width of input images
    INPUT_CHANNELS = 3  # RGB
    EMBEDDING_SIZE = 128  # Output embedding dimension

    # ==================== ArcFace Hyperparameters ====================
    ARCFACE_MARGIN = 0.5  # Angular margin (m)
    ARCFACE_SCALE = 64  # Scaling factor (s)

    # ==================== Training Hyperparameters ====================
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    VALIDATION_SPLIT = 0.1  # Use 10% for validation during training
    EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for N epochs

    # Learning rate scheduler
    LR_REDUCE_FACTOR = 0.5
    LR_REDUCE_PATIENCE = 5
    MIN_LEARNING_RATE = 1e-6

    # ==================== Data Augmentation Settings ====================
    # Brightness augmentation
    BRIGHTNESS_DELTA = 0.2

    # Contrast augmentation
    CONTRAST_FACTOR = [0.8, 1.2]

    # Rotation augmentation (in degrees)
    ROTATION_RANGE = 10

    # Zoom/crop augmentation
    ZOOM_RANGE = 0.1

    # Horizontal flip probability
    FLIP_PROBABILITY = 0.5

    # ==================== Recognition Threshold ====================
    RECOGNITION_THRESHOLD = 0.45  # Cosine similarity threshold for recognition

    # ==================== Dataset Sampling Defaults ====================
    SUBSET_NUM_IDENTITIES = 500  # Number of identities to sample
    SUBSET_IMAGES_PER_IDENTITY = 20  # Images per identity
    SUBSET_RANDOM_SEED = 42  # Random seed for reproducibility

    # ==================== Model Architecture ====================
    # MobileFaceNet-inspired settings
    WIDTH_MULTIPLIER = 1.0  # Width multiplier for depthwise separable convolutions
    DROPOUT_RATE = 0.5

    # ==================== TFLite Export Settings ====================
    TFLITE_QUANTIZATION = False  # Set to True for float16 quantization
    TFLITE_OUTPUT_PATH = MODELS_DIR / "embedding_model.tflite"

    # ==================== Enrollment Settings ====================
    TEMPLATES_JSON_PATH = TEMPLATES_DIR / "templates.json"

    # ==================== System Settings ====================
    RANDOM_SEED = 42
    VERBOSE = 1  # Verbosity level (0=silent, 1=progress, 2=detailed)

    @classmethod
    def create_directories(cls) -> None:
        """Create all necessary directories if they don't exist."""
        for directory in [
            cls.TRAIN_DIR,
            cls.VAL_DIR,
            cls.TEMPLATES_DIR,
            cls.CHECKPOINT_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary for logging and inspection."""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith("_") and key.isupper()
        }


# Create directories on import
Config.create_directories()
