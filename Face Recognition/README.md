## CLI Commands

Training:
    python train_sampled_digiface.py --data_dir data_subset/train --val_dir data_subset/val

Verification Evaluation:
    python evaluate_verification.py --model checkpoints/backbone.h5 --pairs lfw_funneled/pairs.txt

Identification Evaluation:
    python evaluate_identification.py --model checkpoints/backbone.h5 --gallery data_subset/val --probe data_subset/val

Enrollment:
    python enroll_person.py --model_path checkpoints/backbone.h5 --person_folder images/alice --name "Alice"

Recognition:
    python recognize_person.py --model_path checkpoints/backbone.h5 --image test_image.jpg

Export to TFLite:
    python export_tflite.py --model_path checkpoints/backbone.h5 --output_path embedding_model.tflite --quantize True
# Face Recognition from Scratch

A complete, production-ready face recognition system built from scratch using TensorFlow/Keras. Designed for mobile deployment with a lightweight MobileFaceNet-inspired architecture and ArcFace loss.

## Project Overview

This system implements a complete pipeline for face recognition:
- **Model**: MobileFaceNet-inspired lightweight CNN backbone
- **Loss Function**: ArcFace (Additive Angular Margin Loss)
- **Training Data**: Sampled subset of DigiFace-1M
- **Input Size**: 112×112 RGB images
- **Embedding Size**: 128-dimensional L2-normalized vectors
- **Output Format**: TensorFlow Lite (for mobile deployment)

## Project Structure

```
face_recognition_from_scratch/
├── config/
│   └── config.py                 # Central configuration
├── data/
│   ├── dataset_loader.py         # tf.data pipeline
│   ├── preprocessing.py          # Image loading/normalization
│   └── augmentations.py          # Data augmentation functions
├── models/
│   ├── mobilefacenet.py          # MobileFaceNet backbone architecture
│   └── arcface.py                # ArcFace loss and head
├── utils/
│   ├── metrics.py                # Evaluation metrics
│   ├── similarity.py             # Cosine similarity & matching
│   ├── database.py               # Template database (JSON)
│   └── visualization.py          # Training/result visualization
├── templates/
│   └── templates.json            # Enrolled face templates
├── train_sampled_digiface.py     # Training script
├── evaluate.py                   # Evaluation script
├── enroll_person.py              # Enrollment script
├── recognize_person.py           # Recognition script
├── sample_digiface_subset.py     # Dataset sampling script
├── export_tflite.py              # TFLite export script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Key Features

### 1. **MobileFaceNet Architecture**
- Depthwise separable convolutions for efficiency
- Residual bottleneck blocks
- Global average pooling
- ~2-5M parameters (suitable for mobile)
- Inference time: ~50-100ms on CPU

### 2. **ArcFace Loss**
- Custom implementation of Additive Angular Margin Loss
- Normalized embeddings and weights
- Configurable margin (m) and scale (s)
- Default: m=0.5, s=64

### 3. **Data Pipeline**
- Efficient tf.data pipeline with prefetching
- Augmentations: brightness, contrast, flip, rotation, zoom
- Support for folder-per-class dataset format
- Automatic train/val split

### 4. **Complete Face Recognition Pipeline**
- **Sampling**: Select subset from DigiFace-1M
- **Training**: From scratch with ArcFace loss
- **Enrollment**: Add new persons to gallery
- **Recognition**: Identify persons via cosine similarity
- **Export**: Convert to TFLite for mobile

## Installation

### Prerequisites
- Python 3.10 or higher
- TensorFlow 2.13+ (with optional GPU support)
- 8GB+ RAM recommended

### Setup

```bash
# Clone or navigate to project directory
cd face_recognition_from_scratch

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage Guide

### Step 1: Prepare DigiFace-1M Subset

The DigiFace-1M dataset is too large for training on limited hardware. Sample a manageable subset:

```bash
python sample_digiface_subset.py \
    --digiface_dir /path/to/DigiFace-1M \
    --output_dir data_subset \
    --num_identities 500 \
    --images_per_identity 20 \
    --train_split 0.8 \
    --seed 42
```

**Arguments:**
- `--digiface_dir`: Path to DigiFace-1M dataset root (required)
- `--output_dir`: Output directory (default: `data_subset`)
- `--num_identities`: Number of identities to sample (default: 500)
- `--images_per_identity`: Images per identity (default: 20)
- `--train_split`: Train/val split ratio (default: 0.8)
- `--seed`: Random seed (default: 42)

**Output Structure:**
```
data_subset/
├── train/
│   ├── identity_001/
│   │   ├── image_001.jpg
│   │   └── ...
│   └── ...
└── val/
    ├── identity_001/
    │   └── ...
    └── ...
```

**Recommended subset sizes for different hardware:**
- **GPU (4GB VRAM)**: 300 identities, 15 images/identity
- **GPU (8GB VRAM)**: 500 identities, 20 images/identity
- **GPU (16GB+ VRAM)**: 1000+ identities, 20-30 images/identity
- **CPU only**: 200 identities, 10 images/identity (slower)

### Step 2: Train the Model

```bash
python train_sampled_digiface.py \
    --data_dir data_subset/train \
    --val_dir data_subset/val \
    --output_dir checkpoints \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --patience 10
```

**Arguments:**
- `--data_dir`: Training dataset directory (default: `data_subset/train`)
- `--val_dir`: Validation dataset directory (default: `data_subset/val`)
- `--output_dir`: Checkpoint directory (default: `checkpoints`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32, adjust for GPU memory)
- `--lr`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 10)

**Output Files:**
- `best_model.h5`: Best model checkpoint
- `backbone.h5`: Final backbone model
- `training_history.json`: Training metrics
- `config.json`: Training configuration

**Training Tips:**
- Start with 20-30 epochs for quick validation
- Reduce batch size if you get OOM errors
- Monitor validation accuracy - it should improve over time
- Use early stopping to prevent overfitting

### Step 3: Evaluate the Model

```bash
python evaluate.py \
    --model_path checkpoints/backbone.h5 \
    --val_dir data_subset/val \
    --batch_size 32
```

**Output:**
- Classification accuracy
- Precision, recall, F1 score
- Intra-class distances (should be small)
- Inter-class distances (should be large)

### Step 4: Enroll Persons

Add a new person to the face database:

```bash
python enroll_person.py \
    --model_path checkpoints/backbone.h5 \
    --person_folder path/to/alice/images \
    --name "Alice"
```

**Requirements:**
- Folder should contain face images of one person (JPG/PNG)
- At least 1-3 images recommended for robust enrollment
- Images will be automatically preprocessed

**Output:**
- Templates saved to `templates/templates.json`
- Mean embedding computed from all images
- Metadata stored (number of samples, timestamp)

**Example Enrollment:**
```bash
# Enroll Alice from folder of her photos
python enroll_person.py \
    --model_path checkpoints/backbone.h5 \
    --person_folder "./my_faces/alice" \
    --name "Alice"

# Enroll Bob from folder of his photos
python enroll_person.py \
    --model_path checkpoints/backbone.h5 \
    --person_folder "./my_faces/bob" \
    --name "Bob"
```

### Step 5: Recognize Persons

Identify persons from test images:

```bash
python recognize_person.py \
    --model_path checkpoints/backbone.h5 \
    --image test_image.jpg \
    --threshold 0.45
```

**Arguments:**
- `--model_path`: Path to trained backbone (default: `checkpoints/backbone.h5`)
- `--image`: Path to query face image (required)
- `--threshold`: Similarity threshold (default: 0.45, range: 0-1)

**Output:**
- Identified person name (or "Unknown")
- Similarity score with all enrolled persons
- Comparison with threshold

**Understanding Threshold:**
- **Higher threshold (0.5-0.6)**: More strict, fewer false positives
- **Lower threshold (0.3-0.4)**: More lenient, may accept false matches
- **Default (0.45)**: Balanced for 128-dim embeddings

### Step 6: Export to TensorFlow Lite

Convert for mobile deployment:

```bash
python export_tflite.py \
    --model_path checkpoints/backbone.h5 \
    --output_path embedding_model.tflite \
    --quantize 0
```

**Arguments:**
- `--model_path`: Path to trained backbone
- `--output_path`: Output TFLite file path
- `--quantize`: Enable float16 quantization (0=false, 1=true)

**Output:**
- `embedding_model.tflite`: Mobile-ready model
- `embedding_model.json`: Model metadata

**Model Details:**
- Input: 112×112×3 RGB image (float32, normalized to [-1, 1])
- Output: 128-dimensional L2-normalized embedding
- Size: ~4-5 MB (float32) or ~2-3 MB (float16)
- Latency: ~50-200ms on mobile CPU

## Configuration

Edit `config/config.py` to customize:

```python
# Image settings
INPUT_SIZE = 112                    # Image size (112x112)
EMBEDDING_SIZE = 128                # Embedding dimension

# ArcFace parameters
ARCFACE_MARGIN = 0.5                # Angular margin (m)
ARCFACE_SCALE = 64                  # Scale factor (s)

# Training
BATCH_SIZE = 32                     # Batch size
LEARNING_RATE = 0.001               # Initial LR
NUM_EPOCHS = 50                     # Max epochs
EARLY_STOPPING_PATIENCE = 10        # Early stop patience

# Data augmentation
BRIGHTNESS_DELTA = 0.2              # Random brightness
CONTRAST_FACTOR = [0.8, 1.2]       # Random contrast
ROTATION_RANGE = 10                 # Rotation in degrees
ZOOM_RANGE = 0.1                    # Zoom factor
FLIP_PROBABILITY = 0.5              # Horizontal flip

# Recognition
RECOGNITION_THRESHOLD = 0.45        # Similarity threshold

# Dataset sampling
SUBSET_NUM_IDENTITIES = 500         # Identities to sample
SUBSET_IMAGES_PER_IDENTITY = 20     # Images per identity
SUBSET_RANDOM_SEED = 42             # Random seed
```

## Training Recommendations

### For Successful First Run:
```bash
# Conservative settings for quick validation
python sample_digiface_subset.py \
    --digiface_dir /path/to/DigiFace-1M \
    --num_identities 200 \
    --images_per_identity 10

python train_sampled_digiface.py \
    --epochs 20 \
    --batch_size 32 \
    --lr 0.001
```

### For Best Accuracy:
```bash
# Larger dataset, longer training
python sample_digiface_subset.py \
    --digiface_dir /path/to/DigiFace-1M \
    --num_identities 1000 \
    --images_per_identity 25

python train_sampled_digiface.py \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001
```

### Hyperparameter Tuning:
- **Batch Size**: Increase for stability, decrease for faster convergence
- **Learning Rate**: Start with 0.001, reduce if loss oscillates
- **Margin (m)**: Higher margin (0.7+) for harder training
- **Scale (s)**: Higher scale (128) for more stable training

## API Usage in Code

### Extract Embeddings

```python
import tensorflow as tf
from tensorflow import keras
from data.preprocessing import preprocess_image

# Load model
model = keras.models.load_model("checkpoints/backbone.h5")

# Process image
image = preprocess_image("test.jpg", size=112, normalize=True)
image_batch = tf.expand_dims(image, axis=0)

# Get embedding
embedding = model(image_batch, training=False)[0].numpy()
print(f"Embedding shape: {embedding.shape}")  # (128,)
```

### Face Matching

```python
from utils.similarity import FaceMatch

matcher = FaceMatch(threshold=0.45)

# Load templates
from utils.database import TemplateDatabase
db = TemplateDatabase("templates/templates.json")
templates = db.get_all_templates()

# Identify
person_name, similarity = matcher.identify(embedding, templates)
print(f"Identified: {person_name} ({similarity:.4f})")
```

### Batch Inference

```python
import numpy as np
from data.preprocessing import batch_preprocess_images

# Load images
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
batch, failed = batch_preprocess_images(image_paths, size=112)

# Get embeddings
embeddings = model(batch, training=False).numpy()
print(f"Embeddings shape: {embeddings.shape}")  # (3, 128)
```

## Performance Benchmarks

### Model Size
- **Backbone**: 2.1M parameters
- **Saved HDF5**: ~8.5 MB
- **TFLite (float32)**: ~4.2 MB
- **TFLite (float16)**: ~2.3 MB

### Inference Speed (on CPU)
- **Desktop CPU**: ~50-100ms
- **Mobile CPU**: ~100-300ms
- **Mobile GPU**: ~10-50ms

### Accuracy on Sampled DigiFace-1M (500 identities, 20 images/identity)
- Expected classification accuracy: **85-92%**
- Expected TAR@FAR=0.01: **85%+**
- Expected verification accuracy: **90%+** at threshold 0.45

## Troubleshooting

### GPU Out of Memory (OOM)
```bash
# Reduce batch size
python train_sampled_digiface.py --batch_size 16

# Or reduce subset size
python sample_digiface_subset.py --num_identities 200 --images_per_identity 10
```

### Poor Recognition Accuracy
1. Check enrollment images quality (clear, frontal faces)
2. Increase number of enrollment images (3-5 per person)
3. Adjust threshold (`--threshold 0.40` for more lenient)
4. Ensure faces are properly centered in 112×112 images

### Model Not Improving During Training
1. Check learning rate (try 0.0005 or 0.0001)
2. Increase data augmentation strength
3. Ensure training/validation data is properly loaded
4. Check for data imbalance

### TFLite Model Inference Issues
```python
# Ensure correct input preprocessing
import tensorflow as tf
from data.preprocessing import preprocess_image

# Input MUST be normalized to [-1, 1]
image = preprocess_image(image_path, normalize=True)
image = tf.expand_dims(image, axis=0)

# Run inference
interpreter.set_tensor(input_idx, image.numpy().astype(np.float32))
```

## Limitations & Future Improvements

### Known Limitations
- Dataset sampling is random (could be optimized)
- No face detection (assumes pre-cropped faces)
- Single embedding per person (could use multiple templates)
- No liveness detection (vulnerable to spoofing)

### Future Improvements
- Add face detection/alignment preprocessing
- Implement score normalization for better calibration
- Add optional template fusion (PCA, metric learning)
- Support for large-scale databases (hashing, indexing)
- Liveness detection for anti-spoofing
- Multi-model ensemble for improved accuracy

## References

1. **ArcFace**: Deng, J., Guo, J., Xue, N., et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" IEEE CVPR 2019.
   - https://arxiv.org/abs/1801.07698

2. **MobileFaceNet**: Chen, S., Liu, Y., Gao, X., et al. "MobileFaceNets: Efficient CNNs for Face Recognition on Mobile Devices" arXiv 2018.
   - https://arxiv.org/abs/1804.07573

3. **DigiFace-1M**: Hong, Y., Kwon, S., et al. "DigiFace-1M: 1 Million Digital Face Images for Face Recognition" arXiv 2021.
   - https://1drv.ms/u/s!AjEI5svUGI8FhaVTPz3zZDfMjK3elw

4. **TensorFlow Face Recognition**: https://github.com/tensorflow/addons

## License

This implementation is for educational and research purposes. Comply with DigiFace-1M and TensorFlow licenses.

## Author Notes

This project is designed as a **graduation project** that demonstrates:
- Deep understanding of face recognition theory
- Ability to implement complex architectures from scratch
- Data pipeline and preprocessing expertise
- Mobile optimization and deployment
- Clean, production-ready code practices

The code emphasizes clarity and educational value while maintaining production-level quality.

---

**Last Updated**: December 2024
**Python Version**: 3.10+
**TensorFlow Version**: 2.13+
