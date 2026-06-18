
# Beyond The Limits — Assistive AI System

A multi-modal AI system designed to assist visually impaired users by providing real-time perception through **voice commands**, **face recognition**, and **Egyptian currency detection**.

The system integrates three independent deep-learning modules:

1. **Voice Commands (Wake Word Detection)** – Activates the system via the Arabic wake word *"رشدي"*
2. **Face Recognition** – Identifies registered individuals using lightweight facial embeddings
3. **Egyptian Currency Detection** – Recognizes Egyptian banknotes using a mobile-optimized CNN

Each module is implemented as an independent subsystem that can run locally or be integrated into a mobile application.

---

## Repository Structure

```
BeyondTheLimits/
│
├── Face_recognition/
│   ├── scripts/
│   ├── models/
│   ├── best_model.h5
│   ├── backbone.h5
│   ├── README.md
│   └── ...
│
├── Currency_detection/
│   ├── train.py
│   ├── evaluate.py
│   ├── model.py
│   ├── export_ptl.py
│   ├── README.md
│   └── ...
│
├── wake_word/
│   ├── CLEANING/
│   │   └── Voice commands/
│   │       ├── app.py
│   │       ├── templates/
│   │       ├── wake_word_final.ipynb
│   │       ├── README.md
│   │       └── ...
│   └── ...
│
└── README.md (this file)
```

---

## System Overview

### Workflow

```
Wake Word ("رشدي") Detected
        │
        ▼
Voice Command Processing
        │
        ├──→ Face Recognition
        │    │
        │    └──→ Identify person in camera
        │
        └──→ Currency Detection
             │
             └──→ Detect banknote value
```

The wake word activates the assistant, after which the system executes the requested model.

---

## 1. Face Recognition Module

### Purpose

This subsystem performs **deep face recognition** for identifying registered individuals.

Supported operations:
- **1:1 Verification** (Is this person X?)
- **1:N Identification** (Who is this person?)

### Architecture

The model uses **MobileFaceNet**, a lightweight neural network optimized for mobile devices.

| Attribute | Value |
|-----------|-------|
| Backbone | MobileFaceNet |
| Training Head | ArcFace (Additive Angular Margin) |
| Input Size | 112×112 RGB |
| Embedding Size | 128-D normalized |
| Total Parameters | 196,800 |
| Model Size | ~930 KB |

**ArcFace Parameters:**
- Margin: **0.5 rad**
- Scale: **64**

### Training Pipeline

Training uses **TensorFlow / Keras** with `tf.data` pipeline.

**Data Augmentations:**
- Random brightness
- Random contrast
- Horizontal flip

**Training Configuration:**

| Parameter | Value |
|----------|-------|
| Epochs | 30 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Early Stopping | patience = 10 |

### Dataset

Two datasets were combined:

| Dataset | Details |
|---------|---------|
| LFW Funneled | ~13.2k images, 5,749 identities |
| DigiFace-1M Subset | 2,000 identities |

**Split:**
- 1,600 identities for training
- 400 identities for validation

### Training Results

| Metric | Value |
|--------|-------|
| Final Training Accuracy | **91.85%** |
| Final Validation Accuracy | **47.21%** |
| Best Validation Loss | 4.1301 |
| Best Epoch | 25 |

*Note: The validation accuracy reflects the large-class classification problem (5,749 identities to classify).*

### Inference Pipeline

```
Face Image
    │
    ▼
Preprocessing (112×112)
    │
    ▼
MobileFaceNet Backbone
    │
    ▼
128-D Face Embedding
    │
    ├──→ Cosine Similarity (1:1 verification)
    └──→ Gallery Search (1:N identification)
```

### Delivered Artifacts

**Models:**
- `best_model.h5` - Full model with ArcFace head
- `training_model.h5` - Training checkpoint
- `backbone.h5` - Inference backbone only
- `training_history.json` - Training metrics
- `training.log` - Detailed training logs

**Scripts:**
- `evaluate.py` - Model evaluation
- `evaluate_verification.py` - 1:1 verification tests
- `evaluate_identification.py` - 1:N identification tests
- `recognize_person.py` - Real-time recognition
- `enroll_person.py` - Enroll new person
- `camera_test.py` - Live camera testing
- `export_tflite.py` - Mobile deployment export

**Mobile Export:**
- TensorFlow Lite (float16) format for iOS/Android deployment

---

## 2. Egyptian Currency Detection Module

### Purpose

Recognizes Egyptian banknotes using a lightweight CNN designed for mobile devices.

### Supported Classes

| Index | Banknote |
|-------|----------|
| 0 | 1 EGP |
| 1 | 5 EGP |
| 2 | 10 EGP (old design) |
| 3 | 10 EGP (new design) |
| 4 | 20 EGP (old design) |
| 5 | 20 EGP (new design) |
| 6 | 50 EGP |
| 7 | 100 EGP |
| 8 | 200 EGP |

### Model Architecture

Custom **CurrencyMobileNet** inspired by MobileNetV2.

**Key Components:**
- Depthwise separable convolutions
- Inverted residual blocks
- ReLU6 activations
- Global average pooling

**Model Statistics:**

| Attribute | Value |
|-----------|-------|
| Total Parameters | ~2.2M |
| FP32 Model Size | ~8.5 MB |
| TorchScript Lite Size | ~8 MB |
| Inference Speed | Real-time on mobile |

**Advantages:**
- Fast inference on edge devices
- Mobile-optimized architecture
- Quantization ready
- Low memory footprint

### Training Features

**Initialization & Regularization:**
- Kaiming initialization
- Label smoothing (α=0.1)
- Weighted sampling for class imbalance
- Cosine annealing learning rate decay
- Mixed precision training (AMP)
- Gradient clipping

**Data Augmentations:**
- Random crop (80-100% scale)
- Rotation (±15°)
- Perspective distortion
- Gaussian blur
- Random erasing
- Color jitter (brightness, contrast, saturation)

These augmentations handle real-world variations:
- Folded/bent notes
- Partial occlusion
- Variable lighting conditions
- Worn/damaged currency

### Evaluation Metrics

Comprehensive evaluation includes:
- Accuracy
- Per-class Precision, Recall, F1-Score
- Mean Average Precision (mAP)
- Confusion Matrix
- Class-specific analysis

### Training & Evaluation

**Train Model:**
```bash
python train.py
```

**Custom Training:**
```bash
python train.py --epochs 50 --batch-size 64 --lr 0.005
```

**Evaluate on Test Set:**
```bash
python evaluate.py
```

**Export Mobile Model:**
```bash
python export_ptl.py
```

This creates `model.ptl` (TorchScript Lite format).

### Mobile Deployment

The model is exported as **TorchScript Lite (.ptl)**.

**Example Android Integration:**
```kotlin
val module = LiteModuleLoader.load(assetFilePath(this, "model.ptl"))
val inputTensor = TensorImageFactory.imageYUV420CpuImageToFloatDeviceImage(image)
val output = module.forward(inputTensor)
val confidences = output.confidences
```

---

## 3. Voice Commands — Wake Word Detection

### Purpose

Detects the Arabic wake word **"رشدي"** to activate the assistant.

The module includes:
- **Dataset Cleaning Tool** – Web-based collaborative annotation
- **Training Pipeline** – Mel-spectrogram CNN training
- **Evaluation System** – Comprehensive metrics and visualization

For detailed information, see [Voice Commands README](wake_word/CLEANING/Voice%20commands/README.md).

### Dataset Cleaning Tool

A Flask web application for collaborative labeling of wake word audio samples.

**Features:**
- Multi-user labeling with persistent user IDs
- File assignment locking (prevents duplicate labeling)
- Automatic stale assignment recovery (30-minute timeout)
- Real-time sample count tracking
- Keyboard shortcuts for fast labeling
- Thread-safe concurrent access

**Keyboard Shortcuts:**
- `1` → Mark as Positive (wake word detected)
- `2` → Mark as Negative (no wake word)

**Runtime Directory Structure:**
```
cleaned_dataset/
├── positive/          # Wake word samples
├── negative/          # Non-wake word samples
└── in_progress/       # Temporary assignment directory
```

Files are automatically moved after labeling with unique filenames to prevent collisions.

### Training Pipeline

Implemented in `wake_word_final.ipynb`.

**Pipeline Stages:**
1. Load and explore cleaned dataset
2. Extract Mel-spectrogram features from audio
3. Apply data augmentation (pitch shift, time stretch, noise injection)
4. Split into train/validation/test sets
5. Train ResNet-style CNN with early stopping
6. Evaluate on held-out test set
7. Generate performance visualizations
8. Export best model checkpoint

### Audio Configuration

| Parameter | Value |
|-----------|-------|
| Sample Rate | 16 kHz |
| Duration | 1 second |
| Mel-spectrogram Bands | 64 |
| Input Shape | (64, 32) 2D features |
| Normalization | Log-scaled, min-max normalized |

### Model Architecture

**Deep Residual CNN (ResNet-inspired)**

**Layer Structure:**

1. **Initial Convolution Block** → 32 filters
   - 3×3 kernel, MaxPool 2×2
   
2. **Residual Block 1** → 64 filters (stride=2)
   - Dual 3×3 convolutions with BatchNorm
   - Skip connection for improved gradients
   
3. **Residual Block 2** → 128 filters (stride=2)
   - Deeper feature extraction
   
4. **Residual Block 3** → 256 filters (stride=2)
   - High-level feature representation
   
5. **Final Convolution Block** → 512 filters
   - 3×3 kernel + AdaptiveAvgPool → (1, 1)
   
6. **Fully Connected Classifier Head**
   - Dense 512 → 256 (ReLU + Dropout 0.5)
   - Dense 256 → 128 (ReLU + Dropout 0.3)
   - Dense 128 → 2 (Binary classification: wake word vs background)

**Architecture Details:**

| Attribute | Value |
|-----------|-------|
| Total Parameters | ~250K+ |
| Input Channels | 1 (mono mel-spectrogram) |
| Output Classes | 2 (Positive/Negative) |
| Activation | ReLU |
| Normalization | BatchNorm2d after each conv |
| Regularization | Dropout + residual connections |
| Optimization | GPU-accelerated with AMP (mixed precision) |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Max Epochs | 100 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Weight Decay | 1e-5 |
| Early Stopping | patience = 15 epochs |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Loss Function | Weighted CrossEntropyLoss (handles class imbalance) |

**Dataset Split:**
- Training: 80%
- Validation: 10%
- Testing: 10%

### Model Performance

**Test Set Results:**

| Metric | Value |
|--------|-------|
| Overall Accuracy | **~95%+** |
| Macro F1-Score | **~95%** |
| Training Stability | Early stopping (typically 30-50 epochs) |

**Per-Class Performance:**

**Negative Class (Background/Non-Wake Word):**
- Precision: ~96%
- Recall: ~95%

**Positive Class (رشدي Wake Word):**
- Precision: ~94%
- Recall: ~96%

**Training Efficiency:**
- Early stopping prevents overfitting
- Mixed precision training (AMP) enables 2-3x faster convergence on GPU
- Real-time progress tracking with tqdm
- Automatic checkpointing of best model

### Running the Dataset Cleaner

1. **Start the Flask server:**
```bash
cd wake_word/CLEANING/Voice\ commands
python app.py
```

2. **Open in browser:**
```
http://localhost:5000
```

3. **For network access from other machines:**
```
http://<YOUR_LOCAL_IP>:5000
```

### Workflow

1. **Load Sample** – Click "next" or wait for auto-load
2. **Listen** – Play the audio
3. **Label** – Click "Positive" (press `1`) or "Negative" (press `2`)
4. **Repeat** – Automatically loads next sample

### Running the Training Notebook

```bash
cd wake_word/CLEANING/Voice\ commands
jupyter notebook wake_word_final.ipynb
```

Execute cells sequentially:
- Cells 1-2: Imports and configuration setup
- Cells 3-5: Data loading and exploration
- Cells 6-9: Feature extraction and augmentation functions
- Cells 10-15: Model architecture and training loop
- Cells 16+: Evaluation, metrics, and visualization

---

## System Requirements

### Python Environment
- **Python:** ≥ 3.8
- **CUDA:** Optional (GPU acceleration recommended)

### Required Libraries

```
tensorflow >= 2.8
torch >= 1.10
torchaudio >= 0.10
flask >= 2.0
librosa >= 0.9
scikit-learn >= 1.0
numpy >= 1.20
matplotlib >= 3.4
jupyter
scipy >= 1.7
pandas >= 1.3
soundfile >= 0.10
```

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Module Structure

### Face Recognition

**Location:** `Face_recognition/`

Key files:
- `best_model.h5` - Best performing model
- `evaluate.py` - Model evaluation script
- `recognize_person.py` - Real-time recognition
- `enroll_person.py` - Enroll new identities

For detailed instructions, see `Face_recognition/README.md`.

### Currency Detection

**Location:** `Currency_detection/`

Key files:
- `model.py` - CurrencyMobileNet architecture
- `train.py` - Training script
- `evaluate.py` - Evaluation script
- `export_ptl.py` - Mobile export

For detailed instructions, see `Currency_detection/README.md`.

### Voice Commands

**Location:** `wake_word/CLEANING/Voice commands/`

Key files:
- `app.py` - Flask annotation server
- `wake_word_final.ipynb` - Training notebook
- `templates/index.html` - Web UI

For detailed instructions, see `wake_word/CLEANING/Voice commands/README.md`.

---

## Future Integration

The final system will integrate all three modules into a **unified mobile assistant application** capable of:

- ✓ Listening for wake word activation (*"رشدي"*)
- ✓ Identifying people via face recognition
- ✓ Recognizing Egyptian currency denominations
- ✓ Voice-based query responses
- ✓ Haptic feedback for results

This will enable **visually impaired users** to:
- Identify who they're talking to
- Determine the value of banknotes they're holding
- Control the device through voice commands
- Receive instant audio feedback

---

## Project Information

| Field | Details |
|-------|---------|
| **Project Name** | Beyond The Limits |
| **Objective** | Assistive AI for visually impaired users |
| **Type** | Graduation Project |
| **Modules** | 3 (Face Recognition, Currency Detection, Voice Commands) |
| **Framework** | TensorFlow/Keras, PyTorch, Flask |
| **Target Platform** | Mobile (iOS/Android) + Desktop |
| **Model Export** | TensorFlow Lite, TorchScript Lite |

---

## License & Attribution

This is an academic project. See individual module READMEs for specific licensing information.

---

## Contact & Support

For issues or questions about specific modules, refer to their individual README files:

- [Face Recognition](Face_recognition/README.md)
- [Currency Detection](Currency_detection/README.md)
- [Voice Commands](wake_word/CLEANING/Voice%20commands/README.md)

---

**Last Updated:** March 2026
