# Voice Commands - Wake Word Dataset Cleaner & Model Training

A complete pipeline for cleaning audio datasets and training a wake word detection model. Includes a web-based annotation interface and a comprehensive Jupyter notebook for model development.

## Project Overview

This module provides tools for:
1. **Dataset Annotation** - A web-based UI for quickly labeling audio files as positive/negative wake word samples
2. **Model Training** - A complete machine learning pipeline for training a wake word detector
3. **Evaluation & Metrics** - Comprehensive model evaluation with confusion matrices and performance curves

## Directory Structure

```
Voice commands/
├── app.py                          # Flask web server for dataset annotation
├── templates/
│   └── index.html                  # Web UI for audio file labeling
├── screenshots/                    # Preview images
│   ├── accuracy1.png
│   ├── confmetrics.png
│   ├── Data_Summary.png
│   ├── modelexport.png
│   ├── Model_Params.png
│   ├── Telegrambot.jpeg
│   ├── testmetrics.png
│   └── trainingcurve.png
└── wake_word_final.ipynb           # Complete training & evaluation notebook
```

## Core Components

### 1. Web-Based Dataset Cleaner (`app.py` + `index.html`)

A Flask web application for collaborative dataset annotation with the following features:

#### Features:
- **User Management** - Each user gets a unique ID (stored in browser localStorage)
- **Assignment-Based Workflow** - Audio files are claimed by users, preventing duplicate labeling
- **Stale Assignment Recovery** - Automatically reclaims assignments after 30 minutes of inactivity
- **Real-Time Counters** - Displays remaining, in-progress, positive, and negative sample counts
- **Audio Streaming** - Efficient conditional HTTP responses for audio playback
- **Keyboard Shortcuts** - Press `1` for Positive, `2` for Negative
- **Directory Auto-Cleanup** - Automatically removes empty folders after file moves

#### Directory Structure (Runtime):
```
cleaned_dataset/
├── positive/          # Labeled positive samples
├── negative/          # Labeled negative samples
└── (in_progress/)     # Temporary working directory
```

#### Key Endpoints:
- `GET /` - Main annotation interface
- `GET /next?user_id=...` - Get next unclaimed audio file
- `GET /audio/<token>` - Stream audio file
- `POST /label` - Save label decision (positive/negative)

#### Thread Safety:
Uses `threading.Lock()` (`ASSIGNMENT_LOCK`) to prevent race conditions during concurrent user access.

### 2. Training & Evaluation Notebook (`wake_word_final.ipynb`)

A comprehensive Jupyter notebook implementing a complete ML pipeline:

#### Data Processing:
- Loads raw audio files from `cleaned_dataset/` directory
- Extracts Mel-spectrogram features (64 mel bands, 16kHz sample rate, 1-second duration)
- Supports audio augmentation:
  - Time shifting (±0.3 seconds)
  - Pitch shifting (±3 semitones)
  - Background noise injection (0.001-0.01 amplitude)
  - Speed variation (0.9-1.1x)

#### Model Architecture: Deep Residual CNN (ResNet-inspired)

**Input:** 64 mel-bands × 32 time-frames (1-second audio clips)

**Layers:**
1. **Initial Convolution Block** → 32 filters
   - 3×3 kernel, MaxPool 2×2
   
2. **Residual Block 1** → 64 filters (stride=2)
   - Dual 3×3 convolutions with BatchNorm
   - Shortcut connection for gradient flow
   
3. **Residual Block 2** → 128 filters (stride=2)
   - Double the channels, improved feature extraction
   
4. **Residual Block 3** → 256 filters (stride=2)
   - Deep feature representation
   
5. **Final Convolution Block** → 512 filters
   - 3×3 kernel + AdaptiveAvgPool → (1, 1)
   
6. **Fully Connected Head**
   - Dense 512 → 256 (ReLU + Dropout 0.5)
   - Dense 256 → 128 (ReLU + Dropout 0.3)
   - Dense 128 → 2 (Binary classification)

**Total Parameters:** ~250K+ trainable parameters
**Special Features:**
- BatchNorm after every convolution (stabilizes training)
- Residual connections (improves gradient flow)
- Dropout layers (prevents overfitting)
- GPU-accelerated training with mixed precision (AMP)

#### Training Configuration:
- **Batch Size:** 32
- **Epochs:** 100 (with early stopping after 15 consecutive epochs without improvement)
- **Optimizer:** Adam with learning rate 0.001, weight decay 1e-5
- **Loss Function:** Weighted CrossEntropyLoss (handles class imbalance)
- **LR Scheduler:** ReduceLROnPlateau (halve LR if no improvement for 5 epochs)
- **Train/Val/Test Split:** 80% / 10% / 10%

#### Model Performance:

**Test Set Results:**
- **Overall Accuracy:** ~95%+
- **Macro-averaged F1-Score:** ~95%

**Per-Class Metrics:**
- **Negative Class (Background/Non-Wake Word)**
  - Precision: ~96%
  - Recall: ~95%
  
- **Positive Class (رشدي Wake Word)**
  - Precision: ~94%
  - Recall: ~96%

**Training Efficiency:**
- Early stopping prevents overfitting (typically stops at ~30-50 epochs)
- Mixed precision training (AMP) enables 2-3x faster convergence on GPU
- Real-time loss/accuracy tracking with progress bars
- Best model automatically checkpointed when validation loss improves

#### Outputs & Visualizations:
- Model checkpoint: `wake_word_model_best.pth`
- Performance metrics:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrices (saved as PNG)
  - Training/validation loss curves
- Screenshots of evaluation results included

#### Stages:
1. Import & configuration
2. Data loading from `cleaned_dataset/`
3. Audio feature extraction (Mel-spectrograms)
4. Dataset splitting (train/val/test)
5. Model definition
6. Training loop with early stopping
7. Evaluation on test set
8. Visualization of results
9. Model export and statistics

## Setup & Usage

### Prerequisites
```
python >= 3.8
flask
torch
torchaudio
librosa
soundfile
scipy
scikit-learn
numpy
matplotlib
jupyter
```

### Running the Web Annotation App

1. Place raw audio files in the configured directory (default: `D:/Zewail City/GP/Rushdiee_bot/wake_word/clips_exact`)

2. Start the Flask server:
```bash
python app.py
```

3. Access the web interface at `http://localhost:5000`

4. Open in another tab/browser for collaborative labeling with different user IDs

5. For network access from other machines:
```
http://<YOUR_LOCAL_IP>:5000
```

### Workflow
1. **Load Sample** - Click "next" or wait for auto-load
2. **Listen** - Play the audio (keyboard shortcut: just press 1 or 2)
3. **Label** - Click "Positive" (press `1`) or "Negative" (press `2`)
4. **Repeat** - Automatically loads next sample

### Running the Training Notebook

```bash
jupyter notebook wake_word_final.ipynb
```

Execute cells sequentially:
- Cell 1-2: Imports and configuration
- Cell 3-5: Data loading and exploration
- Cell 6-9: Feature extraction and augmentation
- Cell 10-15: Model training
- Cell 16+: Evaluation and visualization

## Key Files & Functions

### app.py Key Functions:
- `claim_next_file(user_id)` - Assign unclaimed audio to user
- `find_next_raw_file()` - Find oldest unclaimed file
- `reclaim_stale_assignments()` - Reclaim assignments inactive >30 min
- `label_sample()` - Process and store label decision
- `unique_output_name()` - Generate unique filenames (prevents overwrites)

### index.html Features:
- Real-time counts display
- Audio player with controls
- One-click labeling with button disable during processing
- Keyboard shortcuts (1=Positive, 2=Negative)
- LocalStorage for persistent user ID
- Error handling with user-friendly messages

## Configuration

### app.py Paths:
```python
RAW_DIR = Path("D:/Zewail City/GP/Rushdiee_bot/wake_word/clips_exact")
CLEANED_DIR = BASE_DIR / "cleaned_dataset"
CLAIM_TIMEOUT_SECONDS = 30 * 60  # 30 minutes
```

### Notebook Parameters:
```python
SAMPLE_RATE = 16000      # 16 kHz
DURATION = 1.0           # 1 second
N_MELS = 64              # Mel bands
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
```

## Performance & Monitoring

- **Progress Tracking** - tqdm progress bars in notebook
- **Real-Time Counters** - Web UI shows remaining vs labeled samples
- **Stale Detection** - Automatically reclaims stale assignments after 30-minute timeout
- **Error Resilience** - Graceful handling of file system errors and network issues

## Screenshots

Reference images included in `screenshots/` folder showing:
- Model accuracy curves
- Confusion matrices
- Data summaries
- Training progress visualization

## Notes

- Raw audio files are automatically moved from `RAW_DIR` to either `POSITIVE_DIR` or `NEGATIVE_DIR`
- Empty subdirectories are auto-cleaned after file moves
- Each sample is hashed to prevent duplicate storage despite name collisions
- Multi-user safe with proper thread locking
- Browser autoplay policies may require user interaction before playback
- GPU acceleration available if CUDA is installed
