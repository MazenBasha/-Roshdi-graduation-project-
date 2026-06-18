# Beyond The Limits — Roshdi Assistive AI System

A multi-modal AI system designed to assist visually impaired users by providing real-time perception through **voice commands**, **face recognition**, and **Egyptian currency detection** — all delivered through a single cross-platform **Flutter** mobile app (*Roshdi / رشدي*).

The system integrates three independent deep-learning modules plus the mobile application that runs them on-device:

1. **Voice Commands (Wake Word Detection)** – Activates the system via the Arabic wake word *"رشدي"*
2. **Face Recognition** – Identifies registered individuals using a from-scratch PyTorch ArcFace model
3. **Egyptian Currency Detection** – Detects, counts, and sums Egyptian banknotes with a YOLOv8 detector
4. **Roshdi Mobile App** – A Flutter app that bundles all three models and runs them offline on the phone

Each ML module is an independent subsystem that can be trained and evaluated on its own, then exported and embedded into the mobile app.

---

## Repository Structure

```
Roshdi graduation project/
│
├── Face Recognition/             # PyTorch face recognition pipeline (iResNet-18 + ArcFace)
│   ├── train.py                  # 2-stage CE → ArcFace training
│   ├── evaluate.py / evaluate_full.py
│   ├── inference.py              # embedding + gallery search
│   ├── realtime.py               # live multi-face recognition + tracking
│   ├── model.py                  # iResNet backbones + ArcFace head
│   ├── quantize.py               # INT8 dynamic quant + ONNX export
│   ├── docs/  reports/  scripts/  tests/
│   ├── requirements.txt
│   └── README.md
│
├── Egyptian Currency Detection/  # YOLOv8 banknote detection + counting
│   ├── src/
│   │   ├── config.py             # paths, classes, hyperparameters
│   │   ├── train.py              # YOLOv8 training
│   │   ├── detect.py             # inference: image / folder / video / webcam
│   │   ├── evaluate.py           # mAP / precision / recall on test split
│   │   ├── explain.py            # Eigen-CAM / Grad-CAM saliency + reasoning
│   │   ├── utils.py              # box drawing, count + total, JSON output
│   │   ├── bootstrap_yolo_dataset.py
│   │   ├── validators.py  logging_config.py  confidence_metrics.py
│   │   ├── profiling.py  model_registry.py  mlflow_history.py
│   ├── data_yolo/                # YOLO dataset (images + labels + data.yaml)
│   ├── outputs/                  # training/eval runs, detections, logs
│   ├── legacy/                   # previous CurrencyMobileNet classifier
│   ├── tests/                    # unit + adversarial tests
│   ├── *_REPORT.md/.pdf  MODEL_DESCRIPTION  FILE_GUIDE
│   ├── requirements.txt
│   └── README.md
│
├── Voice commands/               # Wake word detection ("رشدي")
│   ├── app.py                    # Flask dataset-cleaning / annotation server
│   ├── wake_word_final.ipynb     # Mel-spectrogram CNN training notebook
│   ├── templates/                # Web UI
│   ├── screenshots/
│   └── README.md
│
├── rushdey/                      # Flutter mobile app (Android / iOS / web / desktop)
│   ├── lib/main.dart
│   ├── assets/models/            # bundled exported models (.ptl / .tflite)
│   ├── android/  ios/  web/  windows/  macos/  linux/
│   ├── pubspec.yaml
│   └── test/
│
├── .gitignore
└── README.md                     # this file
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
             └──→ Detect, count & sum banknotes
```

The wake word activates the assistant, after which the requested model runs on-device and results are returned as audio feedback.

---

## 1. Face Recognition Module

**Location:** `Face Recognition/`

### Purpose

A **PyTorch** face recognition pipeline for the Roshdi assistive glasses. It performs:
- **1:1 Verification** (Is this person X?)
- **1:N Identification** (Who is this person?)

### Architecture

Built around an **iResNet-18** backbone trained **from scratch** on CASIA-WebFace using a 2-stage **CrossEntropy → ArcFace** strategy, then wrapped with multi-face tracking + smoothing for real-world robustness. A pretrained FaceNet/VGGFace2 model is bundled as a **baseline for comparison only** — the system runs on the from-scratch model by default.

| Attribute | Value |
|-----------|-------|
| Framework | PyTorch 2 |
| Backbone | iResNet-{18, 34, 50, 100} (default 18) |
| Training Head | ArcFace (Additive Angular Margin) |
| Training Strategy | 2-stage: CE warm-up → ArcFace |
| Input Size | 112×112 RGB (MTCNN + 5-pt alignment) |
| Embedding Size | 512-D normalized |
| Backbone Params | 24.0M (iResNet-18) |

### Headline Results

| Metric | From-scratch iResNet-18 (v4) | Pretrained FaceNet baseline |
|---|---:|---:|
| LFW mean accuracy (10-fold) | **79.65% ± 1.23%** | 99.25% ± 0.42% |
| LFW ROC AUC | 0.881 | 0.9995 |
| TP rate @ thr=0.45 (5 enrollments) | **100%** | 100% |
| FP rate @ thr=0.45 | **8%** | — |
| FP rate @ thr=0.50 | **2%** | — |
| Real-time loop latency (480p, 1 face) | ~25 ms / frame | ~30 ms |

The 79.65% LFW figure reflects a constrained training budget (1,000 of CASIA's 10,572 identities, ~10 effective epochs on Apple Silicon). The model is nonetheless production-ready at sensible thresholds (100% TP / 8% FP @ 0.45).

### Key Features

- **CLAHE on LAB-L channel** for lighting robustness
- **IoU + EMA multi-face tracker** with name voting and sticky labels
- Optional **horizontal-flip TTA** at inference
- **Novelty-gated rolling-window** online enrollment
- Edge export: **INT8 dynamic quantization + ONNX** (FP32/FP16)

### Usage

```bash
cd "Face Recognition"
pip install -r requirements.txt
python train.py            # train iResNet-18 (CE → ArcFace)
python evaluate.py         # / evaluate_full.py for full LFW protocol
python realtime.py         # live multi-face recognition
python quantize.py         # INT8 + ONNX export for the mobile app
```

For full details, see [Face Recognition/README.md](Face%20Recognition/README.md).

---

## 2. Egyptian Currency Detection Module

**Location:** `Egyptian Currency Detection/`

### Purpose

Detect, classify, **count, and sum** Egyptian banknotes in images, videos, or a live camera feed. Built on **YOLOv8 (Ultralytics)**, it outputs annotated media plus a JSON file with per-note detections, per-class counts, and the **total amount**.

> This replaces the previous single-image classifier (now kept under `legacy/`). A classifier could only label one note per frame; YOLOv8 detects and classifies *every* note in one pass — handling overlap and giving bounding boxes — which is what makes multi-note counting possible.

### Supported Classes

7 denominations (old/new variants of a note share one class, since they have the same value):

| ID | Class | Value | ID | Class | Value |
|----|-------|-------|----|-------|-------|
| 0 | 1_EGP | 1 | 4 | 50_EGP | 50 |
| 1 | 5_EGP | 5 | 5 | 100_EGP | 100 |
| 2 | 10_EGP | 10 | 6 | 200_EGP | 200 |
| 3 | 20_EGP | 20 | | | |

### Model & Results

| Attribute | Value |
|-----------|-------|
| Framework | YOLOv8 (Ultralytics) |
| Task | Object detection + counting |
| Detection mAP@0.5 | **0.982** |
| Counting accuracy (after synthetic fine-tune) | 18% → **88%** |
| Mobile export | TFLite / ONNX (`model.export(...)`) |

### Pipeline

```
Image / Video / Webcam
        │
        ▼
YOLOv8 Detector  ──→  per-note boxes + class + confidence
        │
        ▼
Count per class  +  Sum total EGP
        │
        ├──→ Annotated media (boxes + summary panel)
        └──→ JSON (detections, counts, total, metadata)
```

### Production Features (v2)

- **Input validation** (`validators.py`) — corrupted/oversized images return structured error JSON instead of crashing
- **Structured logging** (`logging_config.py`) — one JSONL event per detection under `outputs/logs/`
- **Confidence + risk flags** (`confidence_metrics.py`) — each detection gets a `risk` label and known-confusion warnings (e.g. 50 ↔ 100 EGP)
- **Visual explainability** (`explain.py`) — `--explain` produces an Eigen-CAM / Grad-CAM saliency heatmap + plain-language reasoning per note
- **JSON schema v2** (`utils.py`) — adds `metadata` (timestamp, model version, inference time, image size), normalized bboxes, confidence stats (legacy fields retained)
- **Performance profiling** (`profiling.py`), **model registry** (`model_registry.py`), and a **test suite** (`tests/`, unit + adversarial)
- **MLflow project history** (`mlflow_history.py`) — every milestone (v1 classifier → YOLOv8 → counting fix → hardening → explainability) reconstructed as comparable MLflow runs

### Usage

```bash
cd "Egyptian Currency Detection"
pip install -r requirements.txt              # pulls Ultralytics + a compatible torch

# Prepare dataset (bootstrap YOLO labels from the legacy classification folders)
python src/bootstrap_yolo_dataset.py --src "path/to/classification/data"

python src/train.py                          # train (or --epochs 50 --batch 8 --model yolov8s.pt)
python src/evaluate.py                        # mAP / precision / recall on test split

python src/detect.py --image photo.jpg        # single image  (--image-dir / --video / --webcam)
python src/detect.py --image photo.jpg --explain   # + saliency heatmap & reasoning
```

For full details, see [Egyptian Currency Detection/README.md](Egyptian%20Currency%20Detection/README.md).

---

## 3. Voice Commands — Wake Word Detection

**Location:** `Voice commands/`

### Purpose

Detects the Arabic wake word **"رشدي"** to activate the assistant. The module includes a web-based dataset-cleaning tool, a training pipeline, and an evaluation system.

### Dataset Cleaning Tool (`app.py`)

A Flask web application for collaborative labeling of wake-word audio samples:
- Multi-user labeling with persistent user IDs and file-assignment locking
- Automatic stale-assignment recovery (30-minute timeout)
- Real-time sample-count tracking, thread-safe concurrent access
- Keyboard shortcuts: `1` → Positive (wake word), `2` → Negative

### Training Pipeline (`wake_word_final.ipynb`)

Mel-spectrogram features → augmentation (pitch shift, time stretch, noise) → ResNet-style CNN with early stopping → evaluation + visualization → exported checkpoint.

| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
| Sample Rate | 16 kHz | Optimizer | Adam (lr 0.001) |
| Duration | 1 s | Loss | Weighted CrossEntropy |
| Mel Bands | 64 | Early Stopping | patience 15 |
| Input Shape | (64, 32) | Split | 80 / 10 / 10 |

### Model Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | **~95%+** |
| Macro F1-Score | **~95%** |
| Positive (رشدي) Precision / Recall | ~94% / ~96% |
| Negative Precision / Recall | ~96% / ~95% |

### Usage

```bash
cd "Voice commands"
python app.py                       # start Flask annotation server → http://localhost:5000
jupyter notebook wake_word_final.ipynb   # train the wake-word model
```

For full details, see [Voice commands/README.md](Voice%20commands/README.md).

---

## 4. Roshdi Mobile App (Flutter)

**Location:** `rushdey/`

The cross-platform Flutter application that unifies all three modules and runs them **offline, on-device**. Built with the `pytorch_lite` plugin, it bundles the exported models as assets:

| Asset | Module |
|-------|--------|
| `assets/models/wake_word_trial2.ptl` + `config.json` | Voice Commands |
| `assets/models/face_model.tflite` | Face Recognition |
| `assets/models/currency_model.ptl` / `best.ptl` + `labels.txt` | Currency Detection |

**Stack:** Flutter (Dart SDK ^3.11.1), `pytorch_lite ^4.3.2`, `image`, `shared_preferences`. Targets **Android, iOS, web, Windows, macOS, and Linux**.

### Build & Run

```bash
cd rushdey
flutter pub get
flutter run            # run on a connected device / emulator
flutter build apk      # Android release build (or: flutter build ios / web / windows)
```

> The app loads pre-trained models from `assets/models/`. After retraining a module, export it (`quantize.py` / `model.export(...)`) and replace the corresponding asset, then update `pubspec.yaml` if filenames change.

---

## System Requirements

### Python (ML modules)
- **Python** ≥ 3.8 — **CUDA** optional (GPU recommended)
- Core libs: `torch`, `torchaudio`, `torchvision`, `ultralytics` (YOLOv8), `tensorflow` (FR baseline export), `flask`, `librosa`, `scikit-learn`, `numpy`, `matplotlib`, `jupyter`, `pandas`, `soundfile`, `onnx`, `mlflow`

Each module ships its own `requirements.txt`:

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r "Face Recognition/requirements.txt"
pip install -r "Egyptian Currency Detection/requirements.txt"
```

### Mobile (Roshdi app)
- **Flutter SDK** with Dart ^3.11.1 — run `flutter doctor` to verify your toolchain

---

## Project Information

| Field | Details |
|-------|---------|
| **Project Name** | Beyond The Limits (Roshdi / رشدي) |
| **Objective** | Assistive AI for visually impaired users |
| **Type** | Graduation Project |
| **ML Modules** | 3 — Face Recognition, Currency Detection (YOLOv8), Voice Commands |
| **App** | Flutter cross-platform mobile/desktop (`rushdey/`) |
| **Frameworks** | PyTorch, Ultralytics YOLOv8, TensorFlow/Keras, Flask, Flutter |
| **Target Platform** | Android / iOS (+ web & desktop) |
| **Model Export** | TorchScript Lite (.ptl), TensorFlow Lite, ONNX, INT8 |

---

## Future Integration

The Roshdi app brings the three modules together into a unified assistant that:

- ✓ Listens for the wake word (*"رشدي"*)
- ✓ Identifies people via face recognition
- ✓ Detects, counts, and sums Egyptian banknotes
- ✓ Responds to voice queries with audio feedback

Enabling visually impaired users to identify who they're talking to, determine the value of banknotes they're holding, and control the device entirely by voice.

---

## License & Attribution

This is an academic graduation project. See individual module READMEs for specific licensing and attribution.

---

**Last Updated:** June 2026
