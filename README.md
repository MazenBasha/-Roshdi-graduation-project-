Beyond The Limits — Roshdi Assistive AI System

A multi-modal AI system designed to assist visually impaired users by providing real-time perception through voice commands, face recognition, Egyptian currency detection, and obstacle / object detection — all delivered through a single cross-platform Flutter mobile app (Roshdi / رشدي).

The system integrates four independent deep-learning modules plus the mobile application that runs them on-device:


Voice Commands (Wake Word Detection) – Activates the system via the Arabic wake word "رشدي"
Face Recognition – Identifies registered individuals using a from-scratch PyTorch ArcFace model
Egyptian Currency Detection – Detects, counts, and sums Egyptian banknotes with a YOLOv8 detector
Object Detection (Obstacle Detection) – Detects nearby objects with a YOLO11n model and turns them into voice-friendly proximity warnings (English + Arabic)
Roshdi Mobile App – A Flutter app that bundles all four models and runs them offline on the phone


Each ML module is an independent subsystem that can be trained and evaluated on its own, then exported and embedded into the mobile app.


Repository Structure

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
├── Object Detection/             # YOLO11n obstacle / proximity detection
│   ├── live_camera_proximity.py  # webcam test: detection → proximity → spoken warning (EN/AR)
│   ├── requirements.txt
│   └── README_LOCAL_TEST.md
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


System Overview

Workflow

Wake Word ("رشدي") Detected
        │
        ▼
Voice Command Processing (Vosk)
        │
        ├──→ Face Recognition
        │    │
        │    └──→ Identify person in camera
        │
        ├──→ Currency Detection
        │    │
        │    └──→ Detect, count & sum banknotes
        │
        └──→ Obstacle Detection
             │
             └──→ Real-time visual-proximity warnings

The wake word activates the assistant, after which the requested model runs on-device and results are returned as audio feedback.


1. Face Recognition Module

Location: Face Recognition/

Purpose

A PyTorch face recognition pipeline for the Roshdi assistive glasses. It performs:


1:1 Verification (Is this person X?)
1:N Identification (Who is this person?)


Architecture

Built around an iResNet-18 backbone trained from scratch on CASIA-WebFace using a 2-stage CrossEntropy → ArcFace strategy, then wrapped with multi-face tracking + smoothing for real-world robustness. A pretrained FaceNet/VGGFace2 model is bundled as a baseline for comparison only — the system runs on the from-scratch model by default.

AttributeValueFrameworkPyTorch 2BackboneiResNet-{18, 34, 50, 100} (default 18)Training HeadArcFace (Additive Angular Margin)Training Strategy2-stage: CE warm-up → ArcFaceInput Size112×112 RGB (MTCNN + 5-pt alignment)Embedding Size512-D normalizedBackbone Params24.0M (iResNet-18)

Headline Results

MetricFrom-scratch iResNet-18 (v4)Pretrained FaceNet baselineLFW mean accuracy (10-fold)79.65% ± 1.23%99.25% ± 0.42%LFW ROC AUC0.8810.9995TP rate @ thr=0.45 (5 enrollments)100%100%FP rate @ thr=0.458%—FP rate @ thr=0.502%—Real-time loop latency (480p, 1 face)~25 ms / frame~30 ms

The 79.65% LFW figure reflects a constrained training budget (1,000 of CASIA's 10,572 identities, ~10 effective epochs on Apple Silicon). The model is nonetheless production-ready at sensible thresholds (100% TP / 8% FP @ 0.45).

Key Features


CLAHE on LAB-L channel for lighting robustness
IoU + EMA multi-face tracker with name voting and sticky labels
Optional horizontal-flip TTA at inference
Novelty-gated rolling-window online enrollment
Edge export: INT8 dynamic quantization + ONNX (FP32/FP16)


Usage

cd "Face Recognition"
pip install -r requirements.txt
python train.py            # train iResNet-18 (CE → ArcFace)
python evaluate.py         # / evaluate_full.py for full LFW protocol
python realtime.py         # live multi-face recognition
python quantize.py         # INT8 + ONNX export for the mobile app

For full details, see Face Recognition/README.md.


2. Egyptian Currency Detection Module

Location: Egyptian Currency Detection/

Purpose

Detect, classify, count, and sum Egyptian banknotes in images, videos, or a live camera feed. Built on YOLOv8 (Ultralytics), it outputs annotated media plus a JSON file with per-note detections, per-class counts, and the total amount.


This replaces the previous single-image classifier (now kept under legacy/). A classifier could only label one note per frame; YOLOv8 detects and classifies every note in one pass — handling overlap and giving bounding boxes — which is what makes multi-note counting possible.



Supported Classes

7 denominations (old/new variants of a note share one class, since they have the same value):

IDClassValueIDClassValue01_EGP1450_EGP5015_EGP55100_EGP100210_EGP106200_EGP200320_EGP20

Model & Results

AttributeValueFrameworkYOLOv8 (Ultralytics)TaskObject detection + countingDetection mAP@0.50.982Counting accuracy (after synthetic fine-tune)18% → 88%Mobile exportTFLite / ONNX (model.export(...))

Pipeline

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

Production Features (v2)


Input validation (validators.py) — corrupted/oversized images return structured error JSON instead of crashing
Structured logging (logging_config.py) — one JSONL event per detection under outputs/logs/
Confidence + risk flags (confidence_metrics.py) — each detection gets a risk label and known-confusion warnings (e.g. 50 ↔ 100 EGP)
Visual explainability (explain.py) — --explain produces an Eigen-CAM / Grad-CAM saliency heatmap + plain-language reasoning per note
JSON schema v2 (utils.py) — adds metadata (timestamp, model version, inference time, image size), normalized bboxes, confidence stats (legacy fields retained)
Performance profiling (profiling.py), model registry (model_registry.py), and a test suite (tests/, unit + adversarial)
MLflow project history (mlflow_history.py) — every milestone (v1 classifier → YOLOv8 → counting fix → hardening → explainability) reconstructed as comparable MLflow runs


Usage

cd "Egyptian Currency Detection"
pip install -r requirements.txt              # pulls Ultralytics + a compatible torch

# Prepare dataset (bootstrap YOLO labels from the legacy classification folders)
python src/bootstrap_yolo_dataset.py --src "path/to/classification/data"

python src/train.py                          # train (or --epochs 50 --batch 8 --model yolov8s.pt)
python src/evaluate.py                        # mAP / precision / recall on test split

python src/detect.py --image photo.jpg        # single image  (--image-dir / --video / --webcam)
python src/detect.py --image photo.jpg --explain   # + saliency heatmap & reasoning

For full details, see Egyptian Currency Detection/README.md.


3. Voice Commands — Wake Word Detection

Location: Voice commands/

Purpose

Detects the Arabic wake word "رشدي" to activate the assistant. The module includes a web-based dataset-cleaning tool, a training pipeline, and an evaluation system.

Dataset Cleaning Tool (app.py)

A Flask web application for collaborative labeling of wake-word audio samples:


Multi-user labeling with persistent user IDs and file-assignment locking
Automatic stale-assignment recovery (30-minute timeout)
Real-time sample-count tracking, thread-safe concurrent access
Keyboard shortcuts: 1 → Positive (wake word), 2 → Negative


Training Pipeline (wake_word_final.ipynb)

Mel-spectrogram features → augmentation (pitch shift, time stretch, noise) → ResNet-style CNN with early stopping → evaluation + visualization → exported checkpoint.

ParameterValueParameterValueSample Rate16 kHzOptimizerAdam (lr 0.001)Duration1 sLossWeighted CrossEntropyMel Bands64Early Stoppingpatience 15Input Shape(64, 32)Split80 / 10 / 10

Model Performance

MetricValueOverall Accuracy~95%+Macro F1-Score~95%Positive (رشدي) Precision / Recall~94% / ~96%Negative Precision / Recall~96% / ~95%

Usage

cd "Voice commands"
python app.py                       # start Flask annotation server → http://localhost:5000
jupyter notebook wake_word_final.ipynb   # train the wake-word model

For full details, see Voice commands/README.md.


4. Object Detection Module

Location: Object Detection/

Purpose

Real-time obstacle / proximity detection that helps visually impaired users navigate around nearby objects. It turns YOLO detections into voice-friendly proximity warnings (English + Arabic). It does not estimate distance in meters — it estimates visual proximity from the normalized bounding-box size inside the camera frame.

Architecture

Built on YOLO11n (Ultralytics) pretrained on the Microsoft COCO 80-class dataset. Each detection is enriched with a horizontal_position (left / center / right) and a distance_hint (far / near / very_near), then passed through smoothing and cooldown logic before a warning is spoken.

AttributeValueFrameworkUltralytics YOLO11nClasses80 (COCO)Processing rate5 FPS (configurable)Proximity metricnormalized bbox area_ratio + height_ratioPositionleft / center / right (frame thirds)OutputEnglish + Arabic warning messages, structured JSONSpeechoptional pyttsx3 text-to-speech (en / ar)

Proximity Logic

Instead of one global threshold, objects are grouped so a car, a chair, and a bottle are each judged by size-appropriate thresholds:


Large danger: person, car, bus, truck, motorcycle, bicycle
Medium obstacle: chair, bench, couch, dining table, suitcase, backpack
Small object: bottle, cup, cell phone, book, remote, mouse, keyboard
Default: all other COCO classes


Robustness features: confidence filtering, temporal smoothing across the last 3 processed frames (an alert needs ≥ 2 hits), priority selection (very-near and danger classes first), and a 2-second cooldown to avoid repeating the same warning.

Pipeline

Live Camera / Webcam
        │
        ▼
YOLO11n Detector  ──→  per-object boxes + class + confidence
        │
        ▼
Proximity analysis  (area_ratio + height_ratio → far / near / very_near)
        │
        ├──→ position (left / center / right) + priority selection
        ▼
Smoothing + cooldown  ──→  spoken warning (EN / AR) + structured JSON

Usage

cd "Object Detection"
pip install -r requirements.txt
python live_camera_proximity.py                                   # local webcam test
python live_camera_proximity.py --weights yolo11n.pt --conf 0.40 --process-fps 5
python live_camera_proximity.py --speak --lang ar                 # spoken Arabic warnings
python live_camera_proximity.py --print-json                      # structured output

Press Q in the camera window to stop.

For full details, see Object Detection/README_LOCAL_TEST.md.


5. Roshdi Mobile App (Flutter)

Location: rushdey/

The cross-platform Flutter application that unifies all modules and runs them offline, on-device. The app relies heavily on custom native Android (Kotlin) ML engines directly calling PyTorch Mobile and TensorFlow Lite to ensure optimal performance and avoid Flutter-plugin tensor issues.

AssetModuleassets/models/wake_word_trial2.ptl + config.jsonVoice Commandsassets/models/face_model.tfliteFace Recognitionassets/models/currency_model.ptl / best.ptl + labels.txtCurrency Detectionassets/models/object_detector.ptlObstacle Detection

Key Native Kotlin Engines:


WakeWordDetector / VoskIntentEngine: Robust fuzzy matching for the wake word and Vosk Arabic ASR for offline intent routing.
CurrencyEngine: Native PyTorch Mobile implementation handling both YOLOv8 boxes and classifier logits.
ObjectDetectionEngine: YOLO-based real-time obstacle detection engine providing distance hints.
FaceEngine: TFLite-based embedding extractor and matcher.


Stack: Flutter (Dart SDK ^3.11.1), Kotlin, PyTorch Mobile, TensorFlow Lite, Vosk. Targets Android primarily for native ML engines, with Flutter UI.

Build & Run

cd rushdey
flutter pub get
flutter run            # run on a connected device / emulator
flutter build apk      # Android release build (or: flutter build ios / web / windows)


The app loads pre-trained models from assets/models/. After retraining a module, export it (quantize.py / model.export(...)) and replace the corresponding asset, then update pubspec.yaml if filenames change.




System Requirements

Python (ML modules)


Python ≥ 3.8 — CUDA optional (GPU recommended)
Core libs: torch, torchaudio, torchvision, ultralytics (YOLOv8 / YOLO11), tensorflow (FR baseline export), flask, librosa, scikit-learn, numpy, matplotlib, jupyter, pandas, soundfile, onnx, mlflow, pyttsx3 (Object Detection speech)


Each module ships its own requirements.txt:

python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r "Face Recognition/requirements.txt"
pip install -r "Egyptian Currency Detection/requirements.txt"
pip install -r "Object Detection/requirements.txt"

Mobile (Roshdi app)


Flutter SDK with Dart ^3.11.1 — run flutter doctor to verify your toolchain



Project Information

FieldDetailsProject NameBeyond The Limits (Roshdi / رشدي)ObjectiveAssistive AI for visually impaired usersTypeGraduation ProjectML Modules4 — Face Recognition, Currency Detection (YOLOv8), Voice Commands, Object Detection (YOLO11n)AppFlutter cross-platform mobile/desktop (rushdey/)FrameworksPyTorch, Ultralytics YOLOv8 / YOLO11, TensorFlow/Keras, Flask, FlutterTarget PlatformAndroid / iOS (+ web & desktop)Model ExportTorchScript Lite (.ptl), TensorFlow Lite, ONNX, INT8


Future Integration

The Roshdi app brings the modules together into a unified assistant that:


✓ Listens for the wake word ("رشدي") via fuzzy matching
✓ Processes voice commands locally via Vosk Arabic ASR
✓ Identifies people via face recognition
✓ Detects, counts, and sums Egyptian banknotes
✓ Provides real-time obstacle detection and walking assistance
✓ Responds to voice queries with audio feedback


Enabling visually impaired users to identify who they're talking to, determine the value of banknotes they're holding, navigate around obstacles, and control the device entirely by voice.


License & Attribution

This is an academic graduation project. See individual module READMEs for specific licensing and attribution.

