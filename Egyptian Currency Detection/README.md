# Egyptian Currency Detection (YOLOv8)

Detect, classify, count, and sum Egyptian banknotes in images, videos, or a
live camera feed. Built on YOLOv8 (ultralytics). Outputs annotated media plus
a JSON file with per-note detections, per-class counts, and the total amount.

Supports 7 denominations: **1, 5, 10, 20, 50, 100, 200 EGP**.

## Project layout

```
Egyptian Currency Detection/
├── src/
│   ├── config.py                  # all paths, classes, hyperparameters
│   ├── train.py                   # YOLOv8 training
│   ├── detect.py                  # inference: image / folder / video / webcam
│   ├── evaluate.py                # mAP / precision / recall on test split
│   ├── explain.py                 # Eigen-CAM / Grad-CAM saliency + reasoning
│   ├── utils.py                   # box drawing, count + total, JSON output
│   └── bootstrap_yolo_dataset.py  # convert classification folders -> YOLO format
├── data_yolo/
│   ├── data.yaml        # dataset spec (edit only if you rename classes)
│   ├── images/{train,val,test}/   # *.jpg
│   └── labels/{train,val,test}/   # *.txt (YOLO format)
├── outputs/
│   ├── runs/            # training + eval runs (weights, plots)
│   └── detections/      # annotated images / videos + JSON files
├── legacy/              # the previous classification-only project
├── requirements.txt
└── README.md
```

## Install

```powershell
pip install -r requirements.txt
```

`ultralytics` will pull a compatible torch build automatically. For GPU,
install the matching CUDA torch wheel from https://pytorch.org first.

## 1. Prepare the dataset

### Quick start: bootstrap from the existing classification dataset

If you already have the classification-style dataset
(`<root>/{train,valid,test}/<class>/*.jpg`), generate a YOLO version with
pseudo-labels (whole image = one box) in one command:

```powershell
python src/bootstrap_yolo_dataset.py --src "e:/data/egyptian_currency/data"
```

This copies images into `data_yolo/images/{train,val,test}/` and writes a
matching `.txt` per image with a full-image bounding box. Old/new variants of
the same denomination are merged (`10` + `10 (new)` → `10_EGP`). Result:

- 2637 train / 760 val / 290 test
- 7 classes mapped from 9 source folders

You can train immediately with this. **But heads up:** every label is a
single full-image box, so the model will learn to detect single notes well,
not overlapping or multiple notes in one frame. To get real multi-note
performance, re-label a subset (especially multi-note shots) by hand —
section below.

### Real bounding-box labels (recommended for production accuracy)

YOLO needs **bounding boxes**, not just folders. For each training image, you
write a `.txt` file with one line per visible note:

```
<class_id> <x_center> <y_center> <width> <height>
```

All four coords are floats in `[0, 1]`, normalized by image width / height.

**Class IDs (must match `data_yolo/data.yaml`):**

| ID | Class    | Value |
|----|----------|-------|
| 0  | 1_EGP    | 1     |
| 1  | 5_EGP    | 5     |
| 2  | 10_EGP   | 10    |
| 3  | 20_EGP   | 20    |
| 4  | 50_EGP   | 50    |
| 5  | 100_EGP  | 100   |
| 6  | 200_EGP  | 200   |

> Note: old vs. new note variants of the same denomination share one class —
> they have the same monetary value, which is what the counter cares about.

**Easiest tooling:**
- [Roboflow](https://roboflow.com) — upload images, draw boxes in the browser,
  export as "YOLOv8". Drop the exported `images/` and `labels/` folders into
  `data_yolo/`.
- [LabelImg](https://github.com/HumanSignal/labelImg) — desktop tool, set
  format to "YOLO".

**How much to label:** aim for ~80 / 10 / 10 split between train / val / test.
Realistic minimum is ~50 images per class for a working demo, ~150+ per class
for solid accuracy. Mix single-note shots with multi-note scenes (overlapping,
folded, bad lighting) so the detector learns to count properly.

After labeling, your tree should look like:

```
data_yolo/
  images/train/img_0001.jpg
  labels/train/img_0001.txt
  images/val/...
  labels/val/...
  images/test/...
  labels/test/...
  data.yaml
```

## 2. Train

```powershell
python src/train.py
# or with custom settings:
python src/train.py --epochs 50 --batch 8 --model yolov8s.pt
# resume the last interrupted run:
python src/train.py --resume
```

Training augmentations (set in `src/train.py`) are tuned for hand-held
currency photos: HSV jitter (lighting), rotation ±20°, perspective, scale,
translate, mosaic (synthesizes multi-note scenes from single notes), and
mixup. `flipud` is kept low because notes are rarely upside-down in the wild.

Output:
```
outputs/runs/train/weights/best.pt
outputs/runs/train/weights/last.pt
outputs/runs/train/results.png         # loss + mAP curves
outputs/runs/train/confusion_matrix.png
```

## 3. Evaluate

```powershell
python src/evaluate.py                 # uses best.pt on test split
python src/evaluate.py --split val
```

Reports mAP@0.5, mAP@0.5:0.95, precision, recall, per-class breakdown, and
saves a confusion matrix and PR curves under `outputs/runs/eval/`.

## 4. Detect

### Single image
```powershell
python src/detect.py --image path/to/photo.jpg
```

### Folder of images
```powershell
python src/detect.py --image-dir path/to/folder
```

### Video file
```powershell
python src/detect.py --video path/to/clip.mp4
```

### Webcam
```powershell
python src/detect.py --webcam
python src/detect.py --webcam --camera 1     # second camera
```

### Tuning
```powershell
python src/detect.py --image x.jpg --conf 0.5 --iou 0.45
python src/detect.py --image x.jpg --weights outputs/runs/train2/weights/best.pt
```

`--no-show` runs without an OpenCV preview window (useful on servers).

### Explainability (why did the model output this?)
```powershell
python src/detect.py --image x.jpg --explain
python src/detect.py --image x.jpg --explain --explain-method gradcam
```
`--explain` writes an extra `<stem>_explain.jpg` saliency heatmap and adds a
`salience_ratio` + plain-language `explanation` to every detection in the JSON,
on top of the existing `confidence` / `risk` / `warning` fields. Two methods:

- **`eigencam`** (default) — robust, class-agnostic saliency from the principal
  component of the Detect-head feature maps. Clean, well-localized heatmaps; the
  recommended choice for reports and screenshots.
- **`gradcam`** — class-discriminative Grad-CAM tied to each detection's
  IoU-matched anchors ("why *this* denomination"). More faithful to the class
  decision, but can be diffuse on YOLO's anchor-free head.

`salience_ratio` is mean(heatmap inside the box) / mean(heatmap overall): `>1`
means the model attends to the note more than the image average; `~1` is normal
when a note fills the frame; `<1` flags reliance on surrounding context.

## Output format

For every input image / frame the detector saves:

1. An **annotated image / video** with bounding boxes, class label, confidence,
   and a top-left summary panel listing per-class counts and the total.
2. A **JSON** file matching this schema:

```json
{
  "detections": [
    {"class": "100_EGP", "confidence": 0.94, "bbox": [x1, y1, x2, y2]},
    {"class": "100_EGP", "confidence": 0.91, "bbox": [x1, y1, x2, y2]},
    {"class": "50_EGP",  "confidence": 0.88, "bbox": [x1, y1, x2, y2]}
  ],
  "counts": { "100_EGP": 2, "50_EGP": 1 },
  "total": 250
}
```

Bounding boxes are in absolute image pixel coordinates (`x1, y1` = top-left,
`x2, y2` = bottom-right).

## Tips for higher accuracy

- **Label quality > label quantity.** Tight, consistent boxes beat thousands
  of loose ones. Box exactly the printed paper area, not the table around it.
- **Cover hard cases in training:** overlapping notes, partial occlusion,
  folded notes, both faces of the bill, low light, blurry photos.
- **Watch the confusion matrix** after training. If 10 ↔ 20 confusion is
  high, add more examples of those two classes in similar lighting.
- **Adjust `--conf`** at inference: lower (e.g. `0.25`) finds more notes but
  raises false positives; raise it (e.g. `0.5`) for cleaner, stricter
  detections.
- For mobile deployment, export with `model.export(format="tflite")` or
  `format="onnx"` after training.

## New Features (v2.0)

Production hardening that ships alongside the same single CLI - no API
servers, no architectural changes, fully backward compatible.

- **Robust input validation** (`src/validators.py`) - missing, corrupted, or
  oversized images return a structured error JSON instead of crashing.
- **Structured logging** (`src/logging_config.py`) - one JSONL line per event
  written to `outputs/logs/detection_<timestamp>.jsonl`, plus human-readable
  console output.
- **Confidence + risk flags** (`src/confidence_metrics.py`) - every detection
  gets a `risk` label (`low`/`medium`/`high`/`reject`) and known-confusion
  classes (50 ↔ 100 EGP, rare 1 EGP) get a `warning` field.
- **Visual explainability** (`src/explain.py`) - `--explain` produces an
  Eigen-CAM / Grad-CAM saliency heatmap plus a per-detection `salience_ratio`
  and plain-language `explanation`, answering *why* the model output each note.
- **Enhanced JSON schema v2** (`src/utils.py`) - includes `metadata`
  (timestamp, model version, inference time, image size), normalized bboxes,
  and confidence statistics. Legacy `detections` / `counts` / `total` fields
  are still present.
- **Performance profiling** (`src/profiling.py`) - `PerformanceProfiler`
  benchmarks single calls or batches (latency, memory, throughput).
- **Model registry** (`src/model_registry.py`) - track trained models in
  `outputs/model_registry.json`, switch the active model, roll back safely.
- **Test suite** (`tests/`) - unit tests for validators, utils, confidence;
  adversarial tests for extreme brightness, blur, rotation, etc.

## Project history (MLflow)

The full evolution of the project — from the v1 classification baseline through
the YOLOv8 switch, the multi-note counting fix, production hardening, and the
explainability feature — is reconstructed as MLflow runs so it can be browsed
and compared on one timeline.

```powershell
# Log every milestone as an MLflow run (idempotent — safe to re-run)
python src/mlflow_history.py

# Browse the timeline in the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# then open http://127.0.0.1:5000
```

Each run carries the design **params** (architecture, classes, epochs…), the
reported **metrics** (accuracy / mAP / counting rates), and a **description**
summarizing what changed and why. Milestones logged:

| Version | Date | Change | Native metric | Headline score |
|---|---|---|---|---|
| v1.0 | 2026-03-07 | CurrencyMobileNet classification baseline | 94.14% test acc | 0.557 |
| v2.0 | 2026-05-02 | Switch to YOLOv8 multi-note detection | mAP50 0.982 | 0.663 |
| v2.1 | 2026-05-21 | Synthetic-data fine-tune (counting fix) | count acc 18%→88% | 0.856 |
| prod-2.0 | 2026-06-18 | Production hardening (validation, logging, registry) | 37 tests | 0.931 |
| v2.2 | 2026-06-18 | Visual explainability (Eigen-CAM / Grad-CAM) | 53 tests | 0.956 |

**`headline_score`** is one normalized 0–1 line comparable across the
classification→detection switch (where raw accuracy and mAP aren't). It blends
three logged pillars — `0.40·model_quality + 0.35·task_capability +
0.25·production_readiness` — so you can chart the single trend or its breakdown
in the UI. Weights and per-pillar values are defined transparently in
`src/mlflow_history.py`.

### New commands

```powershell
# Run the full test suite
pytest tests/ -v

# Register a trained model
python -c "from src.model_registry import ModelRegistry; ModelRegistry.register_model('outputs/runs/train_finetune/weights/best.pt', {'mAP50': 0.995, 'mAP50_95': 0.989, 'precision': 0.991, 'recall': 0.964}, notes='Original training run')"

# List registered models
python -c "from src.model_registry import ModelRegistry; ModelRegistry.list_models()"

# Tail structured logs (requires jq)
Get-Content outputs/logs/detection_*.jsonl -Wait | ForEach-Object { $_ | jq }
```

### Updated JSON output (excerpt)

```json
{
  "schema_version": "2.0",
  "metadata": {
    "timestamp": "2026-06-18T14:32:00.123",
    "image": "test.jpg",
    "model_version": "outputs/runs/train_finetune/weights/best.pt",
    "inference_time_ms": 27.3,
    "image_size": [1920, 1080]
  },
  "detections": [
    {"id": 0, "class": "100_EGP", "confidence": 0.94, "risk": "low",
     "bbox": [100,150,250,350], "bbox_normalized": [0.052,0.139,0.130,0.324],
     "area_pixels": 20000}
  ],
  "summary": {
    "num_detections": 1, "counts": {"100_EGP": 1}, "total_egp": 100,
    "confidence": {"min": 0.94, "max": 0.94, "mean": 0.94, "std": 0.0},
    "high_risk_count": 0
  },
  "counts": {"100_EGP": 1},
  "total": 100,
  "confidence_report": { "...": "..." }
}
```

## Why YOLOv8 (and not the previous classifier)?

The previous project (`legacy/`) was a single-image classifier — it could
only output one label for the whole frame. That makes counting multiple notes
in one photo impossible without bolting on a fragile region-proposal step.
YOLOv8 detects and classifies every note in one pass, handles overlap, gives
us bounding boxes for free, and is evaluated with proper detection metrics
(mAP, P/R) instead of just classification accuracy.
