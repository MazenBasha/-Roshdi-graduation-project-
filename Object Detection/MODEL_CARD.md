# Model Card — Roshdi Object Detector (Pretrained YOLO11n)

## Overview
| Field | Value |
|---|---|
| Architecture | YOLO11n (Ultralytics), nano scale |
| Parameters | ≈ 2.6 M |
| FLOPs @ 640×640 | ≈ 6.5 G |
| Weights | **Pretrained**, used as-is — Ultralytics-released `yolo11n.pt` |
| Training data | Microsoft COCO `train2017` (≈ 118 k images, 80 classes) — performed by Ultralytics |
| Fine-tuning by Roshdi | **None.** No further training in our pipeline. |
| Input | RGB image, letterboxed to 640 × 640, pixels in [0, 1] |
| Output | List of (bounding box, class, confidence) over the 80 COCO classes |
| License | AGPL-3.0 (Ultralytics) |
| Version identifier | First 12 chars of the SHA-256 of the served `.pt` file (surfaced in every response as `model_version`) |

## Intended Use
- **Primary**: real-time perception assistance for visually-impaired Roshdi
  users (*"there is a chair on your right, near"*). Single-user, on-device or
  via the local HTTP service.
- **Secondary**: cloud HTTP fallback for development and remote testing.

## Out-of-Scope Use
- Surveillance, person identification, behaviour analysis. YOLO11n only emits
  the COCO class `person`; identity matching is the separate Roshdi Face
  Recognition module's job and is governed by that module's policies.
- Safety-critical autonomy (medical, driving). This is an **assistive** signal,
  not a guarantee.

## Why pretrained-only?
Roshdi's user need maps exactly onto the 80 COCO categories that
`yolo11n.pt` was trained on (people, vehicles, furniture, food, kitchenware,
electronics, …). The Ultralytics-released checkpoint already achieves
mAP@0.50:0.95 ≈ 0.39 / mAP@0.50 ≈ 0.55 on COCO `val2017` — a strong baseline
for an assistive use case where false positives are dampened by the voice
layer's confidence cut. Re-running 30 epochs on the same data and label space
would, in expectation, recover the same accuracy at considerable energy and
time cost. We therefore **deploy the pretrained checkpoint directly** and
reserve fine-tuning for the future case where Roshdi collects a custom
domain-specific dataset (e.g. Egyptian street scenes).

## Performance — published reference (Ultralytics)
| Metric | Value (FP32, 640×640) |
|---|---|
| mAP@0.50:0.95 (COCO val2017) | ≈ 0.39 |
| mAP@0.50 | ≈ 0.55 |
| Params | 2.6 M |
| FLOPs | 6.5 G |

Mobile latency expectations after INT8 export:

| Target | Expected end-to-end | Notes |
|---|---|---|
| iPhone Neural Engine (CoreML INT8) | 15–25 ms / frame | 40+ FPS, headroom for the voice loop |
| Android NPU (TFLite INT8 / NNAPI) | 20–40 ms / frame | 25–50 FPS on modern flagships |
| 4-vCPU cloud server (FP32 ONNX) | 60–90 ms | server fallback only — not the primary path |

## Failure Modes & Limitations
- **Long-tail classes** (`hair drier`, `toothbrush`) have far fewer training
  instances and lower AP. The voice layer can speak *"uncertain object"* when
  confidence < 0.5 for these classes.
- **Small objects**: a known weakness of nano detectors at 640×640. Very-far
  objects may be missed entirely.
- **Heavy occlusion / unusual viewpoints** degrade accuracy.
- **Quantisation cost**: INT8 mobile artefacts give up ~1–3 mAP50-95 versus the
  FP32 baseline in exchange for ~4× compression and NPU acceleration.
- **No temporal modelling**: every frame is processed independently. The
  Roshdi voice layer is responsible for de-duplicating consecutive
  announcements.
- **Domain bias**: COCO over-represents Western indoor/urban contexts.
  Per-class accuracy on Egyptian street scenes may be lower than the
  published COCO numbers; the optional fine-tuning pathway exists for that
  case.

## Calibration / Confidence Semantics
Sigmoid score per class. The API default cut is `conf ≥ 0.25`; the Flutter
caller raises this to `0.35` for spoken output to avoid noisy announcements.
**Confidence is not a calibrated probability** — treat it ordinally within
a session, not as an absolute risk.

## Safety
- **PII**: COCO contains no PII labels. The cloud server can optionally
  pixelate the head region of `person` detections
  (`ROSHDI_OD_BLUR_PERSONS=true`, `server/safety.py`).
- **Bias**: see *Domain bias* above. Documented mitigation: targeted
  fine-tune via `kaggle_pipeline.ipynb` / `train.py`.

## Versioning
The wire identifier is the SHA-256 (first 12 chars) of the loaded weights
file, surfaced in every response as `model_version` and at `/healthz`.
Rolling forward to a new pretrained release or rolling back to a previous one
is a single `weights/` file swap + restart — no code change.

## Maintenance
- Drift monitored via `drift.py` against
  `results/baseline_distribution.json`.
- **Retraining trigger** (if/when we move off the pretrained path):
  PSI > 0.25 on any class or > 0.10 overall, **or** the Roshdi user-feedback
  loop logs > 5 % corrections for a class for two weeks.
