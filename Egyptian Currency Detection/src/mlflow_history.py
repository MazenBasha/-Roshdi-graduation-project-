"""
Reconstruct the full project history as MLflow runs.

This logs one MLflow run per project milestone — from the original
classification baseline through the YOLOv8 switch, the multi-note counting fix,
production hardening, and the explainability feature — so the whole evolution
of the project is browsable, comparable, and chronologically ordered in the
MLflow UI.

It is a *documentation* tool: the numbers come from the project reports
(Progress_Report, PROJECT_REPORT, UPDATE_REPORT, PRODUCTION_REPORT,
MODEL_DESCRIPTION) and the model registry — not from re-training. Re-running the
script is idempotent: it clears prior runs in the experiment and re-logs fresh.

Usage:
    python src/mlflow_history.py            # log all milestones
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse at :5000

Each run carries:
- params   : the design choices at that point (architecture, classes, epochs...)
- metrics  : the accuracy / mAP / counting numbers reported at that milestone
- tags     : version, date, approach, status, and a `mlflow.note.content`
             description summarizing what changed and why.
"""

import os
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT = "Egyptian-Currency-Detection-History"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Keep the MLflow store inside the project so it travels with the repo. Recent
# MLflow retired the bare-filesystem store, so we use a local SQLite backend.
DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db").replace("\\", "/")
TRACKING_URI = f"sqlite:///{DB_PATH}"

# ─── Normalized headline score ───────────────────────────────────────────────
# The native metrics aren't comparable across the v1→v2 classification→detection
# switch (accuracy vs mAP). To get ONE line that tracks overall project maturity,
# each milestone is scored on three 0–1 pillars and blended with these weights:
#
#   headline_score = 0.40*model_quality + 0.35*task_capability + 0.25*readiness
#
# - model_quality      : the milestone's own core metric (test accuracy for the
#                        classifier, mAP50 for the detectors) — both already 0–1.
# - task_capability    : fraction of the END goal delivered (reliably locate +
#                        count + total multiple notes). Anchored to measured
#                        multi-note exact-count accuracy in the detection era.
# - production_readiness: validation / logging / tests / explainability maturity.
#
# All four numbers (the three pillars + the blend) are logged as metrics, so the
# UI can chart the single headline line OR the breakdown. Weights are explicit
# here so the score is auditable, not a black box.
SCORE_WEIGHTS = {
    "model_quality": 0.40,
    "task_capability": 0.35,
    "production_readiness": 0.25,
}


# ─── Milestone history ───────────────────────────────────────────────────────
# Ordered oldest → newest. `date` drives the run start time so the MLflow UI
# shows the real chronology. Metrics are logged as fractions (0–1) where they
# are accuracies/rates so they compare cleanly on one axis.

MILESTONES = [
    {
        "name": "v1-classification-baseline",
        "version": "v1.0",
        "date": datetime(2026, 3, 7),
        "approach": "Image classification (single note)",
        "status": "superseded",
        "score": {"model_quality": 0.9414, "task_capability": 0.30,
                  "production_readiness": 0.30},
        "params": {
            "task": "classification",
            "architecture": "CurrencyMobileNet (MobileNetV2-style, from scratch)",
            "pretrained_weights": "none (Kaiming init)",
            "num_classes": 9,
            "input_size": 224,
            "params_millions": 2.235,
            "dataset_images": 3687,
            "train_images": 2637,
            "val_images": 760,
            "test_images": 290,
            "epochs_trained": 92,
            "best_epoch": 77,
            "optimizer": "SGD+Nesterov (lr=0.01, wd=1e-4)",
            "scheduler": "cosine annealing",
            "device": "CPU",
            "export_format": "TorchScript Lite (.ptl)",
        },
        "metrics": {
            "val_accuracy": 0.9579,
            "val_macro_f1": 0.9576,
            "test_accuracy": 0.9414,
            "val_mAP": 0.9183,
            "model_size_mb": 8.43,
        },
        "note": (
            "## v1.0 — Classification baseline\n\n"
            "Custom **CurrencyMobileNet** (2.2M params, trained from scratch) "
            "classifying a single Egyptian banknote into 9 classes. Reached "
            "**95.79% val** / **94.14% test** accuracy. Shipped a mobile "
            "TorchScript-Lite export (8.43 MB) and an OpenCV camera pipeline "
            "(region detection + crop-then-classify + TTA + EMA smoothing).\n\n"
            "**Key limitation:** classifies only ONE note per image — cannot "
            "count multiple notes or total their value (the actual goal). Also "
            "a train/camera accuracy gap on cluttered backgrounds."
        ),
    },
    {
        "name": "v2-yolov8-detection",
        "version": "v2.0",
        "date": datetime(2026, 5, 2),
        "approach": "Object detection (multi-note) — YOLOv8",
        "status": "superseded",
        "score": {"model_quality": 0.982, "task_capability": 0.45,
                  "production_readiness": 0.45},
        "params": {
            "task": "object_detection",
            "architecture": "YOLOv8n (Ultralytics)",
            "pretrained_weights": "yolov8n.pt (COCO)",
            "num_classes": 7,
            "input_size": 416,
            "epochs_trained": 22,
            "batch": 4,
            "optimizer": "auto",
            "scheduler": "cosine LR (close_mosaic=10)",
            "seed": 42,
            "device": "CPU",
            "run_dir": "outputs/runs/train2",
            "export_format": ".pt / .torchscript / .ptl",
        },
        "metrics": {
            "mAP50": 0.982,
            "mAP50_95": 0.971,
            "precision": 0.978,
            "recall": 0.955,
            "model_size_mb": 5.919,
        },
        "note": (
            "## v2.0 — Switch to YOLOv8 detection\n\n"
            "Re-framed the project from *classification* to **object "
            "detection**, so multiple notes can be located, classified, and "
            "their denominations summed into a running EGP total. Dropped the "
            "two duplicate '(new)' classes → **7 clean denomination classes**. "
            "YOLOv8n fine-tuned from COCO weights; strong per-box metrics "
            "(mAP50 0.982). Mobile export retained for the Flutter app.\n\n"
            "**Key limitation:** labels were full-image pseudo-boxes, so "
            "multi-note **counting** was unreliable (see next milestone)."
        ),
    },
    {
        "name": "v2.1-multinote-counting-fix",
        "version": "v2.1",
        "date": datetime(2026, 5, 21),
        "approach": "Synthetic-data fine-tune + class-agnostic NMS",
        "status": "superseded",
        "score": {"model_quality": 0.995, "task_capability": 0.88,
                  "production_readiness": 0.60},
        "params": {
            "task": "object_detection",
            "architecture": "YOLOv8n (fine-tuned from train2 best.pt)",
            "num_classes": 7,
            "input_size": 416,
            "epochs_trained": 20,
            "batch": 4,
            "device": "CPU",
            "synthetic_train_scenes": 1800,
            "synthetic_val_scenes": 300,
            "notes_per_scene": "1-3",
            "conf_threshold": 0.45,
            "agnostic_nms": True,
            "run_dir": "outputs/runs/train_finetune",
        },
        "metrics": {
            "mAP50": 0.995,
            "mAP50_95": 0.989,
            "precision": 0.991,
            "recall": 0.964,
            "synth_val_mAP50": 0.91,
            "model_size_mb": 5.929,
            # Counting accuracy: before vs after the fix (held-out multi-note set)
            "count_exact_before": 0.18,
            "count_exact_after": 0.88,
            "count_within1_before": 0.58,
            "count_within1_after": 1.00,
            "per_denom_acc_before": 0.25,
            "per_denom_acc_after": 0.91,
        },
        "note": (
            "## v2.1 — Multi-note counting fix\n\n"
            "**Root cause:** every training label was a full-image pseudo-box "
            "`<cls> 0.5 0.5 1.0 1.0`, so the model learned *a note = the whole "
            "frame* and split/duplicated boxes on real multi-note photos.\n\n"
            "**Fix:** generated a **synthetic multi-note dataset** "
            "(`make_synthetic.py`, 1800 train / 300 val scenes with *real* "
            "bounding boxes), fine-tuned from the v2 checkpoint on combined "
            "data, and added a **class-agnostic NMS** guard (conf=0.45).\n\n"
            "**Result:** exact note-count accuracy **18% → 88%**, within-±1 "
            "**58% → 100%**, per-denomination **25% → 91%**. This run is the "
            "current active model in the registry."
        ),
    },
    {
        "name": "v2.0-production-hardening",
        "version": "prod-2.0",
        "date": datetime(2026, 6, 18, 1),
        "approach": "Production hardening (same model)",
        "status": "shipped",
        "score": {"model_quality": 0.995, "task_capability": 0.88,
                  "production_readiness": 0.90},
        "params": {
            "task": "object_detection",
            "model_unchanged": True,
            "added_validators": True,
            "added_structured_logging": True,
            "added_confidence_risk_flags": True,
            "added_json_schema": "v2.0",
            "added_profiling": True,
            "added_model_registry": True,
        },
        "metrics": {
            "num_tests": 37,
            "mAP50": 0.995,  # carried from the active fine-tuned model
        },
        "note": (
            "## prod-2.0 — Production hardening\n\n"
            "No model change — engineering robustness around the same "
            "fine-tuned detector: **input validation**, **structured JSONL "
            "logging**, **confidence + risk flags** (low/medium/high/reject "
            "with 50↔100 confusion warnings), **enriched JSON schema v2** "
            "(metadata, normalized bboxes, confidence stats), **performance "
            "profiler**, **model registry**, and a **37-test suite** "
            "(incl. adversarial: brightness/blur/rotation)."
        ),
    },
    {
        "name": "v2.2-explainability",
        "version": "v2.2",
        "date": datetime(2026, 6, 18, 12),
        "approach": "Visual explainability (Eigen-CAM / Grad-CAM)",
        "status": "shipped (latest)",
        "score": {"model_quality": 0.995, "task_capability": 0.88,
                  "production_readiness": 1.00},
        "params": {
            "task": "object_detection",
            "model_unchanged": True,
            "explain_methods": "eigencam (default), gradcam",
            "default_method": "eigencam",
            "cam_target_layers": "[15, 18, 21] (Detect-head inputs P3/P4/P5)",
            "new_module": "src/explain.py",
            "cli_flag": "--explain / --explain-method",
        },
        "metrics": {
            "num_tests": 53,
            "mAP50": 0.995,
        },
        "note": (
            "## v2.2 — Visual explainability (latest)\n\n"
            "Answers *why did the model output this note?* `--explain` writes a "
            "saliency **heatmap** plus a per-detection **salience_ratio** and a "
            "plain-language **explanation**, on top of the existing confidence "
            "/ risk / warning fields.\n\n"
            "Two methods: **Eigen-CAM** (default — robust, well-localized, no "
            "gradient artifacts) and **Grad-CAM** (opt-in, class-discriminative). "
            "Targets the YOLOv8 Detect-head feature maps. Test suite grew "
            "37 → **53 tests**."
        ),
    },
]


def _reset_experiment(client: MlflowClient) -> str:
    """Return the experiment id, clearing any prior runs so re-runs are clean."""
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        return client.create_experiment(EXPERIMENT)
    for run in client.search_runs([exp.experiment_id], run_view_type=3):  # ALL
        client.delete_run(run.info.run_id)
    return exp.experiment_id


def log_history() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    exp_id = _reset_experiment(client)

    for order, m in enumerate(MILESTONES, start=1):
        start_ms = int(m["date"].timestamp() * 1000)
        run = client.create_run(
            exp_id, start_time=start_ms, run_name=m["name"],
            tags={
                "version": m["version"],
                "date": m["date"].strftime("%Y-%m-%d"),
                "approach": m["approach"],
                "status": m["status"],
                "order": str(order),
                "mlflow.note.content": m["note"],
            },
        )
        rid = run.info.run_id
        for k, v in m["params"].items():
            client.log_param(rid, k, v)
        for k, v in m["metrics"].items():
            client.log_metric(rid, k, float(v))

        # Normalized headline score (one comparable line across all milestones)
        # plus its three component pillars.
        s = m["score"]
        headline = round(sum(SCORE_WEIGHTS[k] * s[k] for k in SCORE_WEIGHTS), 4)
        client.log_metric(rid, "headline_score", headline)
        for k, v in s.items():
            client.log_metric(rid, f"score_{k}", float(v))

        client.set_terminated(rid, status="FINISHED", end_time=start_ms)
        print(f"  [{order}/{len(MILESTONES)}] {m['version']:9s} "
              f"{m['name']:32s} headline={headline:.3f}")

    print(f"\nLogged {len(MILESTONES)} milestones to experiment "
          f"'{EXPERIMENT}'.")
    print(f"Tracking store: {TRACKING_URI}")
    print(f"View the timeline with:\n  mlflow ui --backend-store-uri "
          f"{TRACKING_URI}\nthen open http://127.0.0.1:5000")


if __name__ == "__main__":
    log_history()
