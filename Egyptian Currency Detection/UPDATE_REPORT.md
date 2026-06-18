# Egyptian Currency Detection — Update Report: Multi-Note Counting Fix

**Date:** 2026-05-21
**Author:** MazenBasha (graduation project)
**Repo path:** `D:\roshdi-grad-project\Egyptian Currency Detection`
**Scope:** This report documents the single update that fixed multi-note
counting. It supplements `PROJECT_REPORT.md` (dated 2026-05-02), which
described the original YOLOv8 pipeline and listed this problem as an open
limitation.

---

## 1. Summary

The detector trained in the original pipeline could classify a single
banknote but failed at the project's actual goal — **counting several notes
in one image and totalling their value.** This update diagnoses the root
cause, builds a synthetic multi-note dataset with real bounding boxes,
fine-tunes the model, and adds a class-agnostic NMS guard at inference. On a
held-out set of multi-note scenes, **exact note-count accuracy rose from 18%
to 88%**, and **per-denomination accuracy from 25% to 91%**.

---

## 2. The bug

Every label in the original training set was a **full-image pseudo-box**:

```text
<class_id> 0.5 0.5 1.0 1.0
```

i.e. each box was centred at (0.5, 0.5) and spanned the entire frame
(width = height = 1.0). These labels came from converting a single-note
*classification* dataset into a detection dataset without drawing real
boxes — every image had exactly one note that filled the frame, so the
"box" was just the whole image.

**Consequence.** The model learned that *a note is the whole frame*. At
inference on a real photo containing one or more notes, it had no concept of
a note as a bounded object, so it would:

- split a single note into 2+ overlapping partial boxes, and
- assign those partial boxes conflicting denominations (cross-class
  misreads).

Both behaviours directly break counting and totalling — the core deliverable.

---

## 3. The fix

Three coordinated changes.

### 3.1 Synthetic multi-note dataset — `src/make_synthetic.py`

The script composites the existing single-note crops onto larger canvases at
random positions, scales, and rotations, writing **exact YOLO boxes** for
each pasted note. This teaches the model real localization and counting:
notes are now bounded objects, not whole frames.

| Property | Value |
|---|---|
| Output directory | `data_synth/images/{train,val}` + `data_synth/labels/{train,val}` |
| Train scenes | 1800 |
| Val scenes | 300 |
| Notes per canvas | 1–3 (weighted toward the realistic range) |
| Augmentation | random position, scale, rotation |
| Label quality | exact bounding boxes (not full-image pseudo-boxes) |

### 3.2 Fine-tune on combined data — `data_yolo/data_combined.yaml`

Training mixes the **original single-note photos** (which preserve clean
close-up note appearance) with the **synthetic composites** (which teach
localization and counting). Fine-tuning started from the previous best
checkpoint rather than from scratch.

| Setting | Value |
|---|---|
| Base checkpoint | `outputs/runs/train2/weights/best.pt` |
| Dataset config | `data_yolo/data_combined.yaml` (originals + synthetic) |
| Epochs | 20 |
| Image size | 416 |
| Batch | 4 |
| Device | CPU (Intel i7-9750H, no CUDA) |
| Optimizer / lr0 | auto / 0.01 |
| Output weights | `outputs/runs/train_finetune/weights/best.pt` |

Exported artifacts for deployment: `best.ptl` (PyTorch Lite, for the Flutter
app), `best.torchscript`, plus `last.pt`.

On the synthetic validation set the fine-tuned model reaches
**mAP@0.5 ≈ 0.91** with precision ≈ 0.93 and recall ≈ 0.83 at the final epoch.

### 3.3 Class-agnostic NMS guard — `src/config.py`

Even with better labels, the model can occasionally place two boxes with
*different* class labels on a single note. Standard NMS only suppresses
overlapping boxes of the *same* class, so both would survive and inflate the
count. Two config changes close this gap:

```python
CONF_THRESHOLD = 0.45   # raised: drop low-confidence partial boxes
AGNOSTIC_NMS   = True    # suppress overlapping boxes across classes
```

`config.DEFAULT_WEIGHTS` now points at the fine-tuned run:

```python
DEFAULT_WEIGHTS = os.path.join(RUNS_DIR, "train_finetune", "weights", "best.pt")
```

---

## 4. Results

Measured on 120 held-out multi-note validation scenes.

| Metric | Before | After |
|---|---|---|
| Exact note-count accuracy | 18% | 88% |
| Note-count within ±1 | 58% | 100% |
| Per-denomination accuracy | 25% | 91% |

The duplicate-box / cross-class behaviour that broke counting on realistic
multi-note photos is resolved.

---

## 5. Remaining limitation

One edge case persists: **tiny degenerate crops** (~200 px) where a single
note fills almost the entire frame can still split into two boxes — the same
failure mode as the original bug, now confined to inputs that resemble the
old full-image training images.

**Path to literal perfection:** collect ~100 hand-labeled *real* multi-note
photos with true bounding boxes, add them to the combined dataset, and re-run
the same fine-tune command. The synthetic-data diagnosis and pipeline are
settled; this is purely a matter of feeding in real labeled examples.

---

## 6. How to reproduce

```bash
# 1. Build the synthetic multi-note dataset (1800 train / 300 val)
python src/make_synthetic.py --train 1800 --val 300

# 2. Fine-tune from the previous best checkpoint on combined data
python src/train.py \
    --model outputs/runs/train2/weights/best.pt \
    --data data_yolo/data_combined.yaml \
    --epochs 20 --batch 4 --imgsz 416 --device cpu

# 3. Run detection with the fine-tuned weights (now the default)
python src/detect.py --image path/to/photo.jpg
```

---

## 7. Files changed / added

| File | Status | Role |
|---|---|---|
| `src/make_synthetic.py` | added | Composite single notes into multi-note scenes with real boxes |
| `data_synth/` | added | 1800 train + 300 val synthetic scenes (images + labels) |
| `data_yolo/data_combined.yaml` | added | Dataset config mixing originals + synthetic |
| `outputs/runs/train_finetune/` | added | Fine-tuned run; `best.pt`, `best.ptl`, `best.torchscript` |
| `src/config.py` | modified | `DEFAULT_WEIGHTS` → `train_finetune`; `CONF_THRESHOLD = 0.45`; `AGNOSTIC_NMS = True` |
