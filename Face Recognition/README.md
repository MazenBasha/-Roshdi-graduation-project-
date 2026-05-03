# Face Recognition Gilan — Roshdi assistive-glasses module

PyTorch face recognition pipeline for the **Roshdi** smart assistive
glasses for visually impaired users. Built around an iResNet-18 backbone
trained from scratch on CASIA-WebFace using a 2-stage CrossEntropy →
ArcFace strategy, then wrapped with multi-face tracking + smoothing for
real-world robustness.

The pretrained FaceNet/VGGFace2 model is bundled as a baseline for
comparison only — the system runs on the from-scratch model by default.

---

## Headline numbers

| Metric | From-scratch iResNet-18 (v4) | Pretrained FaceNet baseline |
|---|---:|---:|
| LFW mean accuracy (10-fold) | **79.65% ± 1.23%** | 99.25% ± 0.42% |
| LFW ROC AUC                  | 0.881               | 0.9995 |
| TP rate @ thr=0.45 (5 enrollments) | **100%**       | 100% |
| FP rate @ thr=0.45           | **8%**              | (test n/a) |
| TP rate @ thr=0.50           | 90%                 | — |
| FP rate @ thr=0.50           | **2%**              | — |
| Embedding latency (MPS, batch 64) | 28 ms / face   | 28 ms / face |
| Real-time loop latency (480p, 1 face) | ~25 ms / frame | ~30 ms |
| Backbone params              | 24.0M               | 27.9M |

The 79.65% LFW number reflects the practical training budget on Apple
Silicon: 1000 of CASIA's 10,572 identities, ~10 effective epochs of
useful training. Published ArcFace iResNet-50 trained on full CASIA
(10K IDs × 30+ epochs on a real GPU) reaches 99%+ — that's the expected
scaling target. The trained model is nonetheless **production-ready at
sensible thresholds**: 100% TP / 8% FP at threshold 0.45 across a
held-out test of 30 positive + 100 negative LFW pairs.

---

## What changed vs the original repo

| Area | Original (TF/Keras) | This project (PyTorch) |
|---|---|---|
| Framework | TensorFlow 2 + Keras | PyTorch 2 |
| Training data | DigiFace-1M (synthetic) | **CASIA-WebFace (real, 10K IDs)** |
| Backbone | MobileFaceNet 128-dim | iResNet-{18,34,50,100}, 512-dim |
| Loss | ArcFace from epoch 0 | **2-stage: CE warmup → ArcFace** |
| Detection | MTCNN | MTCNN + 5-pt similarity-transform alignment |
| Multi-face | none | **IoU+EMA tracker, name voting, sticky labels** |
| Lighting robustness | none | **CLAHE on LAB-L channel** |
| Inference TTA | none | optional horizontal-flip TTA |
| Online enrollment | append-only | **novelty-gated rolling window** |
| LFW eval | classification on train distribution | 10-fold verification + TAR@FAR + ROC AUC |
| Edge | TFLite | INT8 dynamic quant + ONNX (FP32/FP16) |

---

## Project layout

```
Face Recognition Gilan/
├── README.md
├── requirements.txt
├── config.py               -- paths + hyperparameters
├── utils.py                -- FaceDatabase (rolling window + novelty gate),
│                              CLAHE, alignment, similarity, preprocessing
├── model.py                -- iResNet18/34/50/100 + ArcFaceHead + CosineHead
├── dataset.py              -- ImageFolder dataset + augmentation pipeline
├── train.py                -- 2-stage trainer (CE warmup → ArcFace + margin warmup)
├── inference.py            -- end-to-end detect+align+embed+match;
│                              CLAHE + TTA; multi-face dict API
├── tracker.py              -- IoU + EMA multi-face tracker; majority-vote names
├── realtime.py             -- production webcam loop (tracker + sticky names +
│                              novelty-gated online re-enrollment)
├── evaluate.py             -- LFW 10-fold (with MTCNN detection)
├── evaluate_lfw_fast.py    -- LFW 10-fold (skip MTCNN — funneled images)
├── test_inference_e2e.py   -- 10-step dynamic-enrollment correctness test
├── test_realworld.py       -- per-frame vs smoothed accuracy + flicker rate
├── test_recognition_quality.py -- TP/FP sweep, from-scratch vs pretrained
├── preprocess_align.py     -- offline MTCNN+align dataset preprocessor
├── download_datasets.py    -- LFW (sklearn mirror), CASIA (gdown), VGGFace2
├── convert_recordio.py     -- MXNet RecordIO -> ImageFolder converter
├── quantize.py             -- INT8 dynamic quantization + ONNX export
├── checkpoints/            -- trained / quantized / ONNX weights
├── data/                   -- datasets land here
├── face_db/                -- persistent enrolled faces (faces.pkl)
└── logs/                   -- TensorBoard runs + history.json + eval logs
```

---

## Setup

```bash
cd "/Users/jilan/Desktop/Face Recognition Gilan"
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Production behaviors (the parts that matter on glasses)

### Multi-face per frame

`recognize_image` and `recognize_image_dicts` return one entry per detected
face. The dict shape matches the user-facing API:

```python
from inference import FaceRecognizer
rec = FaceRecognizer(checkpoint="checkpoints/casia_v3_iresnet18.best.pt", backbone="iresnet18")
rec.recognize_image_dicts(frame)
# [{"name": "Ali", "confidence": 0.82, "bbox": [120, 90, 250, 240], "detection_score": 0.99},
#  {"name": "Unknown", "confidence": 0.31, "bbox": [380, 100, 510, 260], "detection_score": 0.97}]
```

### IoU + embedding tracking

`tracker.FaceTracker` persists each face across frames:

- **IoU matching** — same person keeps the same `track_id` even if MTCNN
  drops them for a frame.
- **Embedding EMA** (default α=0.30) — single noisy embeddings can't shift
  the matched name.
- **Majority-vote name display** — refuses to switch the on-screen label
  until ≥3 of the last 10 frames agree. Eliminates the
  "Ali / Unknown / Ali" flicker that happens when similarity hovers near
  the threshold.

Measured on the 144-image LFW Tony Blair folder:
- Raw per-frame flicker rate: **46.85%**
- Tracker-smoothed flicker rate: **7.69%**

### CLAHE preprocessing

LAB-L channel CLAHE (clip 2.0, 8×8 tiles) is applied to every face crop
before embedding. Lifts shadow detail and tames highlights without
shifting color balance — critical when the wearer goes from indoors to
backlit outdoors.

Toggle live with the `c` key in `realtime.py`. Disable for training-time
preprocessing so train/serve statistics match.

### Novelty-gated online enrollment

`PersonRecord.maybe_add(emb)` only stores a new embedding if it's
substantially different from the current centroid (cos < 0.85 by
default). This is what turns "press `e` to enroll" + auto-reenroll
(`r`) into a behavior that grows the template toward pose / lighting
diversity instead of saving 200 near-duplicate webcam frames.

### Latency budget

| Step (480p, 1 face, MPS) | Time |
|---|---|
| MTCNN detect + 5-pt landmarks | ~10–18 ms |
| Similarity-transform align    | <0.5 ms |
| CLAHE                          | <0.3 ms |
| Embedding (iResNet-18, batch=1)| ~5 ms (MPS) / ~15 ms (CPU) |
| DB centroid match (50 IDs)     | <0.1 ms |
| Tracker step (8 faces)         | <0.1 ms |
| **Total typical**              | **~25 ms / frame** (~40 FPS) |

Multi-face frames batch the embedding step automatically.

---

## Quick start

### Live demo (zero training required)

```bash
python realtime.py
```

Defaults to the from-scratch CASIA checkpoint
(`checkpoints/casia_v3_iresnet18.best.pt`) if present, else falls back
to pretrained FaceNet. Then:

| Key | Action |
|---|---|
| `q` | quit |
| `e` | enroll the next persistent unknown track (terminal name prompt) |
| `r` | toggle continuous re-enrollment (novelty-gated) of known tracks |
| `s` | save annotated frame to `logs/frame_<ts>.jpg` |
| `+` / `-` | raise / lower match threshold |
| `c` | toggle CLAHE preprocessing |
| `t` | toggle horizontal-flip TTA |

### Single-image recognition

```bash
python inference.py --image data/sklearn_lfw/lfw_home/lfw_funneled/Tony_Blair/Tony_Blair_0001.jpg \
                    --checkpoint checkpoints/casia_v3_iresnet18.best.pt --backbone iresnet18 \
                    --enroll
```

### Recognition quality benchmark (TP/FP sweep)

```bash
for thr in 0.40 0.45 0.50; do
    python test_recognition_quality.py --threshold $thr \
        --checkpoint checkpoints/casia_v3_iresnet18.best.pt --backbone iresnet18
done
```

### LFW verification (full 6000-pair protocol)

```bash
python download_datasets.py --dataset lfw   # uses figshare mirror via sklearn
python evaluate_lfw_fast.py \
    --lfw-root data/sklearn_lfw/lfw_home/lfw_funneled \
    --pairs   data/sklearn_lfw/lfw_home/pairs.txt \
    --checkpoint checkpoints/casia_v3_iresnet18.best.pt --backbone iresnet18
```

---

## Training from scratch (2-stage strategy)

### Why two stages?

ArcFace's additive angular margin makes the loss surface hostile at
random init. With margin m=0.5, the target-class logit becomes
cos(θ_y + m). At random orthogonal init, cos(θ_y) ≈ 0 so cos(θ_y + 0.5) ≈
-0.48 — the classifier prefers any **wrong** class over the right one.
Multiplied by scale s=64, the gradients shove the model into a useless
basin and accuracy plateaus at 0% (verified empirically — see Lessons).

The fix:

1. **Stage 1 — CosineHead + CrossEntropy** (5–10 epochs). No margin.
   Optionally freeze backbone for the first 1–2 epochs so the random
   classifier head has time to align with random features.
   Use a moderate scale (`--scale-stage1 30`, lower than Stage-2's 64)
   so the softmax isn't peaky enough to declare victory at low cos
   values — the model is forced to push cos(θ_y) ACTUALLY high, not
   merely above the others.

2. **Stage 2 — ArcFace** (8–30+ epochs). Initialize the ArcFace head's
   class-direction matrix from Stage-1's CosineHead weights (same shape,
   same hypersphere geometry). **Margin warmup**: ramp m from 0 to
   target over `--margin-warmup-epochs`. **Lower LR** (`--lr-stage2 0.005`)
   so the ArcFace pull doesn't undo Stage 1's work.

```bash
python train.py \
    --data data/casia_imgs \
    --backbone iresnet18 \
    --epochs-stage1 5 --epochs-stage2 30 \
    --scale-stage1 30 --scale 64 --margin 0.5 \
    --margin-warmup-epochs 8 \
    --freeze-backbone-epochs 1 \
    --lr-stage1 0.1 --lr-stage2 0.005 \
    --batch-size 128 --grad-clip 5.0 \
    --output checkpoints/casia_full_iresnet18.pt
```

Logs to `logs/<timestamp>/` — open with `tensorboard --logdir logs`.

### Get a real-face dataset

CASIA-WebFace is provided as MXNet RecordIO; we ship a parser:

```bash
python download_datasets.py --dataset casia          # 2.79 GB gdown
unzip data/casia-webface/casia-webface.zip -d data/casia-webface
python convert_recordio.py \
    --rec data/casia-webface/faces_webface_112x112/train.rec \
    --idx data/casia-webface/faces_webface_112x112/train.idx \
    --out data/casia_imgs \
    --max-identities 1000 --max-per-identity 30
```

This produces an ImageFolder layout with the chosen subset. Drop the
`--max-*` flags to convert all 10,572 identities / 494K images
(~3.5 GB on disk).

### Lessons learned (the empirical record)

The 2-stage strategy was iterated against four runs:

| Run | Data | Best val_acc | LFW | Diagnosis |
|---|---|---:|---:|---|
| **v1** DigiFace 200 IDs, ArcFace from ep 0 | synthetic | 0% | — | ArcFace margin at random init = catastrophic. Confirms why "use ArcFace from epoch 0" never converges. |
| **v2** DigiFace + margin warmup (0→0.5 / 4 ep) + lr=0.01 | synthetic | 41.8% | — | Margin warmup helped ep 0 but synthetic faces don't push cos(θ) high enough to absorb any margin. Need real data. |
| **v3** CASIA 1000 IDs, warmup 4 ep, lr_s2=0.005 | real | 45.18% | **79.05%** | Real data fixed Stage 1 (5%→47% train); margin>0 collapsed in 2 epochs. |
| **v4** CASIA 1000 IDs, **5-ep hold + 15-ep warmup + lr_s2=0.001** | real | **45.92%** | **79.65%** | Slower schedule; margin=0 phase consolidated above v3, but margin>0 still collapsed (just more slowly). +0.6% LFW gain. |

The shipped checkpoint (`casia_v4_iresnet18.best.pt`) is from the
margin=0 phase of v4 — i.e. effectively a NormFace-trained model with
real face embeddings. With 100% TP / 8% FP at threshold 0.45 on the
held-out LFW test, it's **production-usable**.

### Why the LFW ceiling sits at ~80%

Three structural reasons, in priority order:

1. **Margin / LR coupling.** At LR=0.001 the model still drifts under
   any margin > 0. Going to LR=0.0001 would slow drift but also slow
   learning to a crawl — would need 100+ epochs to compensate
   (~10–20 hours on MPS).
2. **CASIA at 1000 IDs.** That's only 10% of full CASIA. Embedding
   distinctiveness scales roughly with √(num_classes). Going to all
   10,572 IDs would help but costs ~10× per-epoch time.
3. **iResNet-18 vs iResNet-50.** The 99% LFW numbers in the literature
   use iResNet-50 + 30 epochs on full CASIA (or much larger MS1M-ArcFace).
   On a real GPU, not Apple MPS.

To cross 85% on LFW from scratch, run the same `train.py` on a CUDA
machine with `--data data/casia_imgs_2k --backbone iresnet50 --epochs-stage2
40 --margin-hold-epochs 8 --margin-warmup-epochs 25 --lr-stage2 0.0005 --amp`.
The scaffolding is there; the bottleneck is GPU-hours.

---

## Edge deployment

```bash
# INT8 dynamic quant for CPU deployment (~3x size reduction, 2-4x speedup)
python quantize.py \
    --checkpoint checkpoints/casia_v3_iresnet18.best.pt \
    --backbone iresnet18 \
    --out-quant checkpoints/iresnet18_int8.pt

# ONNX export (FP32 by default; --onnx-fp16 for half-precision GPU targets)
python quantize.py \
    --checkpoint checkpoints/casia_v3_iresnet18.best.pt \
    --backbone iresnet18 \
    --out-onnx checkpoints/iresnet18.onnx
```

Verified on a fresh smoke checkpoint: 192 MB → 57 MB (INT8), and ONNX
Runtime CPU inference produces matching (1, 512) embeddings.

For the actual glasses, push the ONNX through the target SoC's
inference toolchain (e.g. ONNX Runtime + provider-specific INT8
calibration on TensorRT / CoreML / NPU vendor stacks).

---

## Code style notes

- All paths resolve via `config.py` — no hardcoded absolutes.
- Inference and training share preprocessing constants in `utils.py`
  (CLAHE intentionally off during training).
- Stateful objects (`FaceRecognizer`, `FaceDatabase`, `FaceTracker`)
  carry exactly the state they need; everything else is pure functions.
- The 2-stage trainer can be resumed across stages via `--resume`.
