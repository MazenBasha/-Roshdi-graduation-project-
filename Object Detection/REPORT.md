# Real-Time General-Purpose Object Detection for an Assistive Mobile Application: A Pretrained-First Deployment of YOLO11n on Microsoft COCO

**A Technical Report Submitted in Partial Fulfilment of the Requirements for the Graduation Project**

---

## Abstract

Object detection — the joint task of localising and classifying every object of interest within an image — is a foundational capability for assistive mobile systems that describe a visual scene to a user who cannot see it. This report documents the object-detection module of the **Roshdi** assistive AI system, which speaks aloud the presence, identity and approximate spatial location of everyday objects captured by the user's phone camera. The module is built around **YOLO11n**, the nano-scale member of Ultralytics' YOLO11 detector family, **deployed using its publicly released COCO-pretrained weights and used as-is, without further fine-tuning**. The eighty Microsoft COCO categories on which YOLO11n was trained — people, vehicles, furniture, kitchenware, electronics, food, animals and common indoor/outdoor objects — coincide exactly with the general-purpose perception need the assistive use-case imposes, so the additional training stage that a domain-specific deployment would require is, in our case, an avoidable expenditure of energy and engineering time.

The deliverable is consequently a **pretrained-first, training-optional pipeline**: a containerised FastAPI inference service that auto-loads `yolo11n.pt` on first request; reproducible exports to **ONNX**, **INT8-quantised TensorFlow Lite**, and **Core ML** for the Roshdi Flutter application; observability via structured logging and Prometheus metrics; an input/output safety boundary appropriate for an assistive context; and an explicit, sign-posted **optional** fine-tuning pathway (`train.py`, `download_dataset.py`, and Kaggle / Colab notebooks) that can be activated if a future domain-specific dataset is collected. The report explains, in turn, the technical rationale for choosing YOLO11n, the architecture of the network, the semantics of the eighty COCO categories that bound its capability, why fine-tuning was deliberately omitted, the expected inference performance on cloud and mobile targets, and the end-to-end deployment architecture that makes the model accessible to the Roshdi application.

---

## 1. Introduction

### 1.1 Problem definition

Object detection generalises image classification along two axes. Where a classifier answers the question *"what is in this image?"* with a single label, a detector must answer *"what objects are present, and where is each one?"* — emitting, for every instance, a tight axis-aligned bounding box together with a class label and a confidence score. Formally, given an image $I \in \mathbb{R}^{H \times W \times 3}$, a detector produces a variable-length set of predictions $\{(b_i, c_i, s_i)\}_{i=1}^{N}$, where $b_i$ are box coordinates, $c_i \in \{1, \dots, K\}$ is a class index over $K$ categories, and $s_i \in [0,1]$ is a confidence. The variable cardinality of the output, the need to reason about object scale and aspect ratio, and the requirement to suppress duplicate detections together make detection substantially harder than classification.

### 1.2 The assistive-mobile use case

Roshdi is an assistive system for visually-impaired users. Its object-detection module is given a single camera frame and must, within a tight latency budget, return a list of detections that the system's voice layer can speak aloud: *"chair, near, on your right."* Three properties follow from this user-facing brief.

First, **real-time, on-device**: the round-trip from frame capture to spoken description should fit within roughly 100 ms to feel responsive. A cloud round-trip cannot reliably guarantee that latency, and the cellular link may be unavailable.

Second, **privacy by default**: pixel data should not leave the phone. The cloud HTTP fallback we provide is opt-in and exists for development and remote debugging — not for production user traffic.

Third, **general-purpose vocabulary**: Roshdi must announce *anything the user encounters that matters*, from a *chair* to a *dog* to a *cup* on a table. The category set must therefore span the everyday physical world, not a narrow vertical domain.

### 1.3 Contribution

This report and the accompanying repository contribute (i) a **pretrained-first deployment design** for an assistive object detector — including the explicit argument for why fine-tuning was unnecessary and the conditions under which it would become necessary; (ii) a **reproducible inference and export pipeline** producing mobile-ready ONNX/TFLite/CoreML artefacts directly from the released YOLO11n checkpoint; and (iii) a **production-grade serving stack** — input validation, output safety filtering, structured logging, Prometheus metrics, health and readiness probes, container packaging — that brings the module to the maturity level the project rubric demands.

---

## 2. Why pretrained YOLO11n was selected

The selection criterion was a single Pareto trade-off across **accuracy, latency, energy footprint, and fit-to-task**. Four candidate families were considered: two-stage detectors (Faster R-CNN and its descendants), transformer detectors (DETR and successors), single-stage convolutional detectors (the SSD and YOLO families), and on-device-specific designs (MobileNet-SSD, EfficientDet-Lite). The four lenses below combine to make YOLO11n the natural choice.

**1. Single-shot detection wins on the figure of merit that actually matters.** Two-stage detectors run a classifier head over hundreds of region proposals per image — accurate but too slow for the 30+ FPS the Roshdi camera stream needs. Transformer detectors are improving rapidly but their compute profile still favours datacentre GPUs over phone NPUs. The YOLO family casts detection as a single forward pass of a fully-convolutional network producing dense predictions over the whole image, and is therefore the established choice when latency is a first-class constraint.

**2. Nano scale matches a mobile budget.** Within YOLO11, the **nano variant** is obtained by scaling the network's depth and width down by fixed multipliers, yielding ≈ 2.6 M parameters and ≈ 6.5 G FLOPs at 640 × 640. This is small enough to quantise to a few megabytes for on-device deployment, small enough to run inside a phone's thermal envelope, and small enough to remain unnoticeable as an app-bundle increment.

**3. The pretrained label space is *exactly* the assistive label space.** YOLO11n's released weights are trained on Microsoft COCO's 80 "thing" categories — people, animals, vehicles, kitchenware, furniture, electronics, food, accessories, sports equipment and common indoor/outdoor objects (the full list is given in §3). This catalogue is, almost coincidentally, an excellent approximation of "things a visually-impaired user might encounter in daily life and want announced." A heavier or smaller model would, in our case, change the operating point without unlocking any new use-case the pretrained YOLO11n cannot already serve.

**4. Mature, well-tested export tooling.** Ultralytics ships a one-line export path to **ONNX**, **TFLite (with optional INT8 quantisation calibrated on a small image set)**, and **Core ML** — the three runtimes that matter for a Flutter app targeting Android and iOS. Competing detectors typically require hand-conversion steps that are brittle and version-dependent. The export reliability of the Ultralytics stack is a non-trivial reason to prefer it for a graduation-project timescale.

YOLO11n therefore lies at the intersection of *fastest detector family that retains COCO-class accuracy*, *smallest size that retains the 80-class vocabulary we actually need*, and *best-supported export pipeline to the mobile runtimes we actually target*. We adopt it as released.

---

## 3. The 80 COCO categories

The detection track of COCO defines 80 "thing" categories — countable, localisable objects — organised into intuitive super-groups:

- **People and animals**: person; bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe.
- **Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat.
- **Outdoor / street objects**: traffic light, fire hydrant, stop sign, parking meter, bench.
- **Accessories**: backpack, umbrella, handbag, tie, suitcase.
- **Sports equipment**: frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket.
- **Kitchenware**: bottle, wine glass, cup, fork, knife, spoon, bowl.
- **Food**: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake.
- **Furniture**: chair, couch, potted plant, bed, dining table, toilet.
- **Electronics**: TV, laptop, mouse, remote, keyboard, cell phone.
- **Appliances**: microwave, oven, toaster, sink, refrigerator.
- **Indoor / miscellaneous**: book, clock, vase, scissors, teddy bear, hair drier, toothbrush.

(The canonical ordering, used by the model's output indices, lives in `flutter_integration/coco_labels.txt`.)

Two properties of this label set are important to flag explicitly for the assistive use-case.

First, the COCO category distribution is **naturally long-tailed**: *person* dominates by a wide margin, while *hair drier* and *toothbrush* are comparatively rare. Per-class accuracy is consequently uneven; we expose this honestly via the per-class AP table emitted by `evaluate.py`, and the Roshdi voice layer is configured to speak "uncertain object" rather than the bare label when confidence is low on rare classes.

Second, the label set is **broad but not domain-specific**. COCO is sourced primarily from Flickr photographs taken in Western contexts. For an Egyptian deployment, the model will continue to correctly identify the same eighty categories — a chair is a chair regardless of geography — but currency notes, traffic-sign variants, and any non-COCO classes specific to the user's environment fall outside its vocabulary. Currency in particular is a separate Roshdi module (Egyptian Currency Detection, already shipped). The optional fine-tuning pathway (§7) exists for the case where this module-level coverage proves insufficient.

---

## 4. Model architecture

YOLO11n follows the canonical detector decomposition — **backbone → neck → head** — instantiated at nano scale (depth and width multipliers that yield ≈ 2.6 M parameters and ≈ 6.5 G FLOPs at 640 × 640 input resolution).

### 4.1 Backbone (feature extractor)

A hierarchy of convolutional stages progressively downsamples the input, producing feature maps at strides of 8, 16 and 32 — spatial resolutions of 80 × 80, 40 × 40 and 20 × 20 for a 640 × 640 input. The principal building block is the **C3k2** module, a Cross-Stage-Partial design that splits the feature channels, processes one branch through a stack of bottleneck convolutions, and concatenates the result. CSP-style designs improve gradient flow and parameter efficiency relative to a plain stack of residual blocks. The deepest stage applies an **SPPF** (Spatial Pyramid Pooling – Fast) module — a cheap cascade of max-pooling operations that aggregates context across multiple receptive-field sizes — and a **C2PSA** block, a convolutional stage augmented with partial self-attention, that emphasises informative spatial locations without paying the full cost of a transformer.

### 4.2 Neck (feature aggregation)

A **PAN-FPN** (Path Aggregation Network over a Feature Pyramid) fuses the three backbone scales bidirectionally. A top-down pathway propagates semantically rich, coarse features downward; a bottom-up pathway propagates fine spatial detail upward. The result is three fused feature maps, each carrying both high-level semantics and adequate spatial resolution, so that the head can detect objects across a wide scale range — from the chair occupying half the frame to the cup on the far side of the table.

### 4.3 Head (prediction)

YOLO11n uses an **anchor-free, decoupled detection head**. "Decoupled" means classification and box-regression are produced by separate convolutional branches, which empirically improves both. "Anchor-free" means box geometry is predicted directly from grid locations rather than as offsets to predefined anchor boxes, removing the anchor-tuning hyperparameters. Box edges are regressed via the **Distribution Focal Loss (DFL)** formulation: each of the four box offsets is predicted as a discrete probability distribution over a range of bins and decoded by taking its expectation, which yields sub-pixel-accurate, well-calibrated localisation. Predictions are emitted densely at all three scales and reconciled by **non-maximum suppression** at inference. NMS may optionally be embedded directly in the exported graph (`nms=True` in `export.py`), so that the Flutter client receives final, deduplicated detections and does not need to reimplement NMS in application code.

---

## 5. Transfer learning, and why we use the released weights as-is

### 5.1 The economics of transfer learning

Training a detector from random initialisation on COCO is feasible but expensive — typically hundreds of GPU-hours on a single accelerator. The standard practice that has driven the last decade of detector deployment is **transfer learning**: initialise from a checkpoint that has already learned a strong, general-purpose visual representation on a large dataset, and either (a) continue training on a target dataset of interest, or (b) deploy the initialisation directly when the source and target tasks coincide. The early and middle layers of a convolutional detector learn edges, textures, colour-opponent responses and part-level patterns that transfer broadly across detection tasks; only the head and the deeper feature stages typically benefit from task-specific adaptation. The practical payoff is convergence in a handful of epochs on a target task rather than hundreds, and — in the limit where the source and target tasks are *the same* — convergence in **zero** epochs.

### 5.2 Why no further fine-tuning was performed

Our target task is to detect the 80 COCO categories on natural everyday photographs. **This is exactly the task on which the released `yolo11n.pt` was trained.** The label space is identical, the input resolution and preprocessing are identical, and the visual distribution of frames captured by a phone camera is a reasonable approximation of the COCO distribution. The expected outcome of a thirty-epoch fine-tuning run on the same data and labels is therefore that the model recovers, at best, its starting accuracy — with the down-side of consuming on the order of 24–36 hours of GPU time, a non-trivial energy cost, and the engineering overhead of a training infrastructure that the production pipeline never needs again.

A defensible fine-tuning decision requires a *gradient*: a labelled target dataset whose distribution differs materially from COCO's, on which the pretrained checkpoint underperforms by a measurable margin. Roshdi may produce such a dataset in the future — frames from real user sessions, labelled with corrections collected via the voice-feedback loop — at which point the cost of fine-tuning is justified by an expected absolute gain in per-class AP on the deployed distribution. **Until that data exists, the principled engineering action is not to train.** The optional pathway (`train.py`, `kaggle_pipeline.ipynb`, `colab_local_runtime.ipynb`) is retained in the repository and clearly sign-posted, but is not part of the production workflow.

### 5.3 Advantages of the pretrained-first approach

The advantages of this stance are concrete and observable in the repository:

- **Reproducibility.** Every deployment of the module starts from the same publicly-versioned checkpoint. The hash of the loaded `.pt` file is surfaced at `/healthz` and on every detection response (`model_version`), so the artefact in production is unambiguously identifiable and the deployment is byte-for-byte reproducible by any future operator.
- **Zero-setup deployment.** `pip install -r requirements.txt && uvicorn server.main:app` is sufficient to bring the service up. There is no dataset to download, no API key to register, and no training to wait for. The Docker image goes further and bakes the weights into the layer, producing a fully offline container.
- **No training-data risk.** A training pipeline carries operational risks the pretrained pipeline does not: dataset URLs that break, dataset versions that change semantics, augmentation policies that subtly degrade certain classes, checkpoints that overfit to a peculiarity of the validation split. By deploying a fixed, externally-validated checkpoint, the assistive system inherits the considerable evaluation effort that the Ultralytics community has already invested in that checkpoint and avoids re-introducing its own training-time variance.
- **Honest baseline.** Reported metrics for the production model are the well-known, externally-comparable numbers (§6), not bespoke per-project figures that a reviewer must take on trust.

The trade-off is that we accept the published accuracy as the ceiling for our deployment. The Discussion (§9) returns to this and discusses the conditions under which we would revisit the decision.

---

## 6. Expected inference performance

### 6.1 Reference accuracy on COCO `val2017`

Ultralytics publishes the following numbers for YOLO11n at 640 × 640 input, single-scale, FP32 PyTorch:

| Metric | Value |
|---|---|
| mAP@0.50 | ≈ 0.55 |
| mAP@0.50:0.95 | ≈ 0.39 |
| Parameters | 2.6 M |
| FLOPs | 6.5 G |

These are the figures the production pipeline inherits. They are reproducible locally on a held-out split via `python evaluate.py`, which by default validates the released checkpoint against the Ultralytics-bundled `coco128` mini-split (auto-downloaded, no credentials required) and against any custom `data.yaml` via `--data`.

### 6.2 Compression-induced accuracy cost

INT8 quantisation typically gives up **~1–3 mAP50-95** in exchange for a ~4× reduction in model size and access to integer-arithmetic kernels on mobile NPUs. We accept this trade-off as load-bearing for deployment — without it, the model would not reliably hit the latency budget on mid-tier phones. We do **not** quantise the cloud-server FP32 path, so the per-format evaluation can be reported as a delta against an unquantised reference.

### 6.3 Latency targets

| Target | Format / precision | Expected end-to-end | Notes |
|---|---|---|---|
| iPhone Neural Engine | CoreML, INT8 | 15–25 ms / frame | 40+ FPS, headroom for the voice loop |
| Android NPU (NNAPI) | TFLite, INT8 | 20–40 ms / frame | 25–50 FPS on modern flagships |
| Cloud server | ONNX / FP32 | 60–90 ms / request | server fallback only — not the user-facing path |

The Roshdi voice layer throttles spoken announcements to a far lower rate than the camera frame rate (≤ 5 FPS of spoken output, typically), so even the slower cloud-server profile is acceptable for the fallback role it plays.

### 6.4 Inference-time pipeline

The serving pipeline applied to each image is:

1. **Letterbox resize** to 640 × 640 preserving aspect ratio, padding short side with neutral grey.
2. **Pixel normalisation** to $[0, 1]$.
3. **Single forward pass** through the YOLO11n network.
4. **Decode** of the dense grid into candidate detections.
5. **Non-maximum suppression** at IoU = 0.45 (configurable).
6. **Confidence filter** at 0.25 (configurable).
7. **Area filter** to drop boxes smaller than 0.05 % of the image (high-noise low-utility).
8. **Spatial annotation** — `horizontal_position ∈ {left, center, right}` from the box centre x; `distance_hint ∈ {near, medium, far}` from the box area fraction — for the voice layer.

This pipeline is deterministic for a fixed device and input, and the wire response carries the model's version hash so the exact inference behaviour is rooted in a verifiable artefact.

---

## 7. Deployment architecture

The module is deployed in two complementary modes, both serving the same pretrained weights.

### 7.1 On-device (primary)

The Flutter Roshdi application bundles the **INT8-quantised TFLite** export (Android) and the **Core ML** export (iOS) as asset files. A small Dart wrapper (`flutter_integration/object_detector.dart`) loads the model once at app start, runs inference per camera frame, decodes the output into the shared `Detection` value type, and hands it to the existing voice layer for spoken delivery. NMS is embedded in the export graph (`nms=True`), so the Dart side performs no detector-specific post-processing. The on-device path is fully offline; no pixel data leaves the phone.

### 7.2 Cloud HTTP (fallback / debugging)

A containerised FastAPI service (`server/`) provides the same logical interface over HTTP for development, batch evaluation and the (opt-in) case of an unrecoverable on-device failure. The service exposes:

- `GET /healthz` — liveness, returns model version and uptime.
- `GET /readyz` — readiness, 503 until the model has loaded.
- `GET /metrics` — Prometheus exposition (request rate by status, inference-latency histogram, per-class detection counter, model-loaded gauge).
- `GET /v1/classes` — enumeration of the 80 categories.
- `POST /v1/detect` — multipart image upload; returns `DetectResponse` with the same field names as the Dart side.

Cross-cutting properties of the service are deliberately consistent with the Roshdi platform's standards:

- **Input validation** (`server/validation.py`): byte-size, pixel-count (decompression-bomb), content-type, EXIF orientation, mode-coercion-to-RGB.
- **Inference timeout** enforced via `asyncio.wait_for`; a stuck request becomes a 504, not a wedged worker.
- **Structured JSON logs** with a per-request `X-Request-ID` propagated end-to-end.
- **Prometheus metrics** tagged with status, supporting both SLO monitoring and per-class drift detection (`drift.py`).
- **Output safety** (`server/safety.py`): a configurable head-region pixelation for `person` detections, off by default for the single-user Roshdi context, available as a hook for future custom labels.
- **Container hygiene**: multi-stage Dockerfile, non-root runtime user, baked-in weights for offline operation, `HEALTHCHECK` on `/readyz`.

### 7.3 Versioning and rollback

The wire identifier for the deployed model is the first twelve hexadecimal characters of the SHA-256 of the loaded weights file. It is surfaced at `/healthz` and on every detection response as `model_version`. Rolling forward to a new pretrained release, or rolling back to a previous one, is a single file swap and a service restart — there is no schema migration, no training to redo, and the wire format is unchanged. Should the optional fine-tuning pathway ever be activated, the resulting `best.pt` is a drop-in replacement: point `ROSHDI_OD_WEIGHTS_PATH` at it and restart.

---

## 8. Mobile export

The export stage converts the pretrained checkpoint into the three formats the deployment targets actually consume. INT8 quantisation requires a small representative calibration set; we use the Ultralytics-bundled `coco128` (128 representative COCO images, auto-downloaded), so no Roboflow account or custom dataset is required.

### 8.1 ONNX

**ONNX** (Open Neural Network Exchange) is a framework-agnostic intermediate representation. Exporting to ONNX (`format='onnx', opset=12, simplify=True`) decouples the model from PyTorch and enables execution under a wide range of accelerated runtimes — ONNX Runtime on the server, TensorRT on NVIDIA hardware, OpenVINO on Intel, NNAPI on Android via ORT-Mobile. The `simplify` pass (via `onnxslim`) folds constants and prunes redundant nodes, yielding a leaner graph. ONNX is the deployment target for the cloud server and the conversion hub from which the platform-specific exports are derived.

### 8.2 TensorFlow Lite (INT8)

**TensorFlow Lite** is the standard on-device runtime for Android. The export (`format='tflite', int8=True, data=coco128.yaml, nms=True`) performs **full-integer INT8 post-training quantisation**: weights and activations are mapped from 32-bit floating point to 8-bit integers using per-tensor scale/zero-point parameters calibrated on real images. The benefits are an approximately **4×** reduction in model size, materially lower memory bandwidth, and access to the integer-only NPU/DSP kernels that dominate mobile accelerators — at the cost of a small, well-characterised drop in mAP. Setting `nms=True` embeds non-maximum suppression into the exported graph, so the Android client receives final, deduplicated detections directly.

### 8.3 Core ML

**Core ML** is Apple's on-device inference framework, tightly integrated with the Neural Engine across iPhone and iPad. The export (`format='coreml', int8=True, data=coco128.yaml, nms=True`) produces an `.mlpackage` that the Neural Engine can execute at very low energy cost, again with INT8 calibration and NMS baked into the model. This is a drop-in artefact for a native iOS application with no server dependency.

### 8.4 Why compression matters

Model compression is not an optional optimisation for a mobile assistive deployment; it is a deployment prerequisite. A mobile application has a hard install-size budget, a finite RAM allocation that the operating system will reclaim aggressively, and a thermal/energy envelope that throttles sustained floating-point compute. INT8 quantisation addresses all three at once: smaller artefact (faster downloads, smaller installs), reduced memory footprint and bandwidth (fewer cache misses, lower power), and computation routed onto integer accelerators (higher sustained frame rates without thermal throttling). The accompanying accuracy loss is typically a small fraction of a mAP point for a well-calibrated nano detector — a favourable exchange for the order-of-magnitude gains in deployability.

---

## 9. Discussion

### 9.1 The principle behind the decision

The defensible position is **not** that any single configuration is universally optimal, but that this *combination* — pretrained YOLO11n, deployed as released, INT8-quantised for mobile, served behind a production-grade boundary — targets a specific, well-defined operating point: **real-time general-purpose object detection on a phone for an assistive user, with a low-cost cloud fallback for debugging**. A heavier, more accurate detector would fail the latency and energy constraints. A smaller, more compressed one would surrender the COCO-class accuracy that gives the voice layer something useful to say. A re-trained variant of the same nano detector would, on the same label space, recover the same accuracy at considerable cost. The chosen configuration is the principled minimum for the brief.

### 9.2 Honesty about limits

The published numbers are a ceiling, not a floor. On Egyptian street scenes, on poor-light frames, on highly occluded or unusually-framed objects, the model will sometimes miss detections and sometimes hallucinate low-confidence ones. The Roshdi voice layer raises the confidence threshold to 0.4 for spoken announcements precisely to dampen the noise these failure modes generate. The system does not pretend to be infallible; it pretends to be **useful** at a defined operating point, and degrades gracefully when conditions stray from it.

### 9.3 When to revisit

The pretrained-first decision should be revisited when *either* of two conditions holds: (i) a Roshdi-specific labelled dataset becomes available — for example, from corrections collected via the voice-feedback loop after launch — *and* the pretrained checkpoint underperforms on it by a measurable margin; or (ii) the user-need expands beyond the 80 COCO categories. The optional pathway exists for exactly these eventualities, and the wire format is invariant under the swap, so the transition is operational rather than architectural.

---

## 10. Conclusion and future work

### 10.1 Summary of achievements

This project delivered a complete, production-grade object-detection module for the Roshdi assistive system, built around the **pretrained YOLO11n** detector and deployed without further training. The principal contribution is not a new architecture but a **clean, scripted, defensible deployment** — pretrained checkpoint, mobile export pipeline, containerised inference service with input/output safety, observability and rollback, an optional fine-tuning pathway sign-posted but not engaged — together with a critical analysis of the trade-offs that govern detection at the edge.

### 10.2 Future work

Several extensions would strengthen the deployed system:

1. **Empirical mobile measurement.** Run `benchmark.py` on actual Roshdi target devices and report ms/frame and energy per inference for FP32, FP16 and INT8 variants — converting the qualitative §6.3 trade-offs into hard numbers from the field.
2. **Domain-specific fine-tuning trigger.** Stand up the user-feedback labelling pipeline so that corrections collected post-launch flow into a Roboflow project, and activate `kaggle_pipeline.ipynb` when the per-class PSI threshold in `drift.py` is crossed.
3. **Temporal smoothing.** Add a thin temporal layer over the per-frame detections — exponential moving average of class confidences, hysteresis on appearance/disappearance — to reduce the announcement-flicker that a memoryless detector exhibits on a video stream.
4. **Architectural comparison.** Benchmark YOLO11n against YOLO11s and against EfficientDet-Lite0 on the on-device target under identical conditions, to map the accuracy–latency frontier explicitly and revalidate the nano choice with measured numbers.
5. **Coverage analysis.** Track which COCO categories actually appear in Roshdi sessions in the wild, identify the categories the user genuinely cares about most, and use the result to inform either (a) a focused fine-tune on those categories or (b) a Roshdi-specific extension to the label set.

---

## Appendix A — Reproducibility

The complete pipeline is contained in the following artefacts. **Bold** items are part of the production deployment; the rest are optional helpers.

- **`server/`** — FastAPI inference service. Auto-loads `yolo11n.pt`.
- **`export.py`** — exports the pretrained model to ONNX / TFLite-INT8 / Core ML; defaults to the pretrained checkpoint, overridable to a custom `--weights`.
- **`evaluate.py`** — evaluates a checkpoint against a `data.yaml`; defaults to the pretrained checkpoint against `coco128.yaml`.
- **`benchmark.py`** — measures p50/p95/p99 latency and single-worker RPS.
- **`drift.py`** — PSI-based per-class drift detector for production monitoring.
- **`Dockerfile`** / **`docker-compose.yml`** / **`Makefile`** — container packaging and one-command operations.
- `train.py` — *optional* fine-tuning entry point.
- `download_dataset.py` — *optional* Roboflow downloader for fine-tuning data.
- `kaggle_pipeline.ipynb` / `colab_local_runtime.ipynb` — *optional* end-to-end fine-tuning notebooks.

## Appendix B — Glossary

| Term | Definition |
|---|---|
| **mAP** | Mean Average Precision — AP averaged over all classes (and, for mAP50-95, over IoU thresholds 0.50–0.95). |
| **IoU** | Intersection over Union — area of overlap divided by area of union of two boxes. |
| **NMS** | Non-Maximum Suppression — removes duplicate overlapping detections of the same object. |
| **DFL** | Distribution Focal Loss — regresses box edges as discrete distributions for sub-pixel localisation. |
| **PAN-FPN** | Path Aggregation Network over a Feature Pyramid — bidirectional multi-scale feature fusion. |
| **SPPF** | Spatial Pyramid Pooling – Fast — cheap multi-receptive-field context aggregation. |
| **C2PSA** | Convolutional block with partial self-attention, introduced in YOLO11. |
| **INT8 quantisation** | Mapping 32-bit float weights/activations to 8-bit integers, calibrated on real data. |
| **Transfer learning** | Initialising from pretrained weights and continuing training on a target task. (Deploying the pretrained weights without further training is the zero-step limit.) |
| **PSI** | Population Stability Index — a single-number summary of how much two distributions differ; used here to flag class-distribution drift in production. |
