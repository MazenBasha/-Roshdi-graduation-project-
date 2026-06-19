# Maturity Checklist — Roshdi Object-Detection Module

Point-by-point answers to the project rubric. Scope: **this module only**
(the YOLO11n general-object detector). The full Roshdi system's deployment,
team-contribution, and end-to-end UX answers live in the parent repo.

> **Production stance**: the module ships the **pretrained YOLO11n** as-is.
> No training is required to run, deploy, or maintain it. Training scripts
> remain in the repo as an optional pathway for future domain adaptation.
> Anything answered as "see `<file>:<line>`" is verifiable in this repo.

---

## Part 1 — Software Maturity

### 1. Final Gate
| Q | A |
|---|---|
| Deployed & live? | `pip install -r requirements.txt && make serve` or `docker compose up -d` — both bring the service up with zero training, zero dataset, zero credentials. The Docker image ships `yolo11n.pt` baked in. The on-device path ships the same weights as a TFLite asset inside the Roshdi Flutter app. |
| Multiple concurrent users? | Uvicorn runs N worker processes (`--workers 2` default in Dockerfile). Each handles requests asynchronously; the inference call runs in a thread executor so I/O & validation are not blocked. Concurrency is bounded by a per-request `inference_timeout_s`, so a stuck request can't poison the worker pool. Evidence: `benchmark.py` reports single-worker RPS; `roshdi_od_requests_total` + `roshdi_od_inference_seconds` histograms expose live concurrency in Prometheus. |
| Setup from scratch in minutes? | `README.md` §1 — five commands. |
| Explain every decision? | This file plus `REPORT.md` and `MODEL_CARD.md`. |

### 2. Architecture & Design
- **Style**: micro-service — one process exposing a single bounded context
  (object detection). It is also packaged for in-process use inside the Flutter
  app via TFLite. Both share the same artefact.
- **Trade-offs**: a monolithic server with all three Roshdi models would couple
  release cycles and force a single language. Independent modules let the team
  ship currency / face / object models on separate cadences and use the runtime
  that each model exports best to.
- **Components**:
  - `server/main.py` — HTTP boundary.
  - `server/validation.py` — guards against bad/hostile inputs.
  - `server/inference.py` — owns the model lifecycle, predicts.
  - `server/safety.py` — output filters.
  - `server/monitoring.py` — metrics + structured logs.
  - `server/config.py` — env-driven knobs (12-factor).
- **Failure points** & mitigations:
  | failure | symptom | mitigation |
  |---|---|---|
  | weights missing | startup logs error | `/readyz` stays 503, `MODEL_LOADED=0`; client falls back |
  | image decode | bad/corrupt upload | 400 with reason — `tests/test_adversarial.py` |
  | inference hang | slow CPU | hard `asyncio.wait_for` budget → 504 |
  | OOM | huge image | pixel & byte limits in `validation.py` |
  | model crash | unexpected tensor shape | catch-all → 500 with request id |
  | drift | env shifted | `drift.py` PSI alert |

### 3. Functional Completeness
- End-to-end flow: image upload → 80-class detection → structured response with
  spatial cues for the voice layer (`horizontal_position`, `distance_hint`).
  Covered by `tests/test_api.py::test_detect_returns_valid_schema`.
- Edge cases handled in code:
  - empty / oversize / wrong-mime / corrupt image → 400
  - timeout → 504
  - unknown internal failure → 500 (logged with stack + request_id)
  - model not loaded → 503

### 4. Testing
- Suite at `tests/`:
  - `test_validation.py`, `test_schemas.py`, `test_safety.py` — pure unit.
  - `test_api.py` — full FastAPI request path via `TestClient`.
  - `test_adversarial.py` — all-black, all-white, noise, extreme aspect,
    truncated JPEG, polyglot bytes, fake PNG header, zip-bomb disguise.
  - `test_inference.py` — real-model round trip, timeout enforcement,
    spatial-position semantics (skipped only with `OFFLINE=1`; the pretrained
    model is auto-downloaded when network is available).
- CI: `.github/workflows/ci.yml` lints + runs the non-model tests on every push,
  and builds the Docker image.

### 5. Performance & Load
- `benchmark.py` reports p50/p95/p99 + single-worker RPS. Numbers to fill in
  after first deploy; expected for YOLO11n FP32 on a 4-vCPU box ≈ 12–18 RPS,
  on a T4 GPU ≈ 80–120 RPS. INT8 on a phone NPU: ~30–50 FPS.
- Bottleneck: the model forward pass dominates (>90 % of wall time); validation
  and serialisation are <2 ms together. Optimisation lever is INT8 + smaller
  `imgsz`; both quantified in §8 of `REPORT.md`.

### 6. Observability
- **Logs**: JSON to stdout, one event per request, every event tagged with
  `request_id`. Format defined in `server/monitoring.py::JsonFormatter`.
- **Metrics** (Prometheus): `roshdi_od_requests_total{status=}`,
  `roshdi_od_inference_seconds` (histogram), `roshdi_od_detections_per_request`,
  `roshdi_od_class_hits_total{class_name=}`, `roshdi_od_degraded_total`,
  `roshdi_od_model_loaded` (gauge).
- **Debugging a failure**: client gives you the `X-Request-ID`; `grep` it in
  the log stream — you see validation outcome, prediction count, inference time,
  and any exception with stack trace.
- **Tracing**: request-id propagation hook exists; OpenTelemetry can be added
  in `monitoring.py` without touching handlers.

### 7. Deployment & DevOps
- **How**: multi-stage `Dockerfile` produces a slim runtime image; `docker
  compose up -d` brings the service + Prometheus.
- **One-command run**: `make docker-up`.
- **Config**: every knob is an env var with `ROSHDI_OD_` prefix; staging and
  prod differ only in their env, not the image.
- **Rollback**: image tag pinning + swap `weights/best.pt` to the previous
  artefact (we keep the SHA-256 prefix in every response, so you know exactly
  which version was serving).

### 8. Security
- **Auth**: optional `X-API-Key` header guarded in `server/main.py` —
  enforced when `ROSHDI_OD_API_KEY` is set.
- **AuthZ**: single-tenant — Roshdi app only.
- **Input safety**: bytes/pixel/mime/aspect/decompression-bomb checks
  (`validation.py`).
- **Secrets**: `.env` is gitignored. `.env.example` template committed.
  Kaggle uses *Secrets* UI; Docker uses env vars; production should use the
  host's secret manager (e.g. AWS Secrets Manager).
- **Container**: runs as non-root user `roshdi` (uid 1001).

### 9. API Design
- Versioned URL prefix (`/v1/...`). Adding a v2 field is non-breaking; removing
  one bumps to `/v2`.
- Consistent error envelope: 4xx with `{error, detail, request_id}`.
- Auto-generated OpenAPI at `/docs` (Swagger UI) and `/openapi.json`.
- All wire types are Pydantic models — request and response validated.

### 10. Data & Persistence
- This module is **stateless**: no database. The only persistent state is the
  weights file. That choice is deliberate — stateless workers scale
  horizontally and survive pre-emption.
- "Transaction half-applied" doesn't apply; each request is a pure function
  of the input bytes.

### 11. Scalability & Reliability
- **Horizontal**: stateless → put N replicas behind a load balancer.
- **Stateless**: yes by design (above).
- **Under load**: backpressure is via Uvicorn's worker queue + hard inference
  timeout; the metric `requests_total{status="timeout"}` is the SLO indicator.
- **Retries / timeouts / fallback**:
  - Inference timeout enforced (504).
  - Server returns `degraded=true` if the model failed to load at boot, so the
    Flutter caller can speak a graceful "vision currently unavailable" instead
    of crashing.
  - Flutter side uses TFLite — even if the cloud is down the user keeps
    detections.

### 12. Engineering Understanding
Every dependency is named in `requirements.txt`, every architectural choice in
`REPORT.md` §4 and `MODEL_CARD.md`. Failure mode → log signature → fix path
table at the end of `README.md`.

### 13. Reproducibility & Documentation
- `README.md` quickstart.
- `kaggle_pipeline.ipynb` reproduces training end-to-end.
- Versioned weights via SHA-256.
- `pyproject.toml` declares Python ≥ 3.10 and test config.

### 14. Cost Awareness
- **Training**: free Kaggle GPU sessions cover 30 epochs.
- **Cloud inference**: a 2 vCPU / 2 GB VM (~$5/mo) hosts the server at the
  small scale Roshdi needs. The dominant cost is *not* CPU but bandwidth if you
  upload many large images — the `max_image_bytes=8 MiB` cap bounds this.
- **On-device**: zero marginal cost.

### 15. SLA & Production Metrics
- **Availability target**: 99 % (cloud fallback only — on-device is the SLA
  for the user).
- **Latency budget**: p95 inference < 300 ms cloud, < 80 ms on-device (~12 FPS).
- **Degradation**: if `roshdi_od_inference_seconds:p95 > 1s` for 5 min, page;
  drop conf threshold to 0.4 (fewer, higher-quality detections, faster speech).

### 16. Demo Readiness
- Live service: `make docker-up` → `curl` against `/v1/detect`.
- Full flow: Flutter app captures a frame → `ObjectDetector.detect()` →
  voice layer speaks `Detection.speak()`.
- Load test: `make bench` against `samples/test.jpg`.
- Monitoring: Prometheus at `:9090` shows live histograms.
- Failure simulation: `docker compose stop od && curl /v1/detect` → 502 from
  reverse proxy or connection-refused; the Flutter side falls back to TFLite
  (path A).

---

## Part 2 — AI Component

### 0. AI Usage Justification
A rule-based system cannot enumerate the 80 object categories Roshdi needs to
announce. Hand-engineering 80 detectors is intractable; CNN-based detection
is the standard, and the pretrained YOLO11n covers exactly those 80
categories out-of-the-box. Without AI, the user gets none of the spoken
environment cues. **We use the pretrained weights as-is**: re-training on
the same label space would, in expectation, recover the same accuracy at
significant time and energy cost.

### 1. Model Understanding
- **Model**: YOLO11n (single-shot, anchor-free, decoupled head, DFL
  regression). Full architectural narrative in `REPORT.md` §4.2.
- **Limitations**: small objects, long-tail classes, heavy occlusion,
  non-standard viewpoints; INT8 cost ~1–3 mAP50-95.
- **Failure modes**: see `MODEL_CARD.md` *Failure Modes & Limitations*.

### 2. Data & Inputs
- **Inputs**: raw RGB image bytes (JPEG/PNG/WEBP) ≤ 8 MiB, ≤ 25 MP.
- **Validation**: `server/validation.py` — size, mime, header, decompression
  bomb, EXIF normalisation, mode coercion to RGB.
- **Noise / adversarial**: covered by `tests/test_adversarial.py`. The model
  itself will hallucinate low-confidence detections on noise; the conf
  threshold filters most, the area threshold filters the rest.

### 3. Evaluation & Metrics
- **Metric**: mAP@0.50, mAP@0.50:0.95 (the COCO protocol). Per-class AP.
- **Reference**: the Ultralytics-published numbers for `yolo11n.pt` on COCO
  `val2017` (mAP50-95 ≈ 0.39, mAP50 ≈ 0.55). `evaluate.py` reproduces them
  against `coco128` by default; a custom dataset can be passed via `--data`.
- **"Good enough"**: published reference accuracy (no fine-tune needed for our
  use case). For mobile, INT8 must lose < 3 mAP50-95.
- **Evidence**: `results/eval_*.json` from running `make eval`.

### 4. Testing AI Behaviour
- **Edge cases**: `tests/test_adversarial.py` (all-black, all-white, noise,
  extreme aspect, truncated/garbage bytes).
- **Determinism**: identical input → identical output (modulo float jitter
  across hardware). We seed training (`seed=42`) and disable `deterministic`
  trade-off mode for speed. The inference path is deterministic on a fixed
  device.
- **Hallucinations**: not applicable in the LLM sense, but the analogue is
  *low-confidence false positives*. Mitigated by the conf threshold + the
  area threshold + the Flutter-side 0.4 cut.

### 5. Reliability & Failure Handling
- **Wrong output**: surfaced via the user-feedback loop (the Roshdi voice
  layer logs corrections); flows into `drift.py`.
- **Timeout**: hard `asyncio.wait_for` → 504. Metric tracks it.
- **Complete failure**: `/readyz=503`. The Flutter on-device path is the
  human-in-the-loop fallback for cloud outage; the cloud path is the human-in-
  the-loop fallback for on-device install issues.

### 6. Safety & Governance
- **PII**: optional head-region pixelation on `person` boxes
  (`ROSHDI_OD_BLUR_PERSONS=true`, `server/safety.py`).
- **Harmful content**: COCO labels are non-sensitive; the `SENSITIVE_LABELS`
  hook exists to suppress any future custom label.
- **Input filter**: all hostile-input tests pass before inference runs.
- **Output filter**: confidence + area + sensitive-label filters before
  serialisation.

### 7. Prompt / Model Design
N/A — this is not an LLM. The corresponding "design" surface is the training
hyperparameters and augmentation policy, documented in `REPORT.md` §6 and
ablation-ready (every hyperparameter is a CLI flag).

### 8. System Integration
- **Where**: synchronous HTTP API at `/v1/detect`, plus an in-process TFLite
  call inside the Flutter app.
- **Sync vs async**: the call is fast (< 100 ms typical), so synchronous from
  the caller's POV is fine. The Flutter side streams camera frames at a low
  rate (≤ 5 FPS for voice) so no queueing is needed.

### 9. Performance & Cost
- **Response time**: see §5 above and `benchmark.py`.
- **Cost per request**: ~0.05 vCPU-s; effectively free at the Roshdi scale.
- **Scaling**: horizontal via more Uvicorn workers; on-device cost is zero.

### 10. Monitoring AI in Production
- **Degradation**: `roshdi_od_class_hits_total` shows the per-class
  detection-rate distribution; `drift.py` compares it to the training-time
  baseline using PSI.
- **Bad outputs**: 1 % of predictions are sampled into the log stream (full
  detection list, class names, latency). The sample rate is a config knob.
- **Trace a bad decision**: `request_id` → log → reproduces via the raw image
  body if the caller retained it.

### 11. Explainability
- Bounding box + class + confidence are the minimum visual explanation. We
  also surface the spatial annotation (`horizontal_position`,
  `distance_hint`) which is what the user actually hears.
- For deeper introspection, Ultralytics ships Grad-CAM-style visualisations
  (`yolo predict ... visualize=True`); we can wire that into a debug endpoint
  on demand.

### 12. Improvement Strategy
- **Retraining**: rerun `kaggle_pipeline.ipynb` against a new Roboflow version
  or a custom export. No code changes; the SHA-256 in `model_version` records
  the new artefact.
- **Drift response**: PSI threshold breach → retrain.
- **Feedback**: the Roshdi voice layer collects user corrections — wire them
  into a labelling queue and re-export from Roboflow on a quarterly cadence.

### 13. Ethical & Responsible AI
- **Bias**: COCO is geographically and culturally skewed. Documented in the
  model card; mitigation is a targeted Egyptian-context fine-tune
  (see `INTEGRATION.md`).
- **Privacy**: the default deployment is offline on-device — no pixel ever
  leaves the user's phone. The cloud path is opt-in and exists for debugging.
- **Fairness**: per-class AP table is part of the standard eval output.
- **Wrong / harmful answer**:
  - *Detect*: `drift.py` + user-feedback log + on-call rotation watching
    `roshdi_od_requests_total{status="error"}`.
  - *Prevent reaching user*: confidence threshold + spatial-area threshold +
    Flutter-side raised cut.
  - *Improve*: feed corrections into the next Roboflow version, retrain,
    re-export, ship the new TFLite asset in the next Roshdi release.
