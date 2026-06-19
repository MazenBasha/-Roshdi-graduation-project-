# Roshdi — Object Detection Module

YOLO11n pretrained on Microsoft COCO (80 classes), served as both an on-device
mobile artifact (TFLite / CoreML / ONNX) and a containerised HTTP service.
Built as a plug-in module for the
[Roshdi assistive AI app](https://github.com/MazenBasha/-Roshdi-graduation-project-/)
for visually-impaired users.

**Stance.** Roshdi only needs the standard 80 COCO classes (person, chair, cup,
car, bottle, …) so we deploy the **pretrained YOLO11n directly** rather than
fine-tune it. Training scripts remain in the repo as an optional future
pathway, but no training is required to run the system.

> Maturity, AI, deployment and observability requirements from the project
> rubric are answered point-by-point in [`CHECKLIST.md`](./CHECKLIST.md).
> The technical report is in [`REPORT.md`](./REPORT.md).
> The model card is in [`MODEL_CARD.md`](./MODEL_CARD.md).
> Roshdi-side integration steps are in [`INTEGRATION.md`](./INTEGRATION.md).

## 1. Quickstart — 3 commands

```bash
pip install -r requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 8000
curl -F "image=@samples/test.jpg" http://localhost:8000/v1/detect | jq
```

On first request the server auto-downloads `yolo11n.pt` (~5.5 MB) from the
Ultralytics hub and caches it. There is no dataset download and no training.

Or with Docker (weights baked into the image — fully offline):

```bash
docker compose up -d        # server :8000 + Prometheus :9090
```

## 2. Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | liveness — process is up |
| GET | `/readyz` | readiness — model loaded |
| GET | `/metrics` | Prometheus exposition |
| GET | `/v1/classes` | enumerate the 80 COCO classes |
| POST | `/v1/detect` | run detection on an uploaded image |

Response (abbreviated):
```json
{
  "request_id": "8f3e…",
  "model_version": "ab12cd34ef56",
  "inference_ms": 42.7,
  "image_width": 1280, "image_height": 720,
  "detections": [
    {"class_name": "chair", "confidence": 0.83,
     "box": {"x1": 410, "y1": 220, "x2": 760, "y2": 690},
     "horizontal_position": "center", "distance_hint": "near"}
  ]
}
```

The spatial annotations (`horizontal_position`, `distance_hint`) are designed
for the Roshdi voice layer to speak directly: *"chair, near, in the center."*

## 3. Files

```
.
├── server/                         FastAPI inference service
│   ├── main.py                     /healthz /readyz /metrics /v1/detect /v1/classes
│   ├── config.py                   pydantic-settings (env-driven)
│   ├── schemas.py                  Pydantic wire types
│   ├── inference.py                YOLO model wrapper + async timeout
│   ├── validation.py               input image guardrails
│   ├── safety.py                   output filters / PII pixelation
│   └── monitoring.py               JSON logs + Prometheus metrics
├── tests/                          pytest suite (unit, integration, adversarial)
├── flutter_integration/            drop-in Dart caller for the Roshdi app
├── monitoring/prometheus.yml       scrape config
├── export.py                       ONNX / TFLite-INT8 / CoreML
├── evaluate.py                     COCO-style metrics on coco128 by default
├── benchmark.py                    p50/p95/p99 latency
├── drift.py                        PSI-based class-distribution drift
│
├── (optional)  train.py            fine-tune on a custom dataset
├── (optional)  download_dataset.py Roboflow downloader
├── (optional)  kaggle_pipeline.ipynb
├── (optional)  colab_local_runtime.ipynb
│
├── Dockerfile, docker-compose.yml, Makefile, pyproject.toml
└── REPORT.md, MODEL_CARD.md, INTEGRATION.md, CHECKLIST.md
```

## 4. Export for mobile

The Flutter app uses the on-device exports, not the HTTP server. Generate them
once from the pretrained checkpoint:

```bash
python export.py --formats onnx tflite coreml --int8 --data coco128.yaml --nms
```

Copy the resulting `.tflite` / `.mlpackage` into the Roshdi app per
[`INTEGRATION.md`](./INTEGRATION.md). `coco128.yaml` is just used for INT8
calibration images — it auto-downloads, no Roboflow account needed.

## 5. (Advanced) Optional fine-tuning

The pretrained model is sufficient for Roshdi. If you ever want to specialise
for a particular domain — e.g. you've collected Egyptian street-scene frames
via the user-feedback loop — the original training pathway is still wired up:

```bash
cp .env.example .env                # add your Roboflow key
python download_dataset.py          # pull a custom Roboflow export
python train.py --data datasets/.../data.yaml --epochs 30
```

Or use `kaggle_pipeline.ipynb` (free GPU, resume-on-timeout) /
`colab_local_runtime.ipynb` (Colab UI, your Mac as kernel). The fine-tuned
`best.pt` becomes a drop-in replacement for `yolo11n.pt` — point
`ROSHDI_OD_WEIGHTS_PATH` at it and restart the service.

## 6. Debugging

Every log line carries a `request_id`; the same id is returned in the
`X-Request-ID` response header so a user-reported error can be traced
end-to-end.

| symptom | endpoint / signal | likely cause |
|---|---|---|
| 503 on `/v1/detect` | `/readyz` returns 503 | network down on first start (auto-download failed) → pre-load the weights manually or use the Docker image |
| 504 on `/v1/detect` | `roshdi_od_requests_total{status="timeout"}` rises | inference budget too tight or CPU starved — raise `ROSHDI_OD_INFERENCE_TIMEOUT_S` or add workers |
| spike in `bad_input` | client sending oversize / wrong content-type | check the calling Flutter code |
