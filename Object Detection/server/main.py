"""FastAPI entrypoint. ASGI app exposed as `server.main:app`.

Endpoints
---------
GET  /healthz          liveness  — process is up
GET  /readyz           readiness — model is loaded
GET  /metrics          Prometheus exposition
POST /v1/detect        run object detection on an uploaded image
GET  /v1/classes       enumerate the model's class list

Run locally:
    uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 2
"""
from __future__ import annotations

import logging
import random
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .config import Settings, get_settings
from .inference import Detector, InferenceTimeout
from .monitoring import (
    CLASS_HITS,
    DEGRADED_RESPONSES,
    DETECTIONS_PER_REQ,
    INFER_LATENCY,
    INFER_REQUESTS,
    MODEL_LOADED,
    configure_logging,
    request_id_var,
)
from .safety import filter_detections
from .schemas import DetectResponse, ErrorResponse, HealthResponse
from .validation import InvalidImage, validate_and_open

log = logging.getLogger("server")
_started = time.time()
_detector: Detector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    configure_logging(cfg.log_level)
    global _detector
    _detector = Detector(cfg)
    try:
        _detector.load()
        MODEL_LOADED.set(1)
    except Exception as e:
        log.error("model failed to load — server running in degraded mode: %s", e)
        MODEL_LOADED.set(0)
    yield
    _detector = None
    MODEL_LOADED.set(0)


app = FastAPI(
    title="Roshdi Object Detection",
    version="1.0.0",
    description="YOLO11n COCO detector served as an HTTP API for the Roshdi assistive app.",
    lifespan=lifespan,
)


@app.middleware("http")
async def assign_request_id(request: Request, call_next):
    cfg = get_settings()
    rid = request.headers.get(cfg.request_id_header) or uuid.uuid4().hex
    token = request_id_var.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers[cfg.request_id_header] = rid
    return response


def require_api_key(
    cfg: Settings = Depends(get_settings),
    x_api_key: str | None = Header(default=None),
) -> None:
    if cfg.api_key is None:
        return
    if x_api_key != cfg.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


@app.get("/healthz", response_model=HealthResponse)
async def healthz(cfg: Settings = Depends(get_settings)) -> HealthResponse:
    loaded = _detector is not None and _detector._info is not None
    version = _detector.info.version if loaded else "unloaded"
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        model_version=version,
        uptime_s=time.time() - _started,
    )


@app.get("/readyz")
async def readyz() -> Response:
    if _detector is None or _detector._info is None:
        return JSONResponse({"status": "not_ready"}, status_code=503)
    return JSONResponse({"status": "ready"})


@app.get("/metrics")
async def metrics(cfg: Settings = Depends(get_settings)) -> Response:
    if not cfg.metrics_enabled:
        raise HTTPException(status_code=404, detail="metrics disabled")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/classes")
async def classes(_: None = Depends(require_api_key)) -> dict:
    if _detector is None or _detector._info is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"classes": _detector.info.class_names, "model_version": _detector.info.version}


@app.post(
    "/v1/detect",
    response_model=DetectResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def detect(
    image: UploadFile = File(..., description="JPEG/PNG/WEBP image, ≤8 MiB"),
    cfg: Settings = Depends(get_settings),
    _: None = Depends(require_api_key),
) -> DetectResponse:
    rid = request_id_var.get()
    t0 = time.perf_counter()
    if _detector is None or _detector._info is None:
        INFER_REQUESTS.labels(status="unavailable").inc()
        DEGRADED_RESPONSES.inc()
        raise HTTPException(status_code=503, detail="model not loaded")

    body = await image.read()
    try:
        img, warnings = validate_and_open(body, image.content_type, cfg)
    except InvalidImage as e:
        INFER_REQUESTS.labels(status="bad_input").inc()
        log.warning("rejected image", extra={"ctx_reason": str(e)})
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        detections, infer_warnings = await _detector.predict(img)
    except InferenceTimeout as e:
        INFER_REQUESTS.labels(status="timeout").inc()
        DEGRADED_RESPONSES.inc()
        log.error("inference timeout", extra={"ctx_seconds": e.seconds})
        raise HTTPException(status_code=504, detail="inference timeout") from e
    except Exception as e:  # last-resort safety net
        INFER_REQUESTS.labels(status="error").inc()
        DEGRADED_RESPONSES.inc()
        log.exception("inference failure")
        raise HTTPException(status_code=500, detail="inference failed") from e

    detections = filter_detections(detections)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    INFER_LATENCY.observe(elapsed_ms / 1000)
    INFER_REQUESTS.labels(status="ok").inc()
    DETECTIONS_PER_REQ.observe(len(detections))
    for d in detections:
        CLASS_HITS.labels(class_name=d.class_name).inc()

    if random.random() < cfg.sample_predictions:
        log.info(
            "prediction sample",
            extra={
                "ctx_num_detections": len(detections),
                "ctx_classes": [d.class_name for d in detections],
                "ctx_inference_ms": elapsed_ms,
            },
        )

    return DetectResponse(
        request_id=rid,
        model_version=_detector.info.version,
        inference_ms=elapsed_ms,
        image_width=img.width,
        image_height=img.height,
        detections=detections,
        warnings=[*warnings, *infer_warnings],
    )
