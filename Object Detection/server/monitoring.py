"""Structured logging + Prometheus metrics."""
from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

from prometheus_client import Counter, Histogram, Gauge

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for k, v in record.__dict__.items():
            if k.startswith("ctx_"):
                payload[k[4:]] = v
        return json.dumps(payload, default=str)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # Quiet noisy libs
    for noisy in ("ultralytics", "httpx", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, getattr(logging, level)))


# --- Metrics ---
INFER_LATENCY = Histogram(
    "roshdi_od_inference_seconds",
    "End-to-end /detect latency (seconds)",
    buckets=(0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
INFER_REQUESTS = Counter(
    "roshdi_od_requests_total",
    "Total /detect requests",
    ["status"],
)
DETECTIONS_PER_REQ = Histogram(
    "roshdi_od_detections_per_request",
    "Number of detections returned per request",
    buckets=(0, 1, 2, 5, 10, 25, 50, 100),
)
CLASS_HITS = Counter(
    "roshdi_od_class_hits_total",
    "Detections per class (drift signal)",
    ["class_name"],
)
DEGRADED_RESPONSES = Counter(
    "roshdi_od_degraded_total",
    "Responses served via the fallback path",
)
MODEL_LOADED = Gauge(
    "roshdi_od_model_loaded",
    "1 if a model is loaded and ready, else 0",
)
