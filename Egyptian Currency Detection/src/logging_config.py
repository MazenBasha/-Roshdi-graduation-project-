"""
Structured JSONL logging for the detection pipeline.

Each log line is a single JSON object so logs can be tail'd through `jq` or
parsed downstream.  A human-readable copy goes to stderr.
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

_EXTRA_FIELDS = (
    "image_path",
    "num_detections",
    "inference_ms",
    "confidence_min",
    "confidence_max",
    "model_version",
    "error_type",
)


class JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON document."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "line_number": record.lineno,
        }
        for key in _EXTRA_FIELDS:
            if hasattr(record, key):
                payload[key] = getattr(record, key)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_logging(
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    name: str = "ecd",
) -> logging.Logger:
    """Configure file + console handlers.  Safe to call repeatedly."""
    if log_dir is None:
        log_dir = os.path.join(_project_root(), "outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if getattr(logger, "_ecd_configured", False):
        return logger

    logger.propagate = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"detection_{timestamp}.jsonl")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(console)

    logger._ecd_configured = True  # type: ignore[attr-defined]
    logger.debug("Logger initialized", extra={"image_path": log_path})
    return logger