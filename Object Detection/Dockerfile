# Multi-stage build. The runtime image ships with yolo11n.pt already inside,
# so the container starts and serves with zero external dependencies.
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt .
# Only the production-runtime deps. Training-only deps (roboflow, tensorflow,
# coremltools) are NOT installed here — keeps the image small.
RUN pip install --user --upgrade pip && \
    pip install --user \
        ultralytics \
        fastapi 'uvicorn[standard]' pydantic-settings prometheus-client \
        python-multipart pillow python-dotenv

# Pre-download the pretrained weights so the container starts offline.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" && \
    mkdir -p /build/weights && \
    cp /root/yolo11n.pt /build/weights/ 2>/dev/null || \
    cp ./yolo11n.pt /build/weights/ 2>/dev/null || true

# --- runtime stage ---
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH \
    ROSHDI_OD_WEIGHTS_PATH=/app/weights/yolo11n.pt

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY --from=builder /build/weights /app/weights
COPY server ./server

# Non-root user
RUN useradd -m -u 1001 roshdi && chown -R roshdi:roshdi /app
USER roshdi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/readyz || exit 1

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
