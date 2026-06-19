"""HTTP-level integration tests using FastAPI's TestClient.

These exercise the full request path: validation -> model -> response.
Tests that need real weights are marked with `needs_weights`.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import needs_weights


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("ROSHDI_OD_API_KEY", raising=False)
    from server.main import app
    with TestClient(app) as c:
        yield c


def test_healthz_always_responds(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "uptime_s" in body


def test_metrics_exposed(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"roshdi_od_requests_total" in r.content


def test_request_id_round_trip(client):
    r = client.get("/healthz", headers={"X-Request-ID": "deadbeef"})
    assert r.headers["X-Request-ID"] == "deadbeef"


def test_request_id_generated_when_missing(client):
    r = client.get("/healthz")
    assert len(r.headers["X-Request-ID"]) >= 16


def test_detect_rejects_empty_upload(client):
    r = client.post("/v1/detect", files={"image": ("x.png", b"", "image/png")})
    # 400/422 if model is loaded (validation rejects), 503 if model isn't loaded.
    # Either way: never 500.
    assert r.status_code in (400, 422, 503)


def test_detect_rejects_garbage(client):
    r = client.post("/v1/detect", files={"image": ("x.png", b"hello", "image/png")})
    assert r.status_code in (400, 503)  # 503 if model failed to load in CI


def test_api_key_enforced(monkeypatch):
    monkeypatch.setenv("ROSHDI_OD_API_KEY", "s3cret")
    from server.config import get_settings
    get_settings.cache_clear()
    from server.main import app
    with TestClient(app) as c:
        r = c.get("/v1/classes")
        assert r.status_code in (401, 503)


@needs_weights
def test_detect_returns_valid_schema(client, png_bytes):
    r = client.post(
        "/v1/detect",
        files={"image": ("x.png", png_bytes(size=(640, 480)), "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "detections" in body
    assert "inference_ms" in body
    assert body["image_width"] == 640
    assert body["image_height"] == 480
    assert "request_id" in body
