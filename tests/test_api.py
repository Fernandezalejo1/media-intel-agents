"""API tests using the FastAPI test client."""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from media_intel.api.main import create_app
from tests.conftest import make_settings


@pytest.fixture()
def client(platform, tmp_path):
    from media_intel.data.seed import ensure_seeded

    ensure_seeded(platform)

    app = create_app(settings=platform.settings, platform=platform)
    # Keep checkpoint dir isolated per test.
    app.state.checkpoint_dir = str(tmp_path / "ckpts")
    return TestClient(app)


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_run_lifecycle_with_approval(client):
    created = client.post("/v1/runs", json={"objective": "Cover the El Clasico match: goals"})
    assert created.status_code == 200
    run_id = created.json()["run_id"]

    # Wait for the async task to hit the interrupt.
    deadline = time.time() + 5
    result = None
    while time.time() < deadline:
        resp = client.get(f"/v1/runs/{run_id}")
        body = resp.json()
        if body.get("status") != "running":
            result = body
            break
        time.sleep(0.05)
    assert result is not None
    assert result["status"] == "awaiting_approval"

    approvals = client.get("/v1/approvals").json()["pending"]
    assert any(p["run_id"] == run_id for p in approvals)

    decided = client.post(f"/v1/approvals/{run_id}", json={"action": "approve"})
    assert decided.status_code == 200
    assert decided.json()["result"]["outputs"]["status"] == "published"


def test_unknown_run_404(client):
    assert client.get("/v1/runs/does-not-exist").status_code == 404


def test_search_endpoint(client):
    resp = client.post("/v1/search", json={"query": "who scored the winner", "k": 2, "index": "transcripts"})
    assert resp.status_code == 200
    hits = resp.json()["results"]
    assert hits and "Bellingham" in hits[0]["text"]


def test_trace_endpoint_returns_spans(client):
    created = client.post("/v1/runs", json={"objective": "Cover the match"})
    run_id = created.json()["run_id"]
    time.sleep(0.4)
    resp = client.get(f"/v1/traces/{run_id}")
    assert resp.status_code == 200
    trace = resp.json()["trace"]
    assert trace["spans"]
    assert trace["token_usage"]["total"] > 0


def test_api_key_rejection(platform):
    from media_intel.core.settings import Settings

    secured = platform.settings.model_copy(update={"api_key": "secret"})
    app = create_app(settings=secured, platform=platform)
    client = TestClient(app)
    assert client.get("/v1/approvals").status_code == 401
    assert client.get("/healthz").status_code == 200
