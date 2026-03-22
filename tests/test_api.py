"""Smoke tests for the FastAPI app (health + predict when a model artifact exists)."""

# Pytest matches fixture names to test parameters; that triggers redefined-outer-name otherwise.
# pylint: disable=redefined-outer-name

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import ROOT, app

_MODEL = ROOT / "models" / "logistic_context.pkl"


@pytest.fixture
def client():
    with TestClient(app) as tc:
        yield tc


def test_health_get_returns_expected_shape(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "logistic_context"
    assert isinstance(body["ok"], bool)
    assert isinstance(body["n_features"], int)


@pytest.mark.skipif(
    not _MODEL.is_file(),
    reason="requires models/logistic_context.pkl (run: python scripts/train_and_export.py)",
)
def test_predict_post_happy_path(client: TestClient) -> None:
    r = client.post("/api/predict", json={"x": 95.0, "y": 40.0})
    assert r.status_code == 200
    data = r.json()
    assert "xg" in data
    assert 0.0 <= data["xg"] <= 1.0
    assert data["features_used"] > 0
