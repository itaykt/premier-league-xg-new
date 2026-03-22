"""
FastAPI service: POST /api/predict returns xG for a synthetic shot (same features as training).

Run from repo root:
  uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import _fill_numeric, load_model
from src.prediction import build_prediction_row, load_feature_artifacts

STATE: dict[str, Any] = {
    "model": None,
    "cols": [],
    "defaults": {},
}


@asynccontextmanager
async def lifespan(_unused_app: FastAPI):
    """Load logistic model and feature metadata on startup."""
    pkl = ROOT / "models" / "logistic_context.pkl"
    if pkl.is_file():
        STATE["model"] = load_model(pkl)
        cols, defaults = load_feature_artifacts(ROOT)
        STATE["cols"] = cols
        STATE["defaults"] = defaults
    else:
        msg = (
            "WARN: models/logistic_context.pkl not found — "
            "run: python scripts/train_and_export.py"
        )
        print(msg)
    yield


app = FastAPI(title="PL xG API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictBody(BaseModel):
    """Request body for synthetic shot features (StatsBomb-style pitch coordinates)."""

    x: float = Field(
        ...,
        ge=0,
        le=120,
        description="StatsBomb pitch x (0–120), goal at high x",
    )
    y: float = Field(..., ge=0, le=80, description="StatsBomb pitch y (0–80)")
    body_part: str = "Right Foot"
    under_pressure: bool = False
    defenders_in_cone: int = Field(2, ge=0, le=22)
    gk_off_line: float | None = Field(
        None,
        description="Optional; defaults to training median",
    )
    closest_opp_dist: float | None = Field(
        None,
        description="Optional; defaults to training median",
    )
    minute_norm: float = Field(0.5, ge=0, le=1)
    goal_diff: int = Field(
        0,
        ge=-5,
        le=5,
        description="Shooting team minus opponent",
    )
    open_play: bool = True
    first_touch: bool = False


@app.get("/api/health")
def health() -> dict:
    """Return whether the model and feature schema are loaded."""
    cols = STATE["cols"]
    model = STATE["model"]
    ok = model is not None and len(cols) > 0
    return {"ok": ok, "model": "logistic_context", "n_features": len(cols)}


@app.post("/api/predict")
def predict(body: PredictBody) -> dict:
    """Return predicted xG for one row of features."""
    model = STATE["model"]
    cols = STATE["cols"]
    defaults = STATE["defaults"]
    if model is None:
        raise HTTPException(503, "Train the model first: python scripts/train_and_export.py")
    try:
        features = build_prediction_row(
            cols,
            defaults,
            x=body.x,
            y=body.y,
            body_part=body.body_part,
            under_pressure=body.under_pressure,
            defenders_in_cone=body.defenders_in_cone,
            gk_off_line=body.gk_off_line,
            closest_opp_dist=body.closest_opp_dist,
            minute_norm=body.minute_norm,
            goal_diff=body.goal_diff,
            open_play=body.open_play,
            first_touch=body.first_touch,
        )
        features = _fill_numeric(features)
        p = float(model.predict_proba(features)[0, 1])
    except Exception as e:  # pragma: no cover
        raise HTTPException(400, str(e)) from e
    return {"xg": p, "features_used": len(cols)}
