"""
Build a single-row feature vector aligned to trained model columns (for API / simulator).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features import shot_angle_radians, shot_distance


def goal_diff_to_buckets(gd: int) -> dict[str, int]:
    return {
        "gd_trail_2plus": int(gd <= -2),
        "gd_trail_1": int(gd == -1),
        "gd_draw": int(gd == 0),
        "gd_lead_1": int(gd == 1),
        "gd_lead_2plus": int(gd >= 2),
    }


def build_prediction_row(
    column_names: list[str],
    defaults: dict[str, float],
    *,
    x: float,
    y: float,
    body_part: str = "Right Foot",
    under_pressure: bool = False,
    defenders_in_cone: int = 2,
    gk_off_line: float | None = None,
    closest_opp_dist: float | None = None,
    minute_norm: float = 0.5,
    goal_diff: int = 0,
    open_play: bool = True,
    first_touch: bool = False,
) -> pd.DataFrame:
    """
    Produce one DataFrame matching training feature columns (same order as `column_names`).
    """
    dist = shot_distance(x, y)
    ang = shot_angle_radians(x, y)
    buckets = goal_diff_to_buckets(int(goal_diff))

    row: dict[str, Any] = {c: float(defaults.get(c, 0.0)) for c in column_names}

    for c in column_names:
        if c.startswith("bp_"):
            part_name = c[3:]
            row[c] = 1.0 if part_name == body_part else 0.0

    scalar_updates = {
        "distance": dist,
        "angle": ang,
        "under_pressure": float(under_pressure),
        "defenders_in_cone": float(defenders_in_cone),
        "minute_norm": float(minute_norm),
        "open_play": float(open_play),
        "first_touch": float(first_touch),
        **buckets,
    }
    for k, v in scalar_updates.items():
        if k in row:
            row[k] = float(v)

    if gk_off_line is not None and "gk_off_line" in row:
        row["gk_off_line"] = float(gk_off_line)
    if closest_opp_dist is not None and "closest_opp_dist" in row:
        row["closest_opp_dist"] = float(closest_opp_dist)

    for c in column_names:
        v = row.get(c, np.nan)
        if isinstance(v, float) and np.isnan(v):
            row[c] = float(defaults.get(c, 0.0))

    return pd.DataFrame([{c: float(row[c]) for c in column_names}], columns=column_names)


def load_feature_artifacts(root: Path | str) -> tuple[list[str], dict[str, float]]:
    root = Path(root)
    cols_path = root / "models" / "feature_columns.json"
    def_path = root / "models" / "feature_defaults.json"
    cols: list[str] = json.loads(cols_path.read_text(encoding="utf-8"))
    defaults: dict[str, float] = json.loads(def_path.read_text(encoding="utf-8"))
    return cols, defaults
