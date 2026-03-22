"""
Feature engineering: geometry, freeze-frame features, categoricals for xG models.
Pitch: StatsBomb-style 120 x 80 yards; attacking goal center (120, 40); posts (120, 36), (120, 44).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

GOAL_CENTER = np.array([120.0, 40.0], dtype=float)
POST_LEFT = np.array([120.0, 36.0], dtype=float)
POST_RIGHT = np.array([120.0, 44.0], dtype=float)


def shot_distance(x: float, y: float) -> float:
    return float(np.hypot(120.0 - x, 40.0 - y))


def shot_angle_radians(x: float, y: float) -> float:
    """Angle subtended at (x,y) by the goal mouth, signed width in radians."""
    a = math.atan2(44.0 - y, 120.0 - x)
    b = math.atan2(36.0 - y, 120.0 - x)
    return abs(a - b)


def point_in_triangle(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _loc_xy(loc: Any) -> tuple[float, float] | None:
    if loc is None:
        return None
    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    if isinstance(loc, dict):
        return float(loc.get("x", loc.get(0))), float(loc.get("y", loc.get(1)))
    return None


def parse_freeze_frame(row: pd.Series) -> list[dict[str, Any]]:
    ff = row.get("shot_freeze_frame") or row.get("freeze_frame")
    if ff is None or (isinstance(ff, float) and np.isnan(ff)):
        return []
    if isinstance(ff, list):
        return ff
    return []


def defenders_in_shot_cone(
    shooter_x: float,
    shooter_y: float,
    freeze_frame: list[dict[str, Any]],
) -> int:
    """Count opposition players (not teammate of shooter) inside shooter–post triangle."""
    a = (shooter_x, shooter_y)
    b = (POST_LEFT[0], POST_LEFT[1])
    c = (POST_RIGHT[0], POST_RIGHT[1])
    n = 0
    for item in freeze_frame:
        if not isinstance(item, dict):
            continue
        teammate = item.get("teammate")
        if teammate is True:
            continue
        loc = item.get("location")
        xy = _loc_xy(loc)
        if xy is None:
            continue
        if point_in_triangle(xy, a, b, c):
            n += 1
    return n


def gk_off_line_error(
    shooter_x: float,
    shooter_y: float,
    freeze_frame: list[dict[str, Any]],
) -> float | None:
    """
    Perpendicular distance from the defending GK to the line shooter → goal center.
    Heuristic: opponent with max x is the GK.
    """
    gk_xy: tuple[float, float] | None = None
    best_x = -1.0
    for item in freeze_frame:
        if not isinstance(item, dict):
            continue
        if item.get("teammate") is True:
            continue
        loc = item.get("location")
        xy = _loc_xy(loc)
        if xy is None:
            continue
        if xy[0] > best_x:
            best_x = xy[0]
            gk_xy = xy
    if gk_xy is None:
        return None

    # Line from shooter S to goal center G; perpendicular distance from GK to line SG
    sx, sy = shooter_x, shooter_y
    gx, gy = GOAL_CENTER[0], GOAL_CENTER[1]
    px, py = gk_xy
    # Area of parallelogram / base length
    line_len = math.hypot(gx - sx, gy - sy)
    if line_len < 1e-6:
        return 0.0
    cross = abs((gx - sx) * (py - sy) - (gy - sy) * (px - sx))
    return cross / line_len


def closest_opponent_distance(shooter_x: float, shooter_y: float, freeze_frame: list[dict[str, Any]]) -> float | None:
    best: float | None = None
    for item in freeze_frame:
        if not isinstance(item, dict) or item.get("teammate") is True:
            continue
        xy = _loc_xy(item.get("location"))
        if xy is None:
            continue
        d = math.hypot(xy[0] - shooter_x, xy[1] - shooter_y)
        if best is None or d < best:
            best = d
    return best


def _shot_dict(row: pd.Series) -> dict[str, Any]:
    s = row.get("shot")
    return s if isinstance(s, dict) else {}


def _body_part_name(row: pd.Series) -> str:
    if pd.notna(row.get("shot_body_part")):
        return str(row["shot_body_part"])
    s = _shot_dict(row)
    bp = s.get("body_part") or {}
    return bp.get("name", "Other") if isinstance(bp, dict) else "Other"


def _play_pattern(row: pd.Series) -> str:
    pp = row.get("play_pattern")
    if isinstance(pp, dict):
        return pp.get("name", "Regular Play")
    if isinstance(pp, str):
        return pp
    return "Regular Play"


def _first_touch(row: pd.Series) -> bool:
    if pd.notna(row.get("shot_first_time")):
        return bool(row["shot_first_time"])
    s = _shot_dict(row)
    return bool(s.get("first_time", False))


def _is_open_play(row: pd.Series) -> bool:
    return _play_pattern(row) == "Regular Play"


def build_feature_frame(shots: pd.DataFrame) -> pd.DataFrame:
    """Expand shots DF into numeric/categorical columns for modeling."""
    rows: list[dict[str, Any]] = []
    for _, row in shots.iterrows():
        loc = _loc_xy(row.get("location"))
        if loc is None:
            continue
        x, y = loc
        ff = parse_freeze_frame(row)
        tid = row.get("team_id")
        if isinstance(tid, float):
            tid = int(tid) if not np.isnan(tid) else None
        elif tid is not None:
            tid = int(tid)

        dist = shot_distance(x, y)
        ang = shot_angle_radians(x, y)
        n_def_cone = defenders_in_shot_cone(x, y, ff)
        gk_err = gk_off_line_error(x, y, ff)
        opp_close = closest_opponent_distance(x, y, ff)

        sdict = _shot_dict(row)
        outcome = sdict.get("outcome") or {}
        goal = 0
        if row.get("shot_outcome") == "Goal":
            goal = 1
        elif isinstance(outcome, dict) and outcome.get("name") == "Goal":
            goal = 1

        minute = float(row.get("minute", 0)) if pd.notna(row.get("minute")) else 0.0
        up = row.get("under_pressure")
        under = bool(up) if pd.notna(up) else False
        rows.append(
            {
                "match_id": row.get("match_id"),
                "team_id": tid,
                "goal": goal,
                "x": x,
                "y": y,
                "distance": dist,
                "angle": ang,
                "under_pressure": under,
                "defenders_in_cone": n_def_cone,
                "gk_off_line": gk_err if gk_err is not None else np.nan,
                "closest_opp_dist": opp_close if opp_close is not None else np.nan,
                "minute": minute,
                "minute_norm": float(row.get("minute_norm", 0.5)),
                "goal_diff": int(row.get("goal_diff", 0)),
                "body_part": _body_part_name(row),
                "open_play": _is_open_play(row),
                "play_pattern": _play_pattern(row),
                "first_touch": _first_touch(row),
            }
        )

    return pd.DataFrame(rows)


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """One-hot body_part and goal_diff bucket; drop raw strings."""
    out = df.copy()
    mapping: dict[str, list[str]] = {}

    if "body_part" in out.columns:
        dummies = pd.get_dummies(out["body_part"], prefix="bp")
        mapping["body_part"] = list(dummies.columns)
        out = pd.concat([out.drop(columns=["body_part"]), dummies], axis=1)

    out["gd_trail_2plus"] = (out["goal_diff"] <= -2).astype(int)
    out["gd_trail_1"] = (out["goal_diff"] == -1).astype(int)
    out["gd_draw"] = (out["goal_diff"] == 0).astype(int)
    out["gd_lead_1"] = (out["goal_diff"] == 1).astype(int)
    out["gd_lead_2plus"] = (out["goal_diff"] >= 2).astype(int)

    out["open_play"] = out["open_play"].astype(int)
    out["first_touch"] = out["first_touch"].astype(int)
    out = out.drop(columns=["play_pattern", "goal_diff"], errors="ignore")

    return out, mapping
