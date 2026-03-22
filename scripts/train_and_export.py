#!/usr/bin/env python3
"""
Load StatsBomb shots, engineer features, train models, export predictions for the web app.

Usage:
  python scripts/train_and_export.py
  python scripts/train_and_export.py --max-matches 15
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import (  # noqa: E402
    DEFAULT_COMPETITION_ID,
    DEFAULT_SEASON_ID,
    build_match_team_id_map,
    load_matches,
    load_season_shots,
)
from src.features import build_feature_frame, encode_categoricals  # noqa: E402
from src.model import (  # noqa: E402
    baseline_feature_subset,
    calibration_bins,
    cross_val_grouped,
    evaluate_probs,
    save_model,
    split_features_labels,
    train_logistic,
    train_xgboost,
)
from sklearn.base import clone  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _team_label(val: object, fallback: str) -> str:
    if isinstance(val, dict):
        for k in ("name", "home_team_name", "away_team_name"):
            if k in val and val[k]:
                return str(val[k])
    if isinstance(val, str) and val.strip():
        return val.strip()
    return fallback


def _match_rows_for_app(
    matches: pd.DataFrame,
    team_ids: dict[int, tuple[int | None, int | None]],
) -> list[dict]:
    rows = []
    for _, r in matches.iterrows():
        mid = int(r["match_id"])
        hid = r.get("home_team")
        aid = r.get("away_team")
        hi, ai = team_ids.get(mid, (None, None))
        rows.append(
            {
                "matchId": mid,
                "date": str(r.get("match_date", "")),
                "homeTeam": _team_label(hid, "Home"),
                "awayTeam": _team_label(aid, "Away"),
                "homeTeamId": hi,
                "awayTeamId": ai,
                "homeScore": int(r["home_score"]) if pd.notna(r.get("home_score")) else None,
                "awayScore": int(r["away_score"]) if pd.notna(r.get("away_score")) else None,
            }
        )
    return rows


def export_app_json(
    features: pd.DataFrame,
    probs: np.ndarray,
    matches: pd.DataFrame,
    team_ids: dict[int, tuple[int | None, int | None]],
    out_path: Path,
    evaluation: dict | None = None,
) -> None:
    """Bundle match list + per-shot xG for static frontend."""
    feats = features.reset_index(drop=True)
    shots_out: list[dict] = []
    for i in range(len(feats)):
        row = feats.iloc[i]
        mid = int(row["match_id"])
        tid = row.get("team_id")
        sb = row.get("statsbomb_xg")
        shot: dict = {
            "matchId": mid,
            "teamId": int(tid) if pd.notna(tid) else None,
            "minute": float(row.get("minute", 0)),
            "xg": float(probs[i]),
            "goal": int(row["goal"]),
            "x": float(row.get("x", 0)),
            "y": float(row.get("y", 0)),
            "distance": float(row.get("distance", 0)),
            "angle": float(row.get("angle", 0)),
        }
        if sb is not None and pd.notna(sb):
            shot["statsbombXg"] = float(sb)
        shots_out.append(shot)

    payload = {
        "meta": {
            "competitionId": DEFAULT_COMPETITION_ID,
            "seasonId": DEFAULT_SEASON_ID,
            "label": "Premier League 2015/16 (StatsBomb open data)",
        },
        "matches": _match_rows_for_app(matches, team_ids),
        "shots": shots_out,
    }
    if evaluation is not None:
        payload["evaluation"] = evaluation
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-matches", type=int, default=None, help="Limit matches for a fast dev run")
    ap.add_argument("--processed", type=str, default=str(ROOT / "data/processed/shots_features.parquet"))
    ap.add_argument("--out-json", type=str, default=str(ROOT / "data/predictions/app_data.json"))
    ap.add_argument("--metrics-json", type=str, default=str(ROOT / "data/predictions/metrics.json"))
    args = ap.parse_args()

    print("Loading shots (StatsBomb API / open data)...")
    shots = load_season_shots(
        DEFAULT_COMPETITION_ID,
        DEFAULT_SEASON_ID,
        max_matches=args.max_matches,
    )
    if shots.empty:
        print("No shots loaded — check network and statsbombpy.")
        sys.exit(1)

    print("Building features...")
    raw_feats = build_feature_frame(shots)
    feats, _enc_map = encode_categoricals(raw_feats)
    proc_path = Path(args.processed)
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(proc_path, index=False)
    print(f"Saved {len(feats)} rows to {proc_path}")

    X, y, names = split_features_labels(feats)
    groups = feats["match_id"].values

    # Artifacts for API / shot simulator (column order + medians)
    med = X.median(numeric_only=True)
    defaults = {c: float(med[c]) if pd.notna(med[c]) else 0.0 for c in names}
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    (ROOT / "models" / "feature_columns.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    (ROOT / "models" / "feature_defaults.json").write_text(json.dumps(defaults, indent=2), encoding="utf-8")

    oof_probs: np.ndarray | None = None
    n_splits_used: int | None = None
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        print("Skipping grouped CV (need at least 2 matches). Use without --max-matches or max-matches>=2.")
        m_log = {}
        m_base = {}
    else:
        n_splits_used = max(2, min(5, n_groups))
        print("Grouped CV — logistic regression (full features)...")
        log_full = LogisticRegression(max_iter=2000, class_weight="balanced")
        oof_probs, m_log = cross_val_grouped(X, y, groups, clone(log_full), n_splits=n_splits_used)
        print("  metrics:", m_log)

        base_cols = baseline_feature_subset(names)
        X_base = X[base_cols] if base_cols else X
        print("Grouped CV — logistic baseline (location-heavy)...")
        log_base = LogisticRegression(max_iter=2000, class_weight="balanced")
        _oof_base, m_base = cross_val_grouped(X_base, y, groups, clone(log_base), n_splits=n_splits_used)
        print("  metrics:", m_base)

    print("Training final models on full data...")
    final_log = train_logistic(X, y)
    save_model(final_log, ROOT / "models/logistic_context.pkl")
    try:
        final_xgb = train_xgboost(X, y)
        save_model(final_xgb, ROOT / "models/xgb_context.pkl")
        print("Saved XGBoost model.")
    except Exception as e:  # pragma: no cover — optional OpenMP / libomp on macOS
        print("Skipping XGBoost (install libomp or fix xgboost):", e)

    p_final = final_log.predict_proba(X)[:, 1]
    insample = evaluate_probs(y, p_final)
    print("In-sample logistic (optimistic — same data as training):", insample)

    # Evaluation block for README / frontend
    cal_pt, cal_pp = calibration_bins(y, oof_probs if oof_probs is not None else p_final, n_bins=10)
    evaluation: dict = {
        "n_shots": int(len(y)),
        "n_matches": int(n_groups),
        "cross_val_grouped_by_match": {
            "context_model": m_log,
            "baseline_location_only": m_base,
            "n_splits": n_splits_used,
            "note": "Out-of-fold predictions; grouped by match_id to reduce leakage.",
        },
        "in_sample_full_context": {
            **insample,
            "note": "Optimistic — use cross_val_* for reporting.",
        },
        "calibration": {
            "strategy": "uniform_10_bins",
            "prob_true": cal_pt.tolist(),
            "prob_pred": cal_pp.tolist(),
            "based_on": "out_of_fold" if oof_probs is not None else "in_sample",
        },
    }

    if "statsbomb_xg" in feats.columns:
        sb = feats["statsbomb_xg"].values
        mask = ~np.isnan(sb)
        if mask.sum() >= 10:
            sb_clip = np.clip(sb[mask], 1e-6, 1 - 1e-6)
            y_sb = y[mask]
            evaluation["statsbomb_official_xg"] = {
                "metrics_on_same_shots": evaluate_probs(y_sb, sb_clip),
                "note": "StatsBomb’s own xG model on the same shots (reference).",
            }
            pred_cmp = oof_probs if oof_probs is not None else p_final
            diff = np.abs(pred_cmp[mask] - sb[mask])
            r_val = float("nan")
            try:
                r_val = float(np.corrcoef(pred_cmp[mask], sb[mask])[0, 1])
            except (IndexError, ValueError):
                pass
            evaluation["model_vs_statsbomb"] = {
                "mean_absolute_error_vs_statsbomb_xg": float(np.mean(diff)),
                "pearson_r_model_vs_statsbomb": r_val,
                "n_shots_compared": int(mask.sum()),
                "predictions_used": "out_of_fold" if oof_probs is not None else "in_sample",
            }

    matches = load_matches(DEFAULT_COMPETITION_ID, DEFAULT_SEASON_ID)
    if args.max_matches:
        mids = set(shots["match_id"].unique().tolist())
        matches = matches[matches["match_id"].isin(mids)]

    print("Resolving home/away team IDs (one events fetch per match)...")
    team_ids = build_match_team_id_map(matches)

    evaluation = _json_sanitize(evaluation)

    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"Wrote {metrics_path}")

    export_app_json(feats, p_final, matches, team_ids, Path(args.out_json), evaluation=evaluation)
    print("Done.")


if __name__ == "__main__":
    main()
