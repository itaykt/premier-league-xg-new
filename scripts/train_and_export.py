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
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

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
    fit_calibrated_logistic,
    load_model,
    logistic_from_calibrated_or_plain,
    logistic_interpretability_payload,
    save_model,
    split_features_labels,
    train_logistic,
    train_xgboost,
)

LOGISTIC_PKL = ROOT / "models" / "logistic_context.pkl"
WEBAPP_PUBLIC_APP_DATA = ROOT / "webapp" / "public" / "app_data.json"


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
        pid = row.get("player_id")
        pname = row.get("player_name")
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
        if pid is not None and pd.notna(pid):
            try:
                shot["playerId"] = int(pid)
            except (TypeError, ValueError):
                pass
        if pname is not None and pd.notna(pname) and str(pname).strip():
            shot["playerName"] = str(pname).strip()
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


def main() -> None:  # pylint: disable=too-many-branches
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="Limit matches for a fast dev run",
    )
    ap.add_argument(
        "--processed",
        type=str,
        default=str(ROOT / "data/processed/shots_features.parquet"),
    )
    ap.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "data/predictions/app_data.json"),
    )
    ap.add_argument(
        "--metrics-json",
        type=str,
        default=str(ROOT / "data/predictions/metrics.json"),
    )
    ap.add_argument(
        "--skip-webapp-copy",
        action="store_true",
        help=(
            "Do not copy app_data.json to webapp/public/ "
            "(for CI or custom --out-json paths)."
        ),
    )
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
    fc_json = ROOT / "models" / "feature_columns.json"
    fd_json = ROOT / "models" / "feature_defaults.json"
    fc_json.write_text(json.dumps(names, indent=2), encoding="utf-8")
    fd_json.write_text(json.dumps(defaults, indent=2), encoding="utf-8")

    n_splits_used: int | None = None
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        print(
            "Skipping grouped CV (need at least 2 matches). "
            "Use without --max-matches or max-matches>=2.",
        )
        m_log = {}
        m_base = {}
    else:
        n_splits_used = max(2, min(5, n_groups))
        print("Grouped CV — logistic regression (full features)...")
        log_full = LogisticRegression(max_iter=2000, class_weight="balanced")
        _oof_probs, m_log = cross_val_grouped(
            X,
            y,
            groups,
            clone(log_full),
            n_splits=n_splits_used,
        )
        print("  metrics:", m_log)

        base_cols = baseline_feature_subset(names)
        X_base = X[base_cols] if base_cols else X
        print("Grouped CV — logistic baseline (location-heavy)...")
        log_base = LogisticRegression(max_iter=2000, class_weight="balanced")
        _oof_base, m_base = cross_val_grouped(
            X_base,
            y,
            groups,
            clone(log_base),
            n_splits=n_splits_used,
        )
        print("  metrics:", m_base)

    print("Training final models on full data...")
    cal_meta: dict[str, Any] = {}
    lr_train_for_meta: LogisticRegression | None = None
    if n_groups >= 2:
        final_model, lr_train, cal_meta = fit_calibrated_logistic(X, y, groups)
        lr_train_for_meta = lr_train
        save_model(final_model, LOGISTIC_PKL)
        inner_lr = logistic_from_calibrated_or_plain(final_model)
        interp = _json_sanitize(logistic_interpretability_payload(inner_lr, names))
        (ROOT / "models" / "logistic_interpretability.json").write_text(
            json.dumps(interp, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {ROOT / 'models/logistic_interpretability.json'}")
    else:
        final_log = train_logistic(X, y)
        save_model(final_log, LOGISTIC_PKL)
        interp = _json_sanitize(logistic_interpretability_payload(final_log, names))
        (ROOT / "models" / "logistic_interpretability.json").write_text(
            json.dumps(interp, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {ROOT / 'models/logistic_interpretability.json'}")
        cal_meta = {
            "note": "Calibration skipped — need at least 2 match groups for a train/holdout split.",
        }

    # Metrics + export use load_model().predict_proba(X)[:, 1] (same as API / on-disk model).
    final_loaded = load_model(LOGISTIC_PKL)
    p_final = final_loaded.predict_proba(X)[:, 1].astype(float)
    if len(p_final) != len(X):
        raise RuntimeError(f"predict_proba length {len(p_final)} != X rows {len(X)}")
    if n_groups >= 2 and lr_train_for_meta is not None:
        p_uncal = lr_train_for_meta.predict_proba(X)[:, 1]
        cal_meta["mean_predicted_probability_uncalibrated"] = float(np.mean(p_uncal))
        cal_meta["mean_predicted_probability_calibrated"] = float(np.mean(p_final))
        mu_u = cal_meta["mean_predicted_probability_uncalibrated"]
        mu_c = cal_meta["mean_predicted_probability_calibrated"]
        print("Platt (sigmoid) calibration: mean p", mu_u, "→", mu_c)
    mean_xg = float(np.mean(p_final))
    print(f"Export xG mean: {mean_xg:.6f} (saved model predict_proba, n={len(p_final)})")

    try:
        final_xgb = train_xgboost(X, y)
        save_model(final_xgb, ROOT / "models/xgb_context.pkl")
        print("Saved XGBoost model.")
    except Exception as e:  # pragma: no cover — optional OpenMP / libomp on macOS
        print("Skipping XGBoost (install libomp or fix xgboost):", e)

    insample = evaluate_probs(y, p_final)
    print("In-sample (calibrated if n_groups>=2 else raw logistic):", insample)

    # Evaluation block for README / frontend (calibration bins = same probs as app export)
    cal_pt, cal_pp = calibration_bins(y, p_final, n_bins=10)
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
            "note": (
                "Calibrated probabilities (Platt sigmoid on holdout groups) when "
                "n_groups>=2; optimistic on same rows as base LR — prefer "
                "cross_val_* for ranking."
            ),
        },
        "probability_calibration": cal_meta,
        "calibration": {
            "strategy": "uniform_10_bins",
            "prob_true": cal_pt.tolist(),
            "prob_pred": cal_pp.tolist(),
            "based_on": "exported_model_full_sample",
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
            pred_cmp = p_final
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
                "predictions_used": "exported_model_full_sample",
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

    out_json_path = Path(args.out_json)
    export_app_json(feats, p_final, matches, team_ids, out_json_path, evaluation=evaluation)
    if not args.skip_webapp_copy:
        WEBAPP_PUBLIC_APP_DATA.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_json_path, WEBAPP_PUBLIC_APP_DATA)
        print(f"Copied to {WEBAPP_PUBLIC_APP_DATA} for the Vite dev server")
    print("Done.")


if __name__ == "__main__":
    main()
