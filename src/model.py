"""
Train and evaluate baseline vs context-aware xG models (logistic regression + XGBoost).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict
import joblib

META_COLS = {
    "match_id",
    "team_id",
    "player_id",
    "player_name",
    "goal",
    "minute",
    "x",
    "y",
    "statsbomb_xg",
}


def _fill_numeric(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == object:
            continue
        X[c] = X[c].replace([np.inf, -np.inf], np.nan).fillna(X[c].median())
    return X


def split_features_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    y = df["goal"].values.astype(int)
    drop = META_COLS & set(df.columns)
    X = df.drop(columns=list(drop), errors="ignore")
    feature_names = list(X.columns)
    X = _fill_numeric(X)
    return X, y, feature_names


def train_logistic(X: pd.DataFrame, y: np.ndarray, **kwargs: Any) -> LogisticRegression:
    kwargs.setdefault("max_iter", 2000)
    kwargs.setdefault("class_weight", "balanced")
    kwargs.setdefault("random_state", 42)
    model = LogisticRegression(**kwargs)
    model.fit(X, y)
    return model


def fit_calibrated_logistic(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    calib_test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Any, LogisticRegression, dict[str, Any]]:
    """
    Train logistic on a train split; Platt scaling (sigmoid) on held-out groups only.
    Returns CalibratedClassifierCV, underlying LogisticRegression, and split metadata.
    """
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=calib_test_size,
        random_state=random_state,
    )
    idx_train, idx_cal = next(gss.split(X, y, groups))
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    lr.fit(X.iloc[idx_train], y[idx_train])
    cal = CalibratedClassifierCV(lr, method="sigmoid", cv="prefit")
    cal.fit(X.iloc[idx_cal], y[idx_cal])
    meta = {
        "method": "sigmoid_platt",
        "calib_test_size": calib_test_size,
        "n_train_shots": int(len(idx_train)),
        "n_calib_shots": int(len(idx_cal)),
        "n_train_groups": int(len(np.unique(groups[idx_train]))),
        "n_calib_groups": int(len(np.unique(groups[idx_cal]))),
    }
    return cal, lr, meta


def logistic_from_calibrated_or_plain(model: Any) -> LogisticRegression:
    """
    Underlying LogisticRegression from CalibratedClassifierCV (cv=prefit) or plain LR.
    """
    if isinstance(model, LogisticRegression):
        return model
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        est = model.calibrated_classifiers_[0].estimator
        if isinstance(est, LogisticRegression):
            return est
    raise TypeError(
        "Expected LogisticRegression or CalibratedClassifierCV (prefit sigmoid fit)"
    )


def train_xgboost(X: pd.DataFrame, y: np.ndarray, **kwargs: Any) -> Any:
    from xgboost import XGBClassifier

    kwargs.setdefault("n_estimators", 300)
    kwargs.setdefault("max_depth", 4)
    kwargs.setdefault("learning_rate", 0.05)
    kwargs.setdefault("subsample", 0.9)
    kwargs.setdefault("colsample_bytree", 0.8)
    kwargs.setdefault("random_state", 42)
    kwargs.setdefault("n_jobs", -1)
    kwargs.setdefault("eval_metric", "logloss")
    model = XGBClassifier(**kwargs)
    model.fit(X, y)
    return model


def evaluate_probs(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, p))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def calibration_bins(
    y_true: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    prob_true, prob_pred = calibration_curve(
        y_true,
        p,
        n_bins=n_bins,
        strategy="uniform",
    )
    return prob_true, prob_pred


def cross_val_grouped(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    model: Any,
    n_splits: int = 5,
) -> tuple[np.ndarray, dict[str, float]]:
    """Out-of-fold probabilities grouped by match_id to reduce leakage."""
    gkf = GroupKFold(n_splits=n_splits)
    pred = cross_val_predict(model, X, y, cv=gkf, groups=groups, method="predict_proba")[:, 1]
    metrics = evaluate_probs(y, pred)
    return pred, metrics


def baseline_feature_subset(feature_names: list[str]) -> list[str]:
    """Location-only style baseline: distance, angle, body_part dummies, open_play."""
    keep: list[str] = []
    for f in feature_names:
        if f in ("distance", "angle", "open_play", "minute_norm"):
            keep.append(f)
        if f.startswith("bp_"):
            keep.append(f)
    return keep


def save_model(model: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> Any:
    return joblib.load(path)


def logistic_interpretability_payload(
    model: LogisticRegression,
    feature_names: list[str],
) -> dict[str, Any]:
    """
    Compact summary for the fitted binary logistic model: coefs and odds ratios (exp(coef)).
    Feature rows are sorted by |coefficient| descending for quick reading.
    """
    coef = np.asarray(model.coef_, dtype=float).ravel()
    intercept = float(np.asarray(model.intercept_, dtype=float).ravel()[0])
    if len(coef) != len(feature_names):
        raise ValueError("feature_names length must match model.coef_")
    rows: list[dict[str, Any]] = []
    for name, c in zip(feature_names, coef, strict=True):
        c = float(c)
        rows.append(
            {
                "name": name,
                "coefficient": c,
                "odds_ratio": float(np.exp(c)),
            }
        )
    rows.sort(key=lambda r: abs(r["coefficient"]), reverse=True)
    return {
        "model": "logistic_context",
        "intercept": intercept,
        "features": rows,
        "note": (
            "odds_ratio = exp(coefficient): multiplicative change in odds per one-unit "
            "increase in that feature, holding others fixed (approximate; features correlate)."
        ),
    }
