"""Pitch plots and matplotlib helpers for EDA and reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from mplsoccer import Pitch
except ImportError:  # pragma: no cover
    Pitch = None  # type: ignore


def plot_shot_map(
    shots: pd.DataFrame,
    outcome_col: str = "goal",
    title: str = "Shots",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter shot locations; color by goal (requires columns from features or raw location)."""
    if Pitch is None:
        raise ImportError("Install mplsoccer for pitch plots: pip install mplsoccer")

    pitch = Pitch(pitch_type="statsbomb", line_zorder=2)
    fig, ax = pitch.draw(figsize=(10, 7))

    if "location" in shots.columns:
        xs, ys = [], []
        goals = []
        for _, row in shots.iterrows():
            loc = row["location"]
            if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                xs.append(loc[0])
                ys.append(loc[1])
            elif isinstance(loc, dict):
                xs.append(loc.get("x"))
                ys.append(loc.get("y"))
            else:
                continue
            g = row.get(outcome_col, 0)
            goals.append(int(g) if g == g else 0)
    elif "x" in shots.columns and "y" in shots.columns:
        xs = shots["x"].tolist()
        ys = shots["y"].tolist()
        goals = shots[outcome_col].fillna(0).astype(int).tolist() if outcome_col in shots.columns else [0] * len(xs)
    else:
        raise ValueError("shots need 'location' or x,y columns")

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    goals = np.array(goals, dtype=int)
    ax.scatter(xs[goals == 0], ys[goals == 0], c="#888888", s=18, alpha=0.6, label="No goal")
    ax.scatter(xs[goals == 1], ys[goals == 1], c="#00ff88", s=36, alpha=0.9, label="Goal")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_calibration(y_true: np.ndarray, probs: np.ndarray, title: str = "Calibration", n_bins: int = 10) -> plt.Figure:
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.plot(prob_pred, prob_true, "o-", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig
