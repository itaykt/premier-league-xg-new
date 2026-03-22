# Implementation summary

This document summarizes what was built for the **Context-Aware Premier League xG** portfolio project: repository layout, Python pipeline, training/export script, and React web app.

## Repository structure

- **`data/raw/`**, **`data/processed/`**, **`data/predictions/`** — data locations (large/raw JSON is gitignored).
- **`models/`** — saved `.pkl` models (ignored by git; keep empty via `.gitkeep`).
- **`src/`** — Python package for data loading, features, modeling, and visualization.
- **`scripts/train_and_export.py`** — single entrypoint to load StatsBomb data, train models, and export JSON for the frontend.
- **`notebooks/`** — four notebooks (`01`–`04`) for EDA, features, training notes, and analysis.
- **`webapp/`** — Vite + React + TypeScript SPA with a dark football-themed UI.

## Python modules (`src/`)

| Module | Role |
|--------|------|
| **`data_loader.py`** | Uses **statsbombpy** (open data, no `creds=None`). Loads matches and events, filters **Shot** events (supports both dict- and flattened string `type` columns). Annotates **goal difference** at each event before goals are applied. Extracts shots with freeze frames (`shot_freeze_frame`). **Home/away team IDs** for exports are resolved by matching event team names to the matches table via `build_match_team_id_map` / `home_away_team_ids_from_events`. |
| **`features.py`** | StatsBomb pitch: goal at `(120, 40)`, posts at `(120, 36)` and `(120, 44)`. Computes **distance**, **angle**, **defenders in shot cone** (point-in-triangle), **GK off-line** (perpendicular distance to shooter–goal-center line), **closest opponent** distance, **pressure**, **minute**, **game-state buckets**, **body part** (flattened `shot_body_part` or nested shot dict), **play pattern**, **first touch**. Target: **goal** from `shot_outcome == "Goal"` or nested outcome. |
| **`model.py`** | **Logistic regression** (default) and **XGBoost** (lazy import; optional if OpenMP/libomp is available). **GroupKFold** by `match_id` for cross-validation. **Baseline** feature subset: distance, angle, body-part dummies, open play, minute. Metrics: log-loss, Brier, ROC-AUC. `META_COLS` exclude ids/coords from the feature matrix. |
| **`viz.py`** | mplsoccer **shot map** and **calibration** plot helpers for notebooks/reports. |

## Training and export (`scripts/train_and_export.py`)

1. **`load_season_shots`** for competition ID **2**, season **27** (PL 2015/16), optionally **`--max-matches`** for dev.
2. **`build_feature_frame`** → **`encode_categoricals`** → saves **`data/processed/shots_features.parquet`**.
3. Grouped CV for **full** logistic model vs **baseline** logistic model.
4. Fits final **logistic** on all rows; fits **XGBoost** if the library loads (on macOS, **`brew install libomp`** may be required).
5. Writes **`models/logistic_context.pkl`** and (when possible) **`models/xgb_context.pkl`**.
6. Builds **`data/predictions/app_data.json`**: metadata, matches (with `homeTeamId` / `awayTeamId`), per-shot **x**, **y**, **xg**, **goal**, **minute**, etc.

Team names in the export use **`_team_label`**, which handles statsbombpy’s **string** `home_team` / `away_team` columns (not only dicts).

## Web app (`webapp/`)

- **React Router** routes: Match explorer, Player dashboard (placeholder until `player_id` is in the pipeline), Team comparison, Model explainer.
- **`useAppData`** loads **`/app_data.json`** from `public/` (copy or symlink from `data/predictions/app_data.json` after training).
- **Match explorer**: cumulative xG timeline (Recharts **AreaChart**), shot map on a CSS “pitch” using **x/y** from the export.
- **Team comparison**: bar chart of xG, goals, and shot counts for two selected team IDs.
- Styling: dark theme, green accents, consistent with a football analytics feel.

## Configuration and tooling

- **`requirements.txt`** — numpy, pandas, scikit-learn, xgboost, statsbombpy, matplotlib, mplsoccer, jupyter, pyarrow, etc.
- **`.gitignore`** — venv, `__pycache__`, raw data, `node_modules`, `models/*.pkl`, etc.
- **`README.md`** — setup, training commands, web app instructions, macOS XGBoost note.

## Issues addressed during implementation

- **statsbombpy** public API: do not pass **`creds=None`**; use defaults so open data works.
- Events **`type`** is often a **string** (e.g. `"Shot"`), not a nested dict — shot filtering and goal detection were updated accordingly.
- Flattened columns (**`shot_outcome`**, **`shot_freeze_frame`**, **`shot_body_part`**, etc.) are supported alongside nested `shot` dicts.
- **Matches** dataframe from statsbombpy exposes team **names** as strings; **IDs** are derived from events for the JSON export.
- **XGBoost** on macOS may fail without **libomp**; training continues with logistic-only and logs a skip message.

## Suggested next steps (not implemented here)

- Add **`player_id`** / player names to the feature table and export for a real **Player dashboard**.
- Full-season run without `--max-matches`, calibration plots in notebooks, SHAP, blog/writeup, and deployment (e.g. Vercel for `webapp`).

---

*Generated to document the initial implementation of the xg-prem codebase.*
