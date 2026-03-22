# Context-aware Premier League xG

## Run this in 2 minutes

```bash
git clone <your-repo-url> && cd xg-prem
bash scripts/quickstart.sh
```

Then open **two terminals** from the repo root:

| Terminal | Command |
|----------|---------|
| 1 — API | `source .venv/bin/activate && uvicorn api.main:app --reload --port 8000` |
| 2 — UI | `cd webapp && npm install && npm run dev` |

Open the **localhost** URL Vite prints. The dev server proxies `/api` to the API (shot simulator on the **Model explainer** page).

**Manual path (same result):**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_and_export.py --max-matches 20
cp data/predictions/app_data.json webapp/public/app_data.json
```

---

## What is this?

A **machine learning pipeline** that estimates **expected goals (xG)** — the probability a shot becomes a goal — from **StatsBomb** event data. It goes beyond “only location” models by adding **match context**: defensive pressure, **freeze-frame** geometry (defenders in the shot cone, goalkeeper positioning), and **game state**.

Deliverables:

- **Python**: feature engineering, **logistic regression** (+ optional **XGBoost**), grouped cross-validation, exported metrics.
- **React** dashboard: matches, xG timeline, pitch shot map, team comparison.
- **FastAPI** **`POST /api/predict`**: production-style inference for a synthetic shot (same features as training).

---

## Why it matters

xG is the standard language for chance quality in football analytics, broadcasting, and recruitment. Most public demos only use distance and angle. This project shows you can **engineer richer features from open data**, **evaluate honestly** (calibration, grouped CV), and **ship** a small API + UI — closer to how real analytics products are structured.

---

## Demo

Add your own screenshot or GIF here after you run the app:

`webapp` → Match explorer (timeline + pitch) and Model explainer (metrics + calibration + simulator).

Suggested path: `docs/demo.png` (optional — create when you have a capture).

---

## Model

| Piece | Detail |
|-------|--------|
| **Primary model** | Multinomial logistic regression (interpretable; common in xG literature). |
| **Optional** | Gradient boosting (**XGBoost**) when OpenMP is available (e.g. macOS: `brew install libomp`). |
| **Baseline** | “Location-heavy”: distance, angle, body-part dummies, open play, minute — same CV protocol. |
| **Validation** | **GroupKFold** by `match_id` so shots from the same match are not split across train/test folds. |

**Features (high level):** distance, angle, body part, under pressure, defenders in shot cone, GK off-line error, closest opponent distance, minute, game-state buckets, play pattern / first-touch flags.

**Reference:** StatsBomb’s own **`shot_statsbomb_xg`** is **not** used as a training input (avoid leakage / fair comparison); it is stored for **evaluation vs your model** on the same shots.

---

## 📊 Results (Grouped Cross-Validation)

Dataset: 9,908 shots across 380 matches

| Model                    | Log Loss ↓ | Brier ↓ | ROC-AUC ↑ |
|-------------------------|-----------|--------|----------|
| Full xG Model           | 0.551     | 0.183  | 0.786    |
| Location Baseline       | 0.581     | 0.196  | 0.765    |

The full model consistently outperforms a location-only baseline, demonstrating the value of contextual features beyond shot geometry.

---

## 📈 Calibration

The model shows reasonable ranking performance (ROC-AUC), but is not perfectly calibrated, tending to overestimate probabilities in higher bins.

---

## ⚖️ Comparison to StatsBomb xG

| Model                | Log Loss ↓ | Brier ↓ | ROC-AUC ↑ |
|---------------------|-----------|--------|----------|
| This model          | 0.551     | 0.183  | 0.786    |
| StatsBomb xG        | 0.255     | 0.071  | 0.809    |

StatsBomb’s model significantly outperforms this implementation, likely due to richer features such as freeze-frame player positions and defensive pressure.

---

## 🔍 Alignment with StatsBomb

- Pearson correlation: **0.72**
- Mean absolute error: **0.30**

Despite lower calibration, the model captures a similar structure of chance quality.

## Repository layout

| Path | Purpose |
|------|---------|
| `scripts/train_and_export.py` | Train, evaluate, save `models/`, `metrics.json`, `app_data.json` |
| `scripts/quickstart.sh` | One-shot setup + train + copy JSON to `webapp/public/` |
| `api/main.py` | FastAPI: `GET /api/health`, `POST /api/predict` |
| `src/` | Data loading, features, model helpers, prediction row builder |
| `webapp/` | Vite + React + TypeScript |
| `notebooks/` | EDA and deeper analysis |

---

## Configuration

- **Data:** StatsBomb Open Data — competition **2**, season **27** (Premier League **2015/16**).
- **macOS XGBoost:** install OpenMP: `brew install libomp` if the XGBoost wheel fails to load.

---

## License

Code: MIT unless you specify otherwise. Data: [StatsBomb open-data terms](https://github.com/statsbomb/open-data).
