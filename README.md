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

**Manual path (same result):** use **Python 3.11 or 3.12** for the venv (not **3.13+** — see Configuration). If `.venv` was created with conda / 3.13 / 3.14, delete it first (`rm -rf .venv`).

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # or python3.12
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

![Match explorer and Model explainer UI](docs/demo.png)

*Match explorer (timeline + pitch) and Model explainer (metrics, calibration, shot simulator). Add `docs/demo.png` after you capture a run of the app.*

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

### Interpretability

After training, **`models/logistic_interpretability.json`** lists each feature’s **coefficient** (effect on log-odds of scoring) and **odds ratio** (`exp(coefficient)`), sorted by absolute coefficient so the strongest linear terms appear first.

**What the fitted model emphasizes (plain English):**

- **Geometry still drives most of the signal:** farther from goal lowers predicted chance; a wider angle to the goal mouth generally raises it—consistent with how humans judge “a good chance.”
- **Freeze-frame and pressure features move predictions beyond location:** more defenders in the shot cone and tighter defensive proximity tend to drag xG down versus the same shot shape with a clear lane.
- **Shot type and how the ball is struck** (body-part dummies, play pattern / first-touch flags) sit on top of that, separating “same spot, different execution.”

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
| `scripts/train_and_export.py` | Train, evaluate, save `models/` (including `logistic_interpretability.json`), `metrics.json`, `app_data.json` |
| `scripts/quickstart.sh` | One-shot setup + train + copy JSON to `webapp/public/` |
| `api/main.py` | FastAPI: `GET /api/health`, `POST /api/predict` |
| `src/` | Data loading, features, model helpers, prediction row builder |
| `webapp/` | Vite + React + TypeScript |
| `notebooks/` | EDA and deeper analysis |

---

## Configuration

- **Python:** Use **3.11** or **3.12** only (same as CI). **3.13+** (including **Anaconda**’s default `python`) usually makes `pip` **build** NumPy/PyArrow from source with these pins, which fails on PyArrow. Fix: `conda deactivate`, then `python3.11 -m venv .venv` (Homebrew/pyenv) — do **not** `pip install -r requirements.txt` into the conda base.
- **Data:** StatsBomb Open Data — competition **2**, season **27** (Premier League **2015/16**).
- **macOS XGBoost:** install OpenMP: `brew install libomp` if the XGBoost wheel fails to load.

---

## License

Code: MIT unless you specify otherwise. Data: [StatsBomb open-data terms](https://github.com/statsbomb/open-data).
