# Context-aware Premier League xG

Portfolio project: **StatsBomb Open Data** → feature engineering (freeze‑frame geometry, game state, pressure) → **logistic regression + XGBoost** → static React dashboard.

## Project layout

- `notebooks/` — EDA and analysis (01–04)
- `src/` — `data_loader`, `features`, `model`, `viz`
- `scripts/train_and_export.py` — load shots, train models, write `data/predictions/app_data.json`
- `webapp/` — Vite + React + TypeScript (Recharts)

## Python setup

```bash
cd /path/to/xg-prem
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train models and export app data

```bash
# Full 2015/16 season (may take several minutes)
python scripts/train_and_export.py

# Quick dev run (first N matches)
python scripts/train_and_export.py --max-matches 20
```

On macOS, **XGBoost** needs OpenMP (`brew install libomp`). If the library fails to load, the script still trains **logistic regression** and exports JSON.

Outputs:

- `data/processed/shots_features.parquet`
- `models/logistic_context.pkl`, `models/xgb_context.pkl`
- `data/predictions/app_data.json`

## Web app

```bash
cd webapp
npm install
npm run dev
```

Copy or symlink the generated `data/predictions/app_data.json` to `webapp/public/app_data.json` to replace the sample bundle, or point the fetch URL in `src/hooks/useAppData.ts` at your hosted JSON.

## Primary data

- Competition ID **2**, Season ID **27** — Premier League **2015/16** (StatsBomb Big 5 open release).

## License

Code in this repository is MIT unless you specify otherwise. StatsBomb data is subject to their [open data terms](https://github.com/statsbomb/open-data).
