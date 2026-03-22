#!/usr/bin/env bash
# Run the full stack locally in ~2 minutes (after Node + Python are installed).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
pip install -q -r requirements.txt

echo "== Training + export (use --max-matches 15 for a fast smoke test) =="
python scripts/train_and_export.py --max-matches 15

cp -f data/predictions/app_data.json webapp/public/app_data.json
echo "== Synced app_data.json to webapp/public/ =="

echo ""
echo "Next (two terminals):"
echo "  Terminal 1:  source .venv/bin/activate && uvicorn api.main:app --reload --port 8000"
echo "  Terminal 2:  cd webapp && npm install && npm run dev"
echo "Then open the printed localhost URL (Vite proxies /api → FastAPI)."
