#!/usr/bin/env bash
# Run the full stack locally in ~2 minutes (after Node + Python are installed).
# Require 3.11/3.12: pinned numpy/pandas/pyarrow use wheels there; 3.13+ (incl. conda base) often triggers source builds that fail.
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v python3.11 &>/dev/null; then
  PY=python3.11
elif command -v python3.12 &>/dev/null; then
  PY=python3.12
else
  PY=python3
fi

if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info < (3, 13) else 1)'; then
  echo "error: $PY is Python 3.13+. This repo's pins need 3.11 or 3.12 (wheels). E.g. brew install python@3.11 && python3.11 -m venv .venv" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  "$PY" -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate

if ! python -c 'import sys; sys.exit(0 if sys.version_info < (3, 13) else 1)'; then
  echo "error: .venv is Python 3.13+. Recreate with 3.11 or 3.12 (conda base pip often uses 3.13 — use a dedicated venv)." >&2
  echo "  rm -rf .venv && bash scripts/quickstart.sh" >&2
  exit 1
fi

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
