.PHONY: help venv install install-dev test train api web lint

help:
	@echo "Targets:"
	@echo "  make venv          Create .venv"
	@echo "  make install       pip install -r requirements.txt"
	@echo "  make install-dev   pip install -r requirements-dev.txt"
	@echo "  make test          pytest"
	@echo "  make train         Full training export (long)"
	@echo "  make train-fast    Training on 15 matches"
	@echo "  make api           uvicorn api.main:app --reload --port 8000"
	@echo "  make web           cd webapp && npm run dev"

venv:
	@command -v python3.11 >/dev/null 2>&1 && python3.11 -m venv .venv || python3.12 -m venv .venv || python3 -m venv .venv

install:
	. .venv/bin/activate && pip install -r requirements.txt

install-dev:
	. .venv/bin/activate && pip install -r requirements-dev.txt

test:
	. .venv/bin/activate && pytest -q

train:
	. .venv/bin/activate && python scripts/train_and_export.py

train-fast:
	. .venv/bin/activate && python scripts/train_and_export.py --max-matches 15 && cp -f data/predictions/app_data.json webapp/public/app_data.json

api:
	. .venv/bin/activate && uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

web:
	cd webapp && npm run dev

lint:
	. .venv/bin/activate && ruff check src tests api scripts
