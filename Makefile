.PHONY: install install-floor lock data train test lint format

# Deterministic install from the fully pinned lockfile (recommended).
install:
	pip install --upgrade pip
	pip install -r requirements.lock

# Install from the loose floors in requirements.txt (what CI + Dependabot track).
install-floor:
	pip install --upgrade pip
	pip install -r requirements.txt

# Regenerate the lockfile from the current environment.
lock:
	pip freeze > requirements.lock

data:
	mkdir -p data/raw
	kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw --unzip

train:
	python -m src.train

test:
	pytest tests/ -v

lint:
	ruff check .

format:
	ruff format .
