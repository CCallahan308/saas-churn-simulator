.PHONY: install data test lint format figures

install:
	pip install --upgrade pip
	pip install -r requirements.txt

data:
	mkdir -p data/raw
	kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw --unzip

test:
	pytest tests/ -v

lint:
	ruff check .

format:
	ruff format .

figures:
	python notebooks/generate_figures.py
