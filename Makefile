# AI Annotation & Served Inference
#
#   make setup    install dependencies
#   make data     download BANKING77
#   make run      the full pipeline: retrieve -> ablate -> calibrate -> gate -> distil -> cost
#   make test     the suite
#   make serve    the classifier UI on :8600
#   make all      data -> run -> test

PY ?= python3

.PHONY: setup data run test serve docker clean all

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) data/fetch.py

run:
	$(PY) -m src.report --test-size 1500

quick:
	$(PY) -m src.report --test-size 400 --few-shot-subset 100

test:
	$(PY) -m pytest tests/ -q

serve:
	$(PY) -m uvicorn serve.app:app --host 127.0.0.1 --port 8600

docker:
	docker compose up --build

all: data run test

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache artifacts/ablation_cache.json
