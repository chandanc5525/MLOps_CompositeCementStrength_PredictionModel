.PHONY: install ingest preprocess train evaluate predict run-all serve test clean \
        docker-build docker-run-all docker-serve

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"

ingest:
	$(PYTHON) main.py ingest

preprocess:
	$(PYTHON) main.py preprocess

train:
	$(PYTHON) main.py train

evaluate:
	$(PYTHON) main.py evaluate

predict:
	$(PYTHON) main.py predict --input data/raw/Concrete_Data.csv \
		--output artifacts/reports/predictions.csv

run-all:
	$(PYTHON) main.py run-all

serve:
	$(PYTHON) main.py serve

test:
	$(PYTHON) -m pytest -q

clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('artifacts', '.pytest_cache')]"

docker-build:
	docker build -f docker/Dockerfile -t composite-cement-strength .

docker-run-all:
	docker compose -f docker/docker-compose.yml run --rm run-all

docker-serve:
	docker compose -f docker/docker-compose.yml up serve
