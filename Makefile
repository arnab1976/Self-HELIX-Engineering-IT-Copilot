PY ?= python

.PHONY: install install-all data bootstrap run test lint demo up down clean

install:            ## core runtime + dev tools
	$(PY) -m pip install -r requirements.txt

install-all:        ## everything except the GPU training stack
	$(PY) -m pip install -e ".[prep,ml,stores,agents,dev]"

data:               ## generate synthetic tickets, logs, metrics, changes, warehouse
	$(PY) scripts/generate_sample_data.py

bootstrap: data     ## LEXORA: prep -> train baselines -> evaluate -> register; NEXUS: build indexes
	$(PY) scripts/bootstrap.py

run:                ## start the NEXUS API + web portal on :8000
	$(PY) -m uvicorn nexus.api.main:app --reload --port 8000

demo:               ## scripted end-to-end walkthrough against the in-process app
	$(PY) scripts/demo.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

up:                 ## full stack: API, Postgres, Qdrant, OpenSearch, Neo4j, Redis, OPA, Prometheus, Grafana, Loki, Ollama
	docker compose -f infra/docker-compose.yml up -d --build

down:
	docker compose -f infra/docker-compose.yml down

clean:
	rm -rf var data/generated .pytest_cache
