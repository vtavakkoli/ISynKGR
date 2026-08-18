.PHONY: install install-dev lint test check \
	example-opcua-aas example-aas-opcua \
	benchmark-smoke benchmark-small benchmark-full reproduce final-report \
	docker-sample-validate docker-full-run docker-run-scenario docker-evaluate docker-report

install:
	python -m pip install --no-build-isolation -e .

install-dev:
	python -m pip install --no-build-isolation -e ".[dev]"

lint:
	ruff check isynkgr benchmark tests

test:
	PYTHONPATH=. pytest -q

check: lint test

example-opcua-aas:
	PYTHONPATH=. python examples/translate_opcua_to_aas.py

example-aas-opcua:
	PYTHONPATH=. python examples/translate_aas_to_opcua.py

benchmark-smoke:
	RUNS_PER_PAIR=1 MAX_ITEMS=5 PYTHONPATH=. python -m benchmark.orchestrate

# Backward-compatible alias for older documentation/scripts.
benchmark-small: benchmark-smoke

benchmark-full:
	RUNS_PER_PAIR=$${RUNS_PER_PAIR:-20} PYTHONPATH=. python -m benchmark.orchestrate

# Executes two identical seeded smoke protocols to make repeatability checks easy.
# Runtime fields are expected to differ; compare scientific metrics rather than raw files byte-for-byte.
reproduce:
	BENCHMARK_SEEDS=11 MAX_ITEMS=5 RUN_ID=repro_a PYTHONPATH=. python -m benchmark.orchestrate
	BENCHMARK_SEEDS=11 MAX_ITEMS=5 RUN_ID=repro_b PYTHONPATH=. python -m benchmark.orchestrate

final-report:
	PYTHONPATH=. python -m benchmark.final_report

docker-sample-validate:
	docker compose up --build sample-validate

docker-full-run:
	docker compose up --build full-run

docker-run-scenario:
	docker compose up --build run-scenario

docker-evaluate:
	docker compose run --rm evaluate

docker-report:
	docker compose run --rm report
