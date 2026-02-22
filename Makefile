.PHONY: install lint test example-opcua-aas example-aas-opcua benchmark-small benchmark-full reproduce docker-quick docker-full

install:
	pip install --no-build-isolation -e .

lint:
	ruff check isynkgr benchmark tests

test:
	PYTHONPATH=. pytest -q

example-opcua-aas:
	PYTHONPATH=. python examples/translate_opcua_to_aas.py

example-aas-opcua:
	PYTHONPATH=. python examples/translate_aas_to_opcua.py

benchmark-small:
	PYTHONPATH=. python -m benchmark.harness

benchmark-full:
	FULL=1 PYTHONPATH=. python -m benchmark.harness

reproduce:
	PYTHONPATH=. python -m benchmark.harness && PYTHONPATH=. python -m benchmark.harness


docker-quick:
	docker-compose up --build quick

docker-full:
	docker-compose up --build full
