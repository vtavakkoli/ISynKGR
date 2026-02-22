.PHONY: up gen-samples bench eval down

up:
	docker compose up -d ollama

gen-samples:
	docker compose run --rm isynkgr-gen-samples

bench:
	docker compose run --rm isynkgr-bench

eval: bench

down:
	docker compose down -v
