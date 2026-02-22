.PHONY: up gen-samples bench eval down

up:
	@echo "ISynKGR uses system Ollama. Ensure Ollama is running and model is installed: ollama pull qwen3:0.6b"

gen-samples:
	docker compose run --rm isynkgr-gen-samples

bench:
	docker compose run --rm isynkgr-bench

eval: bench

down:
	docker compose down
