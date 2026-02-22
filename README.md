# ISynKGR

ISynKGR is an industrial translation library for standards-aware artifact translation (hybrid LLM + KG/GraphRAG + rules), plus a container-isolated benchmark framework.

## Ollama runtime model
Ollama runs on a host/server outside Docker. Containers connect using `OLLAMA_BASE_URL` (default `http://host.docker.internal:11434`) and default model `qwen3:0.6b`.

## Python package usage (non-Docker)
```bash
pip install .
# or
pip install -e .
```

## Docker workflows
Quick baseline-first run:
```bash
docker-compose up --build quick
```

Full run (all baseline SUTs):
```bash
docker-compose up --build full
```

## Benchmark quick start (CLI)
```bash
make benchmark-small
make benchmark-full
```

## Design choices
- Container-isolated SUTs for fair baseline comparison.
- Standard adapters with explicit parse/serialize/validate.
- Mapping-level scoring + structural validity + robustness metrics.
- Reproducibility via seeds, pinned deps, manifest hashes, and provenance.

## Extend adapters
Add a new adapter in `isynkgr/adapters/`, implement `parse/serialize/validate`, and register it in `isynkgr/pipeline/hybrid.py`.

## Add a new baseline SUT
Create Dockerfile under `docker/sut/<name>/`, add service in `docker/compose/docker-compose.bench.yml`, then include baseline in `benchmark/harness.py`.


## Full-run dataset validation
Before `docker-compose up --build full`, the `dataset_validate` container runs automatically to verify dataset sufficiency and generate missing data. Generated data is persisted in `datasets/` and reused in subsequent benchmark runs.

Each benchmark suite writes progress logs (total/completed/remaining) in `results/<run_id>/logs/benchmark.log` (harness mode) and per-SUT `results/<suite>/progress.log` (compose quick/full mode).
