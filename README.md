# ISynKGR (Industrial Semantic Knowledge Graph Reasoner)

ISynKGR is a reproducible, Dockerized framework for industrial semantic translation + benchmarking across IEEE 1451, ISO 15926, IEC 61499, OPC UA (IEC 62541), and AAS (IEC 63278).

## Compact project layout

- `isynkgr/` → installable framework package (LLM adapter, retrieval, data generation, translation logic, benchmark runner)
- `tests/` → unit/integration tests
- `benchmarks/` → benchmark entrypoint and standard config
- `prompts/` → versioned deterministic prompts (`prompts/v1/...`)
- `scripts/` → compatibility wrappers (`.py`, `.ps1`, `.cmd`, `.sh`)
- `docker/` + `docker-compose.yml` → container runtime
- `data/demo_sources/` → minimal source snapshots
- `output/` + `cache/` → generated benchmark output + file-based LLM cache (gitignored)

## Install with pip

From repository root:

```bash
pip install .
```

CLI commands:

- `isynkgr-gen-samples`
- `isynkgr-run-bench`

## Reproducibility controls

- Fixed global seed (`145162578`)
- Deterministic prompt template (`prompts/v1/reasoning_check.txt`)
- File-based LLM cache (`cache/llm`)
- Pinned lock file (`requirements.lock`)
- Deterministic `output/benchmarks/latest` refresh

## Benchmark progress logging

`isynkgr-run-bench` prints timestamped progress messages for:

- run start and finish
- each source→target system pair
- each benchmark method per pair
- periodic sample counters
- per-system completion status

Example:

```text
[2026-02-22 09:59:06 UTC] [PAIR] (3/20) Starting ieee1451 -> opcua62541
[2026-02-22 09:59:06 UTC] [METHOD] [ieee1451 -> opcua62541] (1/5) Running method 'isynkgr' on 20 samples
[2026-02-22 09:59:06 UTC] [STATUS] Per-system progress: ieee1451:3/4, iso15926:0/4, iec61499:0/4, opcua62541:0/4, aas63278:0/4
```

## Run with Docker Compose

```bash
make up
make gen-samples
make bench
make down
```

Containers call the packaged CLI commands and connect to host Ollama via `OLLAMA_BASE_URL` (default `http://host.docker.internal:11434`).

## Windows

```powershell
ollama serve
ollama pull qwen3:0.6b
docker compose run --rm isynkgr-gen-samples
docker compose run --rm isynkgr-bench
docker compose down -v
```
