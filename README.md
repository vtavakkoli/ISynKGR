# ISynKGR

ISynKGR is an industrial translation library for standards-aware artifact translation (hybrid LLM + KG/GraphRAG + rules), plus a container benchmark framework.

Benchmark orchestration is canonical under `benchmark/` (legacy `isynkgr.evaluation.*` modules are deprecated compatibility wrappers).

## How to run benchmark

### 1) Fast fail-first sample validation
```bash
docker compose up --build sample-validate
```

### 2) Full benchmark pipeline (sample validation -> all scenarios -> evaluation -> final report)
```bash
docker compose up --build full-run
```

Scenarios executed:
- baseline
- full_framework
- ablation_no_graphrag
- ablation_no_parallel
- ablation_no_community
- ablation_no_reasoning

## How to reproduce report

After `full-run`, final outputs are in:
- `results/final/report.md`
- `results/final/metrics_merged.json`
- `results/final/charts/*.png`

You can regenerate only report artifacts with:
```bash
docker compose run --rm report
```

## Observability / Logs

- All Python services run with unbuffered output (`python -u`, `PYTHONUNBUFFERED=1`).
- Pipeline progress streams live to compose stdout.
- Per-scenario logs are persisted at `results/<scenario>/logs/run.log`.
- Scenario outputs include:
  - `results/<scenario>/config_resolved.json`
  - `results/<scenario>/ground_truth.jsonl`
  - `results/<scenario>/dataset.jsonl`
  - `results/<scenario>/predictions/mappings.jsonl`
  - `results/<scenario>/metrics.json`

## Determinism

- Use fixed `SEED` (default 42) and explicit `MODEL_NAME`.
- Resolved runtime configuration is written per scenario.
- Git hash is logged when available.


## Mapping Path Validation

ISynKGR now validates mapping `source_path`/`target_path` using deterministic, protocol-specific regexes in `isynkgr/icr/path_validation.py`:

- **AAS**: `aas://{aas_id}/submodel/{sm_idShort}/element/{path...}`
- **OPC UA**: `opcua://ns={ns};s={string_id}` or `opcua://ns={ns};i={int_id}`
- **IEC 61499 (subset)**: `iec61499://{device}/{resource}/{fb}/{var}`
- **IEEE 1451 (subset)**: `ieee1451://{ted_id}/{channel}/{field}`

Implementation notes:
- Patterns are **fully anchored** (`^...$`).
- Segment classes use bounded character sets (`[A-Za-z0-9._~-]+`) with no catastrophic backtracking constructs.
- Validation is shared and reused by ingestion, pipeline output checks, and mapping validity checks.
