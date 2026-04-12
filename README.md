# ISynKGR Benchmark Framework

ISynKGR is a publication-oriented benchmark framework for industrial schema/path mapping across protocols (OPC UA, AAS, IEEE1451, IEC61499, ISO15926).

## Purpose

This repository provides one canonical benchmarking workflow that:
- builds deterministic benchmark datasets,
- runs scenario-based translation pipelines,
- evaluates strict exact-match metrics,
- exports diagnostics and reports.

## Architecture

- `benchmark/` — single active benchmark package (dataset build, orchestration, run/eval/report).
- `isynkgr/` — translation core (adapters, retrieval, rules, ranking, prompt construction).
- `docs/` — benchmark documentation and scenario definitions.

## Official commands

### Full canonical run (Docker)

```bash
docker compose up --build full-run
```

### Full canonical run (Local Python)

```bash
PYTHONPATH=. python -m benchmark.orchestrate
```

### Single scenario run

```bash
PYTHONPATH=. python -m benchmark.run --scenario full_framework --out results/full_framework
```

## Canonical scenario set

- `full_framework`
- `rule_based_only`
- `llm_only`
- `rag_only`
- `embedding_similarity`
- `ablation_no_rules`
- `ablation_no_retrieval`
- `ablation_no_llm`

See `docs/SCENARIO_MATRIX.md` for the exact component toggles.

## Dataset and path conventions

- Canonical AAS target convention: `aas://asset-<n>/submodel/default/element/<signal>/value`
- Ground truth, generated predictions, candidate generation, and strict evaluation all use this same convention.
- Dataset rows include semantic source metadata (`variable_role`, `datatype`, `unit`, `context_entity_id`, `description`) and deterministic `target_candidates`.
- Explicit no-match samples are encoded with `mapping_type=no_match` and empty `target_path`.

## Artifact outputs

Per run:
- `artifacts/run_<timestamp>/pairs/<SOURCE>__TO__<TARGET>/dataset.jsonl`
- `artifacts/run_<timestamp>/pairs/<SOURCE>__TO__<TARGET>/ground_truth.jsonl`
- `artifacts/run_<timestamp>/pairs/<SOURCE>__TO__<TARGET>/results/<scenario>/seed<seed>/...`
- `artifacts/run_<timestamp>/metrics.json`
- `artifacts/run_<timestamp>/report.md`
- `artifacts/run_<timestamp>/report.html`

Convenience mirror paths are also emitted under `results/<SOURCE>__TO__<TARGET>/<scenario>/seed<seed>/`.

## Limitations

- LLM-backed scenarios require a reachable Ollama endpoint.
- Cross-protocol adapter coverage is deterministic but still synthetic-heavy.
- Exact-match F1 is strict and sensitive to target-path formatting.

## Repository cleanup and migration notes

- Consolidated to one official orchestration path: `benchmark.orchestrate -> benchmark.full_workflow`.
- Removed duplicate legacy benchmark entrypoints (`benchmark/harness.py`, `benchmark/runner.py`).
- Removed duplicate top-level documentation file `WHY_NOT_OTHERS.md` (canonical copy remains in `docs/`).
- Unified scenario naming and deprecated alias mapping in `benchmark/scenarios.py`.
