# Reproducibility Guide

This document defines the recommended procedure for reproducing ISynKGR benchmark runs and reporting them in a scientifically interpretable way.

## 1. Record the software revision

Before running an experiment, record the exact Git commit:

```bash
git rev-parse HEAD
```

Also record the Python version, operating system, Docker version when applicable, and the model/runtime configuration used for LLM-backed scenarios.

## 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
make check
```

## 3. Run a small protocol first

Use a small run to verify the environment before starting the full experiment:

```bash
RUNS_PER_PAIR=1 MAX_ITEMS=5 PYTHONPATH=. python -m benchmark.orchestrate
```

LLM-backed scenarios require a reachable Ollama endpoint. Configure it explicitly when the default is not appropriate:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_HOST=http://localhost:11434
export MODEL_NAME=<model-name>
```

On Docker Desktop, the repository's Compose configuration defaults to `http://host.docker.internal:11434`.

## 4. Canonical full protocol

The current canonical configuration uses 20 runs per configured source/target pair:

```bash
RUNS_PER_PAIR=20 PYTHONPATH=. python -m benchmark.orchestrate
```

The equivalent Docker workflow is:

```bash
docker compose up --build full-run
```

To reproduce a specific seed set rather than the generated 20-seed sequence:

```bash
BENCHMARK_SEEDS=11,23,37 PYTHONPATH=. python -m benchmark.orchestrate
```

## 5. Important environment variables

| Variable | Purpose | Typical/default behavior |
|---|---|---|
| `RUNS_PER_PAIR` | Number of seeds/runs per configured pair | Canonical default: 20 |
| `BENCHMARK_SEEDS` | Explicit comma-separated seed list | Overrides `RUNS_PER_PAIR` |
| `MAX_ITEMS` | Maximum benchmark rows/items used per pair | Uses benchmark config when unset |
| `RUN_ID` | Stable output identifier | Timestamp-based ID when unset |
| `BENCHMARK_CONFIG` | Benchmark configuration file | `benchmark/benchmark_full.json` |
| `MODEL_NAME` | LLM model identifier | Runtime/Compose default when unset |
| `OLLAMA_HOST` | Ollama host used by compatible components | Environment/Compose dependent |
| `OLLAMA_BASE_URL` | Ollama base URL | Environment/Compose dependent |

Any non-default value that affects an experiment should be included in the published experiment description.

## 6. Expected artifacts

A canonical run writes a timestamped directory under `artifacts/` and mirrors convenience outputs under `results/`.

```text
artifacts/<RUN_ID>/
├── metrics.json
├── skipped_pairs.json
├── logs/
├── metrics/
│   ├── advanced_analysis.json
│   └── error_summary.csv
├── pairs/
│   └── <SOURCE>__TO__<TARGET>/
│       ├── dataset.jsonl
│       ├── ground_truth.jsonl
│       └── results/<scenario>/seed<seed>/
├── report.md
└── report.html
```

Preserve the complete run directory when results are used in a paper, technical report, or comparison.

## 7. Repeatability checks

`make reproduce` executes two small runs with the same explicit seed and separate run IDs:

```bash
make reproduce
```

Do not compare the directories byte-for-byte because runtime-dependent fields such as elapsed time can differ. Compare scientific outputs after excluding expected nondeterministic runtime metadata.

## 8. Reporting checklist

For every reported result, include:

- repository commit or tagged release;
- source and target standard pairs;
- benchmark configuration and dataset tier;
- number of items per pair;
- scenario/baseline name;
- seed list or `RUNS_PER_PAIR` policy;
- selected LLM model and runtime when applicable;
- relevant environment-variable overrides;
- precision/recall/F1 definition and evaluation mode;
- retrieval metrics when reported;
- mean, standard deviation, and confidence interval where applicable;
- skipped or unsupported pairs;
- hardware/runtime details for latency comparisons;
- limitations and threats to validity.

## 9. Comparability rules

Two experiment sets should not be presented as directly comparable if they use materially different:

- ground truth or path conventions;
- pair sets;
- dataset tiers or sample counts;
- scenario component flags;
- evaluation modes;
- seed policies;
- model families or inference settings for model-dependent claims.

When any of these differ, state the difference explicitly and treat the comparison as exploratory unless the methodology justifies normalization.

## 10. Archiving a publication run

For publication-quality evidence, archive:

1. the exact repository revision;
2. the complete `artifacts/<RUN_ID>/` directory;
3. the benchmark configuration;
4. model/runtime configuration;
5. generated reports and plots;
6. a short environment manifest.

This makes later review, replication, and error analysis substantially easier.
