# ISynKGR

**Adaptive benchmark framework for cross-standard semantic interoperability in industrial systems.**

[![CI](https://github.com/vtavakkoli/ISynKGR/actions/workflows/ci.yml/badge.svg)](https://github.com/vtavakkoli/ISynKGR/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Research software](https://img.shields.io/badge/status-research%20software-6f42c1.svg)](docs/PUBLICATION.md)

ISynKGR provides a reproducible framework for evaluating semantic mapping and translation across heterogeneous industrial standards. It combines deterministic adapters, retrieval, rules, ranking, and optional LLM-assisted reasoning with a scenario-based benchmark harness and publication-oriented reporting.

> **Publication status** — *ISynKGR: An Adaptive Benchmark Framework for Cross-Standard Semantic Interoperability* by K. Mohsenzadegan, V. Tavakkoli, and K. Kyamakya has been **accepted for presentation at AI2M4RI 2026**, in conjunction with the 23rd International Conference on Mobile Systems and Pervasive Computing (MobiSPC), Athens, Greece, August 18–20, 2026.

## Why ISynKGR

Industrial interoperability experiments are difficult to compare when datasets, mapping conventions, model settings, seeds, and evaluation procedures differ. ISynKGR makes these elements explicit and repeatable.

The repository provides:

- deterministic benchmark dataset generation and validation;
- cross-standard translation pipelines for heterogeneous industrial representations;
- canonical scenario definitions for full, baseline, and ablation experiments;
- strict evaluation with precision, recall, F1, retrieval, robustness, latency, and error diagnostics;
- multi-seed experiment orchestration with machine-readable artifacts;
- Markdown and HTML research reports;
- Docker-based execution and local Python workflows;
- tests, linting, CI, citation metadata, and reproducibility guidance.

## Supported standards

| Standard | Role in ISynKGR |
|---|---|
| OPC UA / IEC 62541 | Industrial information modelling and node-based source/target mappings |
| Asset Administration Shell (AAS) | Industry 4.0 digital-twin representation and canonical target paths |
| IEEE 1451 | Smart transducer and TEDS-oriented representations |
| IEC 61499 | Distributed industrial automation function-block representations |
| ISO 15926 | Process-plant lifecycle and semantic class representations |

Support is implemented through adapters and benchmark fixtures. Coverage varies by standard and should not be interpreted as complete implementation of each specification.

## Architecture

```mermaid
flowchart LR
    A[Standard-specific input] --> B[Adapters]
    B --> C[Canonical representation]
    C --> D[Candidate retrieval]
    D --> E[Rules and ranking]
    E --> F[Optional LLM reasoning]
    F --> G[Mapping prediction]
    G --> H[Evaluation]
    H --> I[Metrics and diagnostics]
    I --> J[Markdown / HTML report]
```

Main packages:

- `isynkgr/` — translation core, adapters, canonical models, retrieval, rules, ranking, and LLM integration;
- `benchmark/` — canonical dataset generation, orchestration, execution, evaluation, and reporting;
- `datasets/` — versioned benchmark fixtures and crosswalk ground truth;
- `tests/` — unit, contract, evaluation, and workflow tests;
- `docs/` — architecture, scenarios, publication, and reproducibility documentation.

## Quick start

### Local development

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
make check
```

### Canonical benchmark with Docker

```bash
docker compose up --build full-run
```

The default full run executes **20 seeds per configured pair**. The current benchmark configuration contains six cross-standard pairs. Override the repetition count or provide explicit seeds when needed:

```bash
RUNS_PER_PAIR=3 docker compose up --build full-run
BENCHMARK_SEEDS=11,23,37 docker compose up --build full-run
```

### Canonical benchmark with local Python

```bash
PYTHONPATH=. python -m benchmark.orchestrate
```

A reachable Ollama endpoint is required for LLM-backed scenarios. Docker defaults to `http://host.docker.internal:11434`; configure `OLLAMA_HOST`, `OLLAMA_BASE_URL`, and `MODEL_NAME` as required by your environment.

## Benchmark scenarios

The canonical scenario registry includes:

| Scenario | Purpose |
|---|---|
| `full_framework` | Complete ISynKGR pipeline |
| `rule_based_only` | Deterministic rules baseline |
| `llm_only` | LLM-only baseline |
| `rag_only` | Retrieval-augmented baseline |
| `embedding_similarity` | Similarity baseline |
| `ablation_no_rules` | Full pipeline without rules |
| `ablation_no_retrieval` | Full pipeline without retrieval |
| `ablation_no_llm` | Full pipeline without LLM reasoning |

See [`docs/SCENARIO_MATRIX.md`](docs/SCENARIO_MATRIX.md) for component-level toggles.

## Reproducibility

For a small canonical run:

```bash
RUNS_PER_PAIR=1 MAX_ITEMS=5 PYTHONPATH=. python -m benchmark.orchestrate
```

For the full configured protocol:

```bash
RUNS_PER_PAIR=20 PYTHONPATH=. python -m benchmark.orchestrate
```

Each run records outputs below `artifacts/<RUN_ID>/`, including pair-level datasets, predictions, metrics, logs, robustness analysis, error summaries, and reports. Convenience mirrors are also written below `results/`.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for environment variables, experiment controls, expected artifacts, and a reporting checklist.

## Evaluation outputs

Typical run artifacts include:

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

Headline metrics are generated from the benchmark evaluator; diagnostic exact-mapping and retrieval metrics remain available for deeper analysis. Reported results should always identify the dataset tier, pair set, scenario, seed policy, model configuration, and repository revision.

## Research scope and limitations

ISynKGR is a research benchmark and reference implementation, not a certified industrial interoperability product.

- Parts of the benchmark remain synthetic or synthetic-heavy.
- Adapter coverage is intentionally scoped and does not implement every construct of every standard.
- LLM-backed results depend on the selected model and runtime.
- Strict mapping metrics are sensitive to path canonicalization and ground-truth conventions.
- External validity requires evaluation on additional independently curated industrial datasets.

These limitations are intentional research boundaries and should be disclosed when publishing results derived from this repository.

## Citation

If you use ISynKGR in academic work, please cite the accepted AI2M4RI 2026 paper:

> Mohsenzadegan, K., Tavakkoli, V., & Kyamakya, K. (2026). **ISynKGR: An Adaptive Benchmark Framework for Cross-Standard Semantic Interoperability.** Accepted for presentation at the International Workshop on AI and Mathematical Methods for Real-world Impact (AI2M4RI), in conjunction with the 23rd International Conference on Mobile Systems and Pervasive Computing (MobiSPC), Athens, Greece, August 18–20, 2026.

```bibtex
@inproceedings{mohsenzadegan2026isynkgr,
  author    = {Kabeh Mohsenzadegan and Vahid Tavakkoli and Kyandoghere Kyamakya},
  title     = {ISynKGR: An Adaptive Benchmark Framework for Cross-Standard Semantic Interoperability},
  booktitle = {International Workshop on AI and Mathematical Methods for Real-world Impact (AI2M4RI), in conjunction with the 23rd International Conference on Mobile Systems and Pervasive Computing (MobiSPC)},
  address   = {Athens, Greece},
  year      = {2026},
  note      = {Accepted for presentation, August 18--20, 2026}
}
```

Machine-readable software citation metadata is provided in [`CITATION.cff`](CITATION.cff). See [`docs/PUBLICATION.md`](docs/PUBLICATION.md) for publication-companion guidance.

## Contributing

Contributions are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request. Scientific changes should include tests and clearly state their effect on benchmark comparability or reproducibility.

## Security

Please do not report security-sensitive information in public issues. See [`SECURITY.md`](SECURITY.md).

## License

Released under the [MIT License](LICENSE).
