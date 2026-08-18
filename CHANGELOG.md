# Changelog

## 2026-08-18 - Research repository professionalization

- Reworked the README into a publication-quality research-software landing page with architecture, standards coverage, benchmark scenarios, reproducibility guidance, limitations, and academic citation information.
- Added `CITATION.cff`, publication-companion documentation, contributor guidance, security guidance, a code of conduct, and a pull-request template.
- Fixed stale `Makefile` targets that still referenced the removed `benchmark.harness` entrypoint; benchmark targets now use the canonical `benchmark.orchestrate` workflow.
- Added a development install target and consolidated `make check` for linting and tests.
- Expanded package metadata for research discovery and added pytest/ruff configuration.
- Strengthened GitHub Actions CI with Python 3.10/3.12 coverage, dependency checks, compilation checks, and package build validation.
- Updated reproducibility and publication-readiness documentation to align with the current 20-run-per-pair canonical protocol.

## 2026-04-12 - Benchmark framework cleanup and canonicalization

- Unified orchestration so Docker `full-run` and `python -m benchmark.orchestrate` execute the same `benchmark.full_workflow` pipeline.
- Introduced canonical scenario registry (`benchmark/scenarios.py`) and mapped legacy scenario names to documented deprecated aliases.
- Standardized benchmark target-path convention to `aas://asset-<n>/submodel/default/element/<signal>/value` across dataset generation, rule shortcuts, and benchmark-shape evaluation metrics.
- Improved synthetic dataset rows with explicit semantic source metadata and deterministic per-row target candidate lists.
- Strengthened prompt constraints and ranker behavior to reduce semantic false positives (e.g., equipment labels such as Pump not treated as measurements without evidence).
- Removed duplicate legacy benchmark entrypoints and duplicate top-level docs file.
