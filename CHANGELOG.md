# Changelog

## 2026-04-12 - Benchmark framework cleanup and canonicalization

- Unified orchestration so Docker `full-run` and `python -m benchmark.orchestrate` execute the same `benchmark.full_workflow` pipeline.
- Introduced canonical scenario registry (`benchmark/scenarios.py`) and mapped legacy scenario names to documented deprecated aliases.
- Standardized benchmark target-path convention to `aas://asset-<n>/submodel/default/element/<signal>/value` across dataset generation, rule shortcuts, and benchmark-shape evaluation metrics.
- Improved synthetic dataset rows with explicit semantic source metadata and deterministic per-row target candidate lists.
- Strengthened prompt constraints and ranker behavior to reduce semantic false positives (e.g., equipment labels such as Pump not treated as measurements without evidence).
- Removed duplicate legacy benchmark entrypoints and duplicate top-level docs file.
