# IEEE Publication-Ready Benchmark Protocol

## Problem Definition
This repository benchmarks adaptive semantic interoperability across heterogeneous industrial standards under controlled dataset tiers and cross-standard pairs.

## Dataset Generation Method
- Generator uses deterministic seeds and writes `dataset.jsonl` + `ground_truth.jsonl`.
- Pairs include `OPCUA↔AAS`, `IEEE1451↔IEC61499`, and `ISO15926↔AAS`.
- Tiers include `synthetic`, `noisy`, and `realistic`.
- Default full-run size is 180 samples.

## Benchmark Protocol
1. Run all variants across at least 3 seeds.
2. Evaluate exact-match precision/recall/F1 and retrieval hit@k.
3. Record latency, runtime, token proxy counts, and peak memory.
4. Aggregate as mean ± std with 95% CI.

## Experiment Setup
- Runtime: Docker `full-run` service.
- Model: `MODEL_NAME` env var (default `gemma4:e2b`).
- Seeds: `[11, 23, 37]` in `benchmark/full_workflow.py`.
- Dataset version: generated artifact dataset + `datasets/v1/crosswalk/gt_mappings.jsonl`.

## Error Analysis Taxonomy
- Cardinality issues
- Wrong semantic mapping
- Retrieval failures
- LLM hallucinations

## Threats to Validity
- LLM token usage is estimated with whitespace token proxy in offline mode.
- Retrieval quality depends on available graph evidence.
- Some pairs may require richer domain fixtures for external validity.

## Reproducibility
Run:

```bash
docker-compose up --build full-run
```

Expected outputs:
- `results/final_report.html`
- `artifacts/<RUN_ID>/metrics.json`
- `artifacts/<RUN_ID>/plots/*`
