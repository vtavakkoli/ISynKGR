# Implementation Diagnosis

## Observed problems

1. **Near-identical scenario F1 values**: Legacy adaptive candidate ranker execution always ran the rule branch when enabled, even when adaptive strategy selected retrieval or LLM. This made many ablations effectively equivalent in execution.  
2. **Benchmark leakage in retrieval/prompting**: Candidate selection and prompt target summaries filtered toward a benchmark-specific AAS path shape, and candidate snapping used benchmark-centric behavior.  
3. **Validity reporting mismatch**: `validity_pass_rate` came from `validation.json`, but exported error summaries omitted several actual violation types and mixed validation reasons with post-hoc categories.  
4. **Incomplete stratification outputs**: Evaluation exported per-pair/per-tier, but not per-difficulty; strategy logging lacked per-sample context needed to explain adaptive behavior by pair/tier/difficulty.  

## Root causes

- Scenario flags were defined but several had little/no effect on actual control flow.
- Retrieval candidate handling encoded benchmark-specific assumptions.
- Error export used a fixed subset of categories rather than the true violation taxonomy used by validators.
- Decision logs were not joined with sample metadata.

## What was changed in this implementation pass

- Made adaptive candidate ranker strategy selection actually alter branch execution (rules vs LLM vs retrieval path).  
- Removed benchmark-shaped candidate filtering in prompt and retrieval post-processing.  
- Expanded validation reason coverage and exported full reason summaries (`error_summary.json` + CSV table aggregation).  
- Added per-difficulty metrics and strategy usage/accuracy breakdowns by pair/tier/difficulty.

## Remaining limitations (explicit)

- This pass improves realism metadata generation and reporting, but does not attempt to fully redesign canonical adapters or external model behavior.
- Runtime/token accounting still uses lightweight approximations from traces, not provider billing APIs.
