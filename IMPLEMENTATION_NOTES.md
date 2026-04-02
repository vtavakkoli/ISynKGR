# Implementation Notes: Hybrid Mapping Strengthening

## What changed
- Refactored `HybridPipeline` to execute retrieval + rules + LLM in hybrid mode and merge per source node with weighted scoring.
- Replaced hard-coded retrieval candidate pools with target-model-driven candidate generation (`GraphRAGRetriever`).
- Enforced source-node-local retrieval candidate usage and exposed per-node retrieval traces.
- Upgraded rule matching to use actual target canonical model input and richer heuristics (label/datatype/unit/context).
- Added stronger adaptive signals and logged per-source-node weighting decisions.
- Added high-confidence retrieval constraint mode for LLM and soft-guidance fallback for lower confidence.
- Added semantic validation checks in benchmark SUT validation path and surfaced semantic validity metric.
- Aggregated report outputs by scenario across seeds to reduce chart clutter and improve comparability.
- Added/updated tests for source-node retrieval isolation, rule-target modeling, hybrid merge behavior, and ablation differences.

## Why this changed
The previous implementation produced mostly-valid outputs while missing semantic quality due to branch collapse, weak retrieval pools, and inactive rule context. The refactor enforces true hybrid behavior and makes evidence and merge policy explicit and debuggable.

## Expected metric impact
- `full_framework` should improve exact mapping F1 and per-sample match rates versus `rule_based_only`, `rag_only`, and `llm_only`.
- Retrieval recall@k should become more meaningful because candidates are generated from target schema model nodes.
- Semantic validity should become a differentiating signal independent of syntactic/path validity.
- Strategy usage and component-contribution traces should make scenario/ablation differences observable.

## Remaining limitations
- Target canonical model bootstrapping currently uses deterministic default target-node templates per standard; integrating full external schema registries would improve realism further.
- Vector retrieval is lightweight and currently used only as a controlled score boost signal.
- LLM quality still depends on external model behavior; this refactor improves guardrails and traceability, not LLM capability itself.
