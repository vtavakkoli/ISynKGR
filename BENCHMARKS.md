# Benchmarks

## Pipelines
- Full ISynKGR (KG retrieval + reasoning check + translation logic provenance)
- Baseline 1: Vanilla RAG (heuristic retrieval, no KG reasoning)
- Baseline 2: LLM-only
- Baseline 3: KG-only symbolic mapping
- Baseline 4: Graph retrieval only

## Standard pairs
All directed pairs among:
- IEEE 1451
- ISO 15926
- IEC 61499
- OPC UA (IEC 62541)
- AAS (IEC 63278)

## Metrics
- Precision, Recall, F1, Accuracy
- Property mapping accuracy
- Latency (avg and total)
- Token count (approx via Ollama eval counters)
- KG traversal statistics
- CPU time + peak RSS (best effort)

## Ablations
`--no-graphrag --no-cot --no-community --no-parallel-retrievers`
