# ISynKGR Benchmark Report

## Ranking by F1
|baseline|precision|recall|f1|validity_pass_rate|
|---|---|---|---|---|
|rule_only|1.000|0.100|0.182|1.000|
|graph_only|1.000|0.100|0.182|1.000|
|isynkgr_hybrid|1.000|0.100|0.182|1.000|
|rag_only|1.000|0.100|0.182|1.000|
|llm_only|1.000|0.100|0.182|1.000|

## Ranking by Validity Pass Rate
|baseline|validity_pass_rate|f1|
|---|---|---|
|rule_only|1.000|0.182|
|graph_only|1.000|0.182|
|isynkgr_hybrid|1.000|0.182|
|rag_only|1.000|0.182|
|llm_only|1.000|0.182|

## Error Taxonomy
|violation_type|count|
|---|---|
|none|0|

## Raw Results
```json
[
  {
    "precision": 1.0,
    "recall": 0.1,
    "f1": 0.18181818181818182,
    "validity_pass_rate": 1.0,
    "violation_counts": {},
    "baseline": "rule_only"
  },
  {
    "precision": 1.0,
    "recall": 0.1,
    "f1": 0.18181818181818182,
    "validity_pass_rate": 1.0,
    "violation_counts": {},
    "baseline": "graph_only"
  },
  {
    "precision": 1.0,
    "recall": 0.1,
    "f1": 0.18181818181818182,
    "validity_pass_rate": 1.0,
    "violation_counts": {},
    "baseline": "isynkgr_hybrid"
  },
  {
    "precision": 1.0,
    "recall": 0.1,
    "f1": 0.18181818181818182,
    "validity_pass_rate": 1.0,
    "violation_counts": {},
    "baseline": "rag_only"
  },
  {
    "precision": 1.0,
    "recall": 0.1,
    "f1": 0.18181818181818182,
    "validity_pass_rate": 1.0,
    "violation_counts": {},
    "baseline": "llm_only"
  }
]
```