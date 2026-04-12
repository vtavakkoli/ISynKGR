# Scenario Matrix

Canonical scenarios used by both `python -m benchmark.orchestrate` and `docker compose up --build full-run`.

| Scenario | Rules | Retrieval | LLM | Adaptive selection | Candidate snap |
|---|---:|---:|---:|---:|---:|
| full_framework | ✅ | ✅ | ✅ | ✅ | ❌ |
| rule_based_only | ✅ | ❌ | ❌ | ❌ | ✅ |
| llm_only | ❌ | ❌ | ✅ | ❌ | ✅ |
| rag_only | ❌ | ✅ | ❌ | ❌ | ✅ |
| embedding_similarity | ❌ | ✅ | ❌ | ❌ | ✅ |
| ablation_no_rules | ❌ | ✅ | ✅ | ✅ | ✅ |
| ablation_no_retrieval | ✅ | ❌ | ✅ | ✅ | ✅ |
| ablation_no_llm | ✅ | ✅ | ❌ | ✅ | ✅ |

## Deprecated aliases

The following names are accepted by `benchmark.run` for compatibility, but mapped to canonical names:

- `baseline` → `rule_based_only`
- `ablation_no_graphrag` → `ablation_no_retrieval`
- `ablation_no_parallel` → `rag_only`
- `ablation_no_community` → `embedding_similarity`
- `ablation_no_reasoning` → `ablation_no_llm`
