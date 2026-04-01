# Scenario Matrix

| Scenario | Rules | Retrieval | LLM | Adaptive selection | Candidate snap | Reasoning prompt | Community filter | Parallel retrieval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_framework | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| rule_based_only | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| llm_only | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| rag_only | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| embedding_similarity | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| ablation_no_rules | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ablation_no_retrieval | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ablation_no_graph_expansion | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| ablation_no_llm | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ablation_no_reasoning_prompt | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| ablation_no_community_filter | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| ablation_no_parallel_retrieval | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

Notes:
- `candidate snap` is intentionally disabled for `full_framework` to avoid benchmark-specific target reconstruction behavior.
- `embedding_similarity` uses retrieval evidence without adaptive switching, approximating a simple non-LLM retrieval baseline.
