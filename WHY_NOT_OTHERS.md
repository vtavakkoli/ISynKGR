# Why not other frameworks?

ISynKGR is built for industrial translation constraints where common frameworks are weak:

1. **Deterministic benchmarking**: fixed seeds, fixed prompts, and cache-backed local inference avoid benchmark drift.
2. **Auditability/provenance**: each mapping stores prompt template, model version, and retrieved subgraph IDs.
3. **Air-gapped/offline readiness**: Ollama-hosted Qwen3:0.6b runs locally without paid APIs.
4. **KG-centered reasoning**: graph traversal and neighborhood evidence are first-class, rather than generic text-only retrieval.
5. **Data locality + compliance**: all datasets, GT, outputs, and caches remain inside local containers/volumes.
