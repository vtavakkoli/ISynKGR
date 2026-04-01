# Qualitative Examples

This file is populated from exported run artifacts under `artifacts/<run_id>/predictions/*`.

## Status

No fabricated examples are included in source control. After running:

- `docker compose up --build full-run`
- or `python -m benchmark.orchestrate`

use `predictions/sample_results.jsonl`, `predictions/decision_trace.jsonl`, and `error_analysis.json` to select and document:

1. one case where rules help,
2. one case where retrieval helps,
3. one case where LLM helps,
4. one case where adaptive selection helps,
5. one failure case with explanation.
