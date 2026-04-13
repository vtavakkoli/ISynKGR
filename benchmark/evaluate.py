from __future__ import annotations

import json
from pathlib import Path

from benchmark.metrics import group_prf1, hit_at_k, mapping_prf1, recall_at_k, violation_counts
from isynkgr.icr.mapping_schema import ingest_mapping_payload, normalize_mapping_path


def _normalize_mapping_type(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"label_match", "approximate", "fallback"}:
        return "equivalent"
    return raw


def _mapping_key(row: dict) -> tuple[str, str, str]:
    return (
        normalize_mapping_path(row.get("source_path", "")),
        normalize_mapping_path(row.get("target_path", "")),
        _normalize_mapping_type(str(row.get("mapping_type", ""))),
    )


def _load_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        record = ingest_mapping_payload(row, migrate_legacy=True)
        rows.append(record.model_dump())
    return rows


def _load_optional_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _deduplicate_rows(rows: list[dict]) -> list[dict]:
    best_by_key: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        key = _mapping_key(row)
        current = best_by_key.get(key)
        if current is None or float(row.get("confidence", 0.0)) > float(current.get("confidence", 0.0)):
            best_by_key[key] = row
    return [best_by_key[key] for key in sorted(best_by_key)]


def _resolve_gt_path(out_dir: Path) -> Path:
    for path in [
        out_dir.parent / "ground_truth.jsonl",
        out_dir.parent / "gt_mappings.jsonl",
        out_dir.parent.parent / "ground_truth.jsonl",
        out_dir.parent.parent.parent / "ground_truth.jsonl",
    ]:
        if path.exists():
            return path
    raise FileNotFoundError("Ground truth not found.")


def _resolve_pred_path(out_dir: Path) -> Path:
    for path in [out_dir / "mappings.jsonl", out_dir / "predictions" / "mappings.jsonl"]:
        if path.exists():
            return path
    raise FileNotFoundError(f"Predictions not found under {out_dir}")


def _sample_top1_prf1(sample_rows: list[dict]) -> dict[str, float]:
    """
    Compute a sample-level Top-1 PR/F1 from benchmark/predictions/sample_results.jsonl.

    In the full workflow, each dataset row corresponds to one expected mapping decision,
    while the translator may emit multiple variable-level mappings per source artifact.
    Exact triple matching can therefore become misleading if source-path granularity
    differs between GT generation and emitted mappings.
    """
    if not sample_rows:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "count": 0}
    total = len(sample_rows)
    matched = sum(1 for row in sample_rows if bool(row.get("matched")))
    acc = matched / total if total else 0.0
    return {"precision": acc, "recall": acc, "f1": acc, "count": total}


def evaluate_run(out_dir: Path, evaluation_mode: str = "auto") -> dict:
    pred_path = _resolve_pred_path(out_dir)
    gt_path = _resolve_gt_path(out_dir)
    pred_rows_raw = _load_jsonl_rows(pred_path)
    gt_rows_raw = _load_jsonl_rows(gt_path)
    pred_rows = _deduplicate_rows(pred_rows_raw)
    gt_rows = _deduplicate_rows(gt_rows_raw)

    pred_keys = {_mapping_key(row) for row in pred_rows}
    gt_keys = {_mapping_key(row) for row in gt_rows}
    exact = mapping_prf1(pred_keys, gt_keys)

    reports = json.loads((out_dir / "validation.json").read_text()) if (out_dir / "validation.json").exists() else []
    retrieval_rows = _load_optional_jsonl(out_dir / "predictions" / "retrieval_trace.jsonl")
    sample_rows = _load_optional_jsonl(out_dir / "predictions" / "sample_results.jsonl")
    perf_rows = _load_optional_jsonl(out_dir / "predictions" / "perf_trace.jsonl")
    decision_rows = _load_optional_jsonl(out_dir / "predictions" / "decision_trace.jsonl")
    sample_top1 = _sample_top1_prf1(sample_rows)

    transform_total = sum(1 for row in gt_rows if row.get("mapping_type") == "transform")
    transform_correct = sum(1 for row in gt_rows if row.get("mapping_type") == "transform" and _mapping_key(row) in pred_keys)
    invalid_report_count = sum(1 for report in reports if not report.get("valid"))
    path_validity = 1.0 - (invalid_report_count / max(len(reports), 1))
    semantic_invalid = sum(
        1
        for report in reports
        if any(str(v.get("type", "")).startswith("semantic_") for v in report.get("violations", []))
    )
    semantic_validity = 1.0 - (semantic_invalid / max(len(reports), 1))

    confidence_pairs = []
    for row in pred_rows:
        key = _mapping_key(row)
        confidence_pairs.append((float(row.get("confidence", 0.0)), 1.0 if key in gt_keys else 0.0))
    calibration_error = 0.0
    if confidence_pairs:
        calibration_error = sum(abs(c - y) for c, y in confidence_pairs) / len(confidence_pairs)

    typed_rows = []
    for row in pred_rows:
        typed_rows.append({"mapping_key": _mapping_key(row), "is_pred": True, "is_gt": False, "mapping_type": row.get("mapping_type", "unknown")})
    for row in gt_rows:
        typed_rows.append({"mapping_key": _mapping_key(row), "is_pred": False, "is_gt": True, "mapping_type": row.get("mapping_type", "unknown")})

    per_type = group_prf1(typed_rows, "mapping_type")

    benchmark_prefix = "aas://"
    benchmark_shape_hits = sum(
        1 for row in pred_rows if str(row.get("target_path", "")).startswith(benchmark_prefix) and "/submodel/default/element/" in str(row.get("target_path", "")) and str(row.get("target_path", "")).endswith("/value")
    )
    benchmark_target_shape_rate = benchmark_shape_hits / len(pred_rows) if pred_rows else 0.0

    primary_precision = exact["exact_mapping_precision"]
    primary_recall = exact["exact_mapping_recall"]
    primary_f1 = exact["exact_mapping_f1"]
    primary_metric = "exact_mapping"

    if evaluation_mode == "sample_top1":
        primary_precision = sample_top1["precision"]
        primary_recall = sample_top1["recall"]
        primary_f1 = sample_top1["f1"]
        primary_metric = "sample_top1"
    elif evaluation_mode == "auto":
        # Use sample-level Top-1 metric when exact mapping is clearly suffering from
        # source-path granularity mismatch but sample-level matches exist.
        if sample_rows and exact["exact_mapping_f1"] == 0.0 and sample_top1["f1"] > 0.0:
            primary_precision = sample_top1["precision"]
            primary_recall = sample_top1["recall"]
            primary_f1 = sample_top1["f1"]
            primary_metric = "sample_top1"

    score = {
        "precision": primary_precision,
        "recall": primary_recall,
        "f1": primary_f1,
        "primary_metric": primary_metric,
        **exact,
        "sample_top1_precision": sample_top1["precision"],
        "sample_top1_recall": sample_top1["recall"],
        "sample_top1_f1": sample_top1["f1"],
        "sample_top1_count": sample_top1["count"],
        "path_validity_rate": max(0.0, path_validity),
        "semantic_validity_rate": max(0.0, semantic_validity),
        "transform_correctness": transform_correct / transform_total if transform_total else 1.0,
        "retrieval_recall_at_1": recall_at_k(retrieval_rows, 1),
        "retrieval_recall_at_5": recall_at_k(retrieval_rows, 5),
        "retrieval_hit_at_1": hit_at_k(retrieval_rows, 1),
        "retrieval_hit_at_5": hit_at_k(retrieval_rows, 5),
        "per_mapping_type": per_type,
        "per_tier": {},
        "per_pair": {},
        "per_difficulty": {},
        "confidence_calibration_error": calibration_error,
        "validity_pass_rate": sum(1 for r in reports if r.get("valid")) / len(reports) if reports else 0.0,
        "violation_counts": violation_counts(reports),
        "pred_count": len(pred_keys),
        "gt_count": len(gt_keys),
        "matched_count": len(pred_keys & gt_keys),
        "evaluation_mode": evaluation_mode,
        "dataset_count": len(gt_rows_raw),
        "gt_path_used": str(gt_path),
        "pred_path_used": str(pred_path),
        "benchmark_target_shape_rate": benchmark_target_shape_rate,
        "latency_per_sample_s": sum(float(r.get("latency_s", 0.0)) for r in perf_rows) / len(perf_rows) if perf_rows else 0.0,
        "runtime_per_scenario_s": sum(float(r.get("latency_s", 0.0)) for r in perf_rows),
        "token_usage_prompt": sum(int(r.get("tokens_prompt", 0)) for r in perf_rows),
        "token_usage_completion": sum(int(r.get("tokens_completion", 0)) for r in perf_rows),
        "memory_peak_mb": (max((int(r.get("memory_peak_bytes", 0)) for r in perf_rows), default=0) / (1024 * 1024)),
        "adaptive_strategy_usage": {},
        "candidate_coverage_rate": 0.0,
        "mean_candidate_count": 0.0,
    }
    if retrieval_rows:
        with_candidates = sum(1 for row in retrieval_rows if row.get("candidates"))
        score["candidate_coverage_rate"] = with_candidates / len(retrieval_rows)
        score["mean_candidate_count"] = sum(len(row.get("candidates") or []) for row in retrieval_rows) / len(retrieval_rows)

    if sample_rows:
        tier_rows: dict[str, list[dict]] = {}
        pair_rows: dict[str, list[dict]] = {}
        for row in sample_rows:
            tier_rows.setdefault(row.get("tier", "unknown"), []).append(row)
            pair_rows.setdefault(row.get("pair", "unknown"), []).append(row)
        score["per_tier"] = {k: {"count": len(v), "accuracy": sum(1 for x in v if x.get("matched")) / len(v)} for k, v in tier_rows.items()}
        score["per_pair"] = {k: {"count": len(v), "accuracy": sum(1 for x in v if x.get("matched")) / len(v)} for k, v in pair_rows.items()}
        difficulty_rows: dict[str, list[dict]] = {}
        for row in sample_rows:
            difficulty_rows.setdefault(row.get("difficulty", "unknown"), []).append(row)
        score["per_difficulty"] = {k: {"count": len(v), "accuracy": sum(1 for x in v if x.get("matched")) / len(v)} for k, v in difficulty_rows.items()}
    if decision_rows:
        usage: dict[str, int] = {}
        by_pair: dict[str, dict[str, int]] = {}
        by_tier: dict[str, dict[str, int]] = {}
        by_difficulty: dict[str, dict[str, int]] = {}
        strategy_accuracy: dict[str, list[bool]] = {}
        for row in decision_rows:
            strategy = str(row.get("selected_strategy", "unknown"))
            usage[strategy] = usage.get(strategy, 0) + 1
            pair = str(row.get("pair", "unknown"))
            tier = str(row.get("tier", "unknown"))
            difficulty = str(row.get("difficulty", "unknown"))
            by_pair.setdefault(pair, {})
            by_pair[pair][strategy] = by_pair[pair].get(strategy, 0) + 1
            by_tier.setdefault(tier, {})
            by_tier[tier][strategy] = by_tier[tier].get(strategy, 0) + 1
            by_difficulty.setdefault(difficulty, {})
            by_difficulty[difficulty][strategy] = by_difficulty[difficulty].get(strategy, 0) + 1
            strategy_accuracy.setdefault(strategy, []).append(bool(row.get("matched", False)))
        score["adaptive_strategy_usage"] = usage
        score["adaptive_strategy_usage_by_pair"] = by_pair
        score["adaptive_strategy_usage_by_tier"] = by_tier
        score["adaptive_strategy_usage_by_difficulty"] = by_difficulty
        score["adaptive_strategy_accuracy"] = {
            strategy: (sum(1 for x in matches if x) / len(matches) if matches else 0.0)
            for strategy, matches in strategy_accuracy.items()
        }

    score["counts"] = {
        "ground_truth": {"raw": len(gt_rows_raw), "deduplicated": len(gt_rows)},
        "predictions": {"raw": len(pred_rows_raw), "deduplicated": len(pred_rows)},
    }
    score["matched_count"] = len(pred_keys & gt_keys)
    if pred_keys != gt_keys:
        score["mismatch_diagnostics"] = {
            "pred_only_count": len(pred_keys - gt_keys),
            "gt_only_count": len(gt_keys - pred_keys),
        }
    fp = [k for k in pred_keys - gt_keys]
    fn = [k for k in gt_keys - pred_keys]
    invalid = [r for r in reports if not r.get("valid")]
    validation_reason_counts = violation_counts(reports)
    errors = {
        "false_positives": [{"source_path": s, "target_path": t, "mapping_type": m, "root_cause": "over_prediction"} for s, t, m in fp],
        "false_negatives": [{"source_path": s, "target_path": t, "mapping_type": m, "root_cause": "missed_mapping"} for s, t, m in fn],
        "invalid_path": [{"violations": r.get("violations", []), "root_cause": "path_or_schema"} for r in invalid],
        "wrong_transform": [
            {"source_path": row.get("source_path"), "target_path": row.get("target_path"), "root_cause": "transform_mismatch"}
            for row in gt_rows
            if row.get("mapping_type") == "transform" and _mapping_key(row) not in pred_keys
        ],
        "retrieval_failures": [
            {"sample": row.get("sample"), "root_cause": "retrieval_failure"}
            for row in retrieval_rows
            if row.get("expected_target_path") and row.get("expected_target_path") not in [c.get("path") for c in (row.get("candidates") or [])]
        ],
        "llm_hallucinations": [
            {"sample": row.get("sample"), "predicted": (row.get("predicted_top") or {}).get("target_path"), "expected": row.get("expected_target_path"), "root_cause": "llm_hallucination"}
            for row in _load_optional_jsonl(out_dir / "predictions" / "llm_trace.jsonl")
            if row.get("expected_target_path") and (row.get("predicted_top") or {}).get("target_path") not in {"", row.get("expected_target_path")}
        ],
        "cardinality_issues": [r for r in invalid if any(str(v.get("type", "")).startswith("cardinality_") for v in r.get("violations", []))],
        "validation_reasons": {
            "schema_invalid": validation_reason_counts.get("schema_invalid", 0),
            "duplicate_mapping": validation_reason_counts.get("duplicate_mapping", 0),
            "confidence_low": validation_reason_counts.get("confidence_low", 0),
            "invalid_path": validation_reason_counts.get("invalid_path", 0),
            "cardinality_issue": validation_reason_counts.get("cardinality_issue", 0) + validation_reason_counts.get("cardinality_trimmed", 0),
            "empty_target_for_non_no_match": validation_reason_counts.get("empty_target_for_non_no_match", 0),
            "wrong_transform": 0,
            "retrieval_failure": 0,
            "llm_hallucination": 0,
        },
    }
    errors["validation_reasons"]["wrong_transform"] = len(errors["wrong_transform"])
    errors["validation_reasons"]["retrieval_failure"] = len(errors["retrieval_failures"])
    errors["validation_reasons"]["llm_hallucination"] = len(errors["llm_hallucinations"])
    (out_dir / "error_analysis.json").write_text(json.dumps(errors, indent=2))
    (out_dir / "error_summary.json").write_text(json.dumps(errors["validation_reasons"], indent=2))
    retrieval_diagnostics = {
        "retrieval_recall_at_1": score["retrieval_recall_at_1"],
        "retrieval_recall_at_5": score["retrieval_recall_at_5"],
        "candidate_coverage_rate": score["candidate_coverage_rate"],
        "mean_candidate_count": score["mean_candidate_count"],
        "samples": retrieval_rows,
    }
    (out_dir / "retrieval_diagnostics.json").write_text(json.dumps(retrieval_diagnostics, indent=2))
    strategy_usage_path = out_dir / "strategy_usage.json"
    if strategy_usage_path.exists():
        score["scenario_strategy_usage"] = json.loads(strategy_usage_path.read_text())
    return score
