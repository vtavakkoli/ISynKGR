from __future__ import annotations

import json
from pathlib import Path

from benchmark.metrics import group_prf1, hit_at_k, mapping_prf1, recall_at_k, violation_counts
from isynkgr.icr.mapping_schema import ingest_mapping_payload, normalize_mapping_path


def _mapping_key(row: dict) -> tuple[str, str, str]:
    return (
        normalize_mapping_path(row.get("source_path", "")),
        normalize_mapping_path(row.get("target_path", "")),
        str(row.get("mapping_type", "")),
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
    for path in [out_dir.parent / "ground_truth.jsonl", out_dir.parent / "gt_mappings.jsonl", out_dir.parent.parent / "ground_truth.jsonl"]:
        if path.exists():
            return path
    raise FileNotFoundError("Ground truth not found.")


def evaluate_run(out_dir: Path, evaluation_mode: str = "exact_match") -> dict:
    pred_rows_raw = _load_jsonl_rows(out_dir / "mappings.jsonl")
    gt_rows_raw = _load_jsonl_rows(_resolve_gt_path(out_dir))
    pred_rows = _deduplicate_rows(pred_rows_raw)
    gt_rows = _deduplicate_rows(gt_rows_raw)

    pred_keys = {_mapping_key(row) for row in pred_rows}
    gt_keys = {_mapping_key(row) for row in gt_rows}
    exact = mapping_prf1(pred_keys, gt_keys)

    reports = json.loads((out_dir / "validation.json").read_text()) if (out_dir / "validation.json").exists() else []
    retrieval_rows = _load_optional_jsonl(out_dir / "predictions" / "retrieval_trace.jsonl")
    sample_rows = _load_optional_jsonl(out_dir / "predictions" / "sample_results.jsonl")

    pred_by_source = {k[0]: k for k in pred_keys}
    gt_by_source = {k[0]: k for k in gt_keys}
    transform_total = sum(1 for row in gt_rows if row.get("mapping_type") == "transform")
    transform_correct = sum(1 for row in gt_rows if row.get("mapping_type") == "transform" and _mapping_key(row) in pred_keys)
    path_validity = 1.0 - (sum((violation_counts(reports) or {}).values()) / max(len(reports), 1))

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

    score = {
        "precision": exact["exact_mapping_precision"],
        "recall": exact["exact_mapping_recall"],
        "f1": exact["exact_mapping_f1"],
        **exact,
        "path_validity_rate": max(0.0, path_validity),
        "transform_correctness": transform_correct / transform_total if transform_total else 1.0,
        "retrieval_recall_at_1": recall_at_k(retrieval_rows, 1),
        "retrieval_recall_at_5": recall_at_k(retrieval_rows, 5),
        "retrieval_hit_at_1": hit_at_k(retrieval_rows, 1),
        "retrieval_hit_at_5": hit_at_k(retrieval_rows, 5),
        "per_mapping_type": per_type,
        "per_tier": {},
        "per_pair": {},
        "confidence_calibration_error": calibration_error,
        "validity_pass_rate": sum(1 for r in reports if r.get("valid")) / len(reports) if reports else 0.0,
        "violation_counts": violation_counts(reports),
        "pred_count": len(pred_keys),
        "gt_count": len(gt_keys),
        "matched_count": len(pred_keys & gt_keys),
        "evaluation_mode": evaluation_mode,
    }

    if sample_rows:
        tier_rows: dict[str, list[dict]] = {}
        pair_rows: dict[str, list[dict]] = {}
        for row in sample_rows:
            tier_rows.setdefault(row.get("tier", "unknown"), []).append(row)
            pair_rows.setdefault(row.get("pair", "unknown"), []).append(row)
        score["per_tier"] = {k: {"count": len(v), "accuracy": sum(1 for x in v if x.get("matched")) / len(v)} for k, v in tier_rows.items()}
        score["per_pair"] = {k: {"count": len(v), "accuracy": sum(1 for x in v if x.get("matched")) / len(v)} for k, v in pair_rows.items()}

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
    errors = {
        "false_positives": [{"source_path": s, "target_path": t, "mapping_type": m, "root_cause": "over_prediction"} for s, t, m in fp],
        "false_negatives": [{"source_path": s, "target_path": t, "mapping_type": m, "root_cause": "missed_mapping"} for s, t, m in fn],
        "invalid_path": [{"violations": r.get("violations", []), "root_cause": "path_or_schema"} for r in invalid],
        "wrong_transform": [
            {"source_path": row.get("source_path"), "target_path": row.get("target_path"), "root_cause": "transform_mismatch"}
            for row in gt_rows
            if row.get("mapping_type") == "transform" and _mapping_key(row) not in pred_keys
        ],
    }
    (out_dir / "error_analysis.json").write_text(json.dumps(errors, indent=2))
    return score
