from __future__ import annotations

import json
from pathlib import Path

from benchmark.metrics import prf1, violation_counts
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


def _deduplicate_rows(rows: list[dict]) -> list[dict]:
    best_by_key: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        key = _mapping_key(row)
        current = best_by_key.get(key)
        if current is None or float(row.get("confidence", 0.0)) > float(current.get("confidence", 0.0)):
            best_by_key[key] = row
    return [best_by_key[key] for key in sorted(best_by_key)]


def _resolve_gt_path(out_dir: Path) -> Path:
    candidates = [
        out_dir.parent / "ground_truth.jsonl",
        out_dir.parent / "gt_mappings.jsonl",  # backward compatibility
        out_dir.parent.parent / "ground_truth.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Ground truth not found. Expected artifacts/<run_id>/ground_truth.jsonl "
        "or artifacts/<run_id>/predictions/gt_mappings.jsonl"
    )


def evaluate_run(out_dir: Path, evaluation_mode: str = "exact_match") -> dict:
    pred_path = out_dir / "mappings.jsonl"
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file missing: {pred_path}")
    pred_rows_raw = _load_jsonl_rows(pred_path)
    if not pred_rows_raw:
        raise ValueError(f"Predictions file is empty: {pred_path}")

    gt_path = _resolve_gt_path(out_dir)
    gt_rows_raw = _load_jsonl_rows(gt_path)
    if not gt_rows_raw:
        raise ValueError(f"Ground truth file is empty: {gt_path}")

    pred_rows = _deduplicate_rows(pred_rows_raw)
    gt_rows = _deduplicate_rows(gt_rows_raw)

    pred_keys = {_mapping_key(row) for row in pred_rows}
    gt_keys = {_mapping_key(row) for row in gt_rows}

    reports = []
    validation_path = out_dir / "validation.json"
    if validation_path.exists():
        reports = json.loads(validation_path.read_text())

    matched_keys = pred_keys & gt_keys
    scores = prf1(pred_keys, gt_keys)

    mismatch_diagnostics = None
    if pred_keys != gt_keys:
        pred_only = sorted(pred_keys - gt_keys)
        gt_only = sorted(gt_keys - pred_keys)
        mismatch_diagnostics = {
            "message": "Prediction and GT deduplicated counts differ. Metrics computed on key-set intersection/union.",
            "gt_count": len(gt_keys),
            "pred_count": len(pred_keys),
            "difference": abs(len(gt_keys) - len(pred_keys)),
            "pred_only_count": len(pred_only),
            "gt_only_count": len(gt_only),
            "pred_only_examples": pred_only[:10],
            "gt_only_examples": gt_only[:10],
            "intersection_count": len(matched_keys),
            "intersection_precision": (len(matched_keys) / len(pred_keys)) if pred_keys else 0.0,
            "intersection_recall": (len(matched_keys) / len(gt_keys)) if gt_keys else 0.0,
        }

    scores["validity_pass_rate"] = sum(1 for r in reports if r.get("valid")) / len(reports) if reports else 0.0
    scores["violation_counts"] = violation_counts(reports)
    scores["counts"] = {
        "ground_truth": {"raw": len(gt_rows_raw), "deduplicated": len(gt_keys)},
        "predictions": {"raw": len(pred_rows_raw), "deduplicated": len(pred_keys)},
        "matched": len(matched_keys),
    }
    scores["gt_count"] = len(gt_keys)
    scores["pred_count"] = len(pred_keys)
    scores["matched_count"] = len(matched_keys)
    scores["coverage"] = len(matched_keys) / len(gt_keys) if gt_keys else 0.0
    scores["dataset_count"] = len(gt_rows_raw)
    scores["evaluation_mode"] = evaluation_mode
    scores["gt_path_used"] = str(gt_path)
    scores["pred_path_used"] = str(pred_path)
    if mismatch_diagnostics is not None:
        scores["mismatch_diagnostics"] = mismatch_diagnostics

    print(
        f"Evaluation summary | GT count={scores['gt_count']} Pred count={scores['pred_count']} "
        f"Matched={scores['matched_count']} Coverage={scores['coverage']:.3f}\n"
        f"GT path={scores['gt_path_used']}\nPred path={scores['pred_path_used']}"
    )
    return scores
