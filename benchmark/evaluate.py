from __future__ import annotations

import json
from pathlib import Path

from benchmark.metrics import prf1, violation_counts
from isynkgr.icr.mapping_schema import ingest_mapping_payload, normalize_mapping_path


def _load_jsonl_pairs(path: Path) -> tuple[list[dict], set[tuple[str, str]]]:
    rows: list[dict] = []
    pairs: set[tuple[str, str]] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        record = ingest_mapping_payload(row, migrate_legacy=True)
        canonical = record.model_dump()
        rows.append(canonical)
        pairs.add((normalize_mapping_path(record.source_path), normalize_mapping_path(record.target_path)))
    return rows, pairs


def _has_blocking_violations(report: dict) -> bool:
    return any(v.get("type") != "confidence_low" for v in report.get("violations", []))


def _confidence_low_stats(reports: list[dict]) -> tuple[int, float]:
    count = 0
    for report in reports:
        count += sum(
            1 for v in report.get("violations", []) if v.get("type") == "confidence_low"
        )
    rate = count / len(reports) if reports else 0.0
    return count, rate


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
    pred_rows, pred_pairs = _load_jsonl_pairs(pred_path)
    if not pred_rows:
        raise ValueError(f"Predictions file is empty: {pred_path}")

    gt_path = _resolve_gt_path(out_dir)
    gt_rows, gt_pairs = _load_jsonl_pairs(gt_path)
    if not gt_rows:
        raise ValueError(f"Ground truth file is empty: {gt_path}")

    reports = []
    validation_path = out_dir / "validation.json"
    if validation_path.exists():
        reports = json.loads(validation_path.read_text())

    matched_pairs = pred_pairs & gt_pairs
    scores = prf1(pred_pairs, gt_pairs)
    if len(pred_pairs) != len(gt_pairs):
        scores["mismatch"] = {
            "message": "Prediction and GT counts differ. Metrics computed on available sets; coverage reported.",
            "gt_count": len(gt_pairs),
            "pred_count": len(pred_pairs),
        }

    confidence_low_count, confidence_low_rate = _confidence_low_stats(reports)
    scores["validity_pass_rate"] = (
        sum(1 for r in reports if not _has_blocking_violations(r)) / len(reports)
        if reports
        else 0.0
    )
    scores["confidence_low_count"] = confidence_low_count
    scores["confidence_low_rate"] = confidence_low_rate
    scores["violation_counts"] = violation_counts(reports)
    scores["gt_count"] = len(gt_pairs)
    scores["pred_count"] = len(pred_pairs)
    scores["matched_count"] = len(matched_pairs)
    scores["coverage"] = len(matched_pairs) / len(gt_pairs) if gt_pairs else 0.0
    scores["dataset_count"] = len(gt_rows)
    scores["evaluation_mode"] = evaluation_mode
    scores["gt_path_used"] = str(gt_path)
    scores["pred_path_used"] = str(pred_path)

    print(
        f"Evaluation summary | GT count={scores['gt_count']} Pred count={scores['pred_count']} "
        f"Matched={scores['matched_count']} Coverage={scores['coverage']:.3f}\n"
        f"GT path={scores['gt_path_used']}\nPred path={scores['pred_path_used']}"
    )
    return scores
