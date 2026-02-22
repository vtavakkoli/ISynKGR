from __future__ import annotations

import json
from pathlib import Path

from benchmark.metrics import prf1, violation_counts


def evaluate_run(out_dir: Path) -> dict:
    pred = set()
    gold = set()
    reports = []
    if (out_dir / "mappings.jsonl").exists():
        for line in (out_dir / "mappings.jsonl").read_text().splitlines():
            row = json.loads(line)
            pred.add((row["source_id"], row["target_id"]))
    gt = out_dir.parent / "gt_mappings.jsonl"
    if gt.exists():
        for line in gt.read_text().splitlines():
            row = json.loads(line)
            gold.add((row["source_id"], row["target_id"]))
    if (out_dir / "validation.json").exists():
        reports = json.loads((out_dir / "validation.json").read_text())
    scores = prf1(pred, gold)
    scores["validity_pass_rate"] = sum(1 for r in reports if r.get("valid")) / len(reports) if reports else 0.0
    scores["violation_counts"] = violation_counts(reports)
    return scores
