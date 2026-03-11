from __future__ import annotations

import math
from collections import Counter, defaultdict


def prf1(pred: set[tuple[str, str]], gold: set[tuple[str, str]]) -> dict[str, float]:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def mapping_prf1(pred: set[tuple[str, str, str]], gold: set[tuple[str, str, str]]) -> dict[str, float]:
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"exact_mapping_precision": precision, "exact_mapping_recall": recall, "exact_mapping_f1": f1}


def violation_counts(reports: list[dict]) -> dict[str, int]:
    c = Counter()
    for r in reports:
        for v in r.get("violations", []):
            c[v.get("type", "unknown")] += 1
    return dict(c)


def recall_at_k(rows: list[dict], k: int) -> float:
    if not rows:
        return 0.0
    hits = 0
    for row in rows:
        candidates = row.get("candidates") or []
        expected = row.get("expected_target_path")
        topk = [c.get("path", "") for c in candidates[:k]]
        hits += int(bool(expected and expected in topk))
    return hits / len(rows)


def hit_at_k(rows: list[dict], k: int) -> float:
    return recall_at_k(rows, k)


def mean_std_ci(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": mean, "std": 0.0, "ci95": 0.0}
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    std = math.sqrt(var)
    ci95 = 1.96 * std / math.sqrt(len(values))
    return {"mean": mean, "std": std, "ci95": ci95}


def group_prf1(rows: list[dict], key: str) -> dict[str, dict[str, float]]:
    grouped_pred: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    grouped_gt: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for row in rows:
        grp = str(row.get(key, "unknown"))
        if row.get("is_pred"):
            grouped_pred[grp].add(tuple(row["mapping_key"]))
        if row.get("is_gt"):
            grouped_gt[grp].add(tuple(row["mapping_key"]))
    out: dict[str, dict[str, float]] = {}
    for grp in sorted(set(grouped_pred) | set(grouped_gt)):
        out[grp] = mapping_prf1(grouped_pred.get(grp, set()), grouped_gt.get(grp, set()))
    return out
