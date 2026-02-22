from __future__ import annotations

from collections import Counter


def prf1(pred: set[tuple[str, str]], gold: set[tuple[str, str]]) -> dict[str, float]:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def violation_counts(reports: list[dict]) -> dict[str, int]:
    c = Counter()
    for r in reports:
        for v in r.get("violations", []):
            c[v.get("type", "unknown")] += 1
    return dict(c)
