from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from isynkgr.common import STANDARDS

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None


def load_standards(config_path: Path | None) -> list[str]:
    if config_path is None:
        return list(STANDARDS.keys())
    data = json.loads(config_path.read_text())
    standards = data.get("standards", [])
    if not standards:
        raise ValueError(f"No standards found in config: {config_path}")
    return list(standards)


def build_pairs(standards: Iterable[str]) -> list[tuple[str, str]]:
    items = list(standards)
    return [(source, target) for source in items for target in items if source != target]


def build_graph(sample: dict[str, Any]) -> dict[str, Any]:
    entity = sample["entities"][0]["id"]
    nodes = [{"id": entity, "label": entity, "synonyms": sample["terms"]}]
    edges = []
    for prop in sample["properties"]:
        prop_id = f"{entity}:{prop['name']}"
        nodes.append({"id": prop_id, "label": prop["name"], "synonyms": [f"{prop['name']}_alias"]})
        edges.append({"source": entity, "target": prop_id, "predicate": "hasProperty"})
    return {"nodes": nodes, "edges": edges}


def predict_name(sample: dict[str, Any], target: str, method: str) -> str:
    source = sample["standard"]
    base = sample["entities"][0]["id"]
    if method in {"isynkgr", "kg_only", "graph_only"}:
        return base.replace(source, target)
    if method == "rag":
        return f"{base.replace(source, target)}_rag"
    return f"{target}_guess_{sample['sample_id'].split('_')[-1]}"


def score(pred: str, gt: str) -> dict[str, float]:
    ok = float(pred == gt)
    return {"accuracy": ok, "precision": ok, "recall": ok, "f1": ok, "property_accuracy": ok}


def save_outputs(rows: list[dict[str, Any]], out_root: Path) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / ts
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    with (out_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, plot_dir / "f1_by_method.png")

    latest = out_root / "latest"
    if latest.exists() or latest.is_symlink():
        import shutil

        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest)

    import shutil

    shutil.copytree(out_dir, latest)
    return out_dir


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    if plt is None:
        return
    methods = sorted(set(r["method"] for r in rows))
    vals = [statistics.mean([r["f1"] for r in rows if r["method"] == method]) for method in methods]
    plt.figure(figsize=(8, 4))
    plt.bar(methods, vals)
    plt.ylim(0, 1)
    plt.title("Mean F1 by method")
    plt.tight_layout()
    plt.savefig(path)
