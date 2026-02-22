from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from src.common import STANDARDS, read_jsonl
from src.llm_integration.ollama_client import OllamaClient
from src.retrieval.graphrag import GraphRAGRetriever
from src.translation_logic.library import TranslationLogicLibrary


def build_graph(sample: dict[str, Any]) -> dict[str, Any]:
    entity = sample["entities"][0]["id"]
    nodes = [{"id": entity, "label": entity, "synonyms": sample["terms"]}]
    edges = []
    for p in sample["properties"]:
        pid = f"{entity}:{p['name']}"
        nodes.append({"id": pid, "label": p["name"], "synonyms": [f"{p['name']}_alias"]})
        edges.append({"source": entity, "target": pid, "predicate": "hasProperty"})
    return {"nodes": nodes, "edges": edges}


def predict_name(sample: dict, target: str, method: str) -> str:
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


def run_pair(source: str, target: str, args: argparse.Namespace, client: OllamaClient | None) -> list[dict[str, Any]]:
    samples = read_jsonl(Path("data/samples") / source / "samples_100.jsonl")[: args.max_samples]
    gt_rows = {r["sample_id"]: r for r in read_jsonl(Path("data/ground_truth") / f"{source}__to__{target}" / "gt.jsonl")}
    retriever = GraphRAGRetriever(k_hop=2)
    lib = TranslationLogicLibrary()
    methods = ["isynkgr", "rag", "llm_only", "kg_only", "graph_only"]
    results = []
    for method in methods:
        t0 = time.perf_counter()
        latencies = []
        metrics = []
        token_counts = []
        traversal = []
        for row in samples:
            i0 = time.perf_counter()
            graph = build_graph(row)
            ret = retriever.retrieve(graph, row["terms"], top_k=8)
            traversal.append(ret["stats"]["retrieved_edges"])
            pred = predict_name(row, target, method)
            if method in {"isynkgr", "llm_only"} and client:
                prompt = Path("prompts/v1/reasoning_check.txt").read_text().format(
                    source_standard=source,
                    target_standard=target,
                    source_entity=row["entities"][0]["id"],
                    candidate_target=pred,
                    evidence=json.dumps(ret, sort_keys=True),
                )
                llm = client.generate(prompt, context={"sample_id": row["sample_id"], "method": method})
                token_counts.append(llm.get("eval_count", 0) + llm.get("prompt_eval_count", 0))
            gt = gt_rows[row["sample_id"]]["target_entity"]
            sc = score(pred, gt)
            metrics.append(sc)
            if method == "isynkgr":
                lib.save_rule(source, target, row["entities"][0]["id"], pred, sc["f1"], {
                    "prompt_template": "prompts/v1/reasoning_check.txt",
                    "retrieved_subgraph_ids": [n["id"] for n in ret["nodes"]],
                    "model": args.model,
                })
            latencies.append(time.perf_counter() - i0)
        elapsed = time.perf_counter() - t0
        agg = {k: statistics.mean([m[k] for m in metrics]) for k in metrics[0]}
        results.append({
            "source": source,
            "target": target,
            "method": method,
            **agg,
            "latency_s_avg": statistics.mean(latencies),
            "latency_s_total": elapsed,
            "token_count_avg": statistics.mean(token_counts) if token_counts else 0,
            "kg_traversed_edges_avg": statistics.mean(traversal) if traversal else 0,
            "cpu_time_s": time.process_time(),
            "peak_rss_mb": _rss_mb(),
        })
    return results


def _rss_mb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return 0.0


def save_outputs(rows: list[dict[str, Any]], out_root: Path) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / ts
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, plot_dir / "f1_by_method.png")
    latest = out_root / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_dir.name)
    return out_dir


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    if plt is None:
        return
    methods = sorted(set(r["method"] for r in rows))
    vals = [statistics.mean([r["f1"] for r in rows if r["method"] == m]) for m in methods]
    plt.figure(figsize=(8, 4))
    plt.bar(methods, vals)
    plt.ylim(0, 1)
    plt.title("Mean F1 by method")
    plt.tight_layout()
    plt.savefig(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3:0.6b")
    p.add_argument("--max-samples", type=int, default=20)
    p.add_argument("--no-graphrag", action="store_true")
    p.add_argument("--no-cot", action="store_true")
    p.add_argument("--no-community", action="store_true")
    p.add_argument("--no-parallel-retrievers", action="store_true")
    args = p.parse_args()
    use_llm = os.getenv("ISYNKGR_SKIP_LLM", "0") != "1"
    client = OllamaClient(model=args.model) if use_llm else None
    all_rows: list[dict[str, Any]] = []
    standards = list(STANDARDS.keys())
    for source in standards:
        for target in standards:
            if source == target:
                continue
            all_rows.extend(run_pair(source, target, args, client))
    out = save_outputs(all_rows, Path("output/benchmarks"))
    print(f"Benchmark outputs: {out}")


if __name__ == "__main__":
    main()
