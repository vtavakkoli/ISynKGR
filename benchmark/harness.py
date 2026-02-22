from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from benchmark.evaluate import evaluate_run
from benchmark.report import write_report

BASELINES = ["isynkgr_hybrid", "llm_only", "rag_only", "rule_only", "graph_only"]


def run_benchmark(compose_file: str = "docker/compose/docker-compose.bench.yml", dataset_dir: str = "datasets/v1/crosswalk", full: bool = False) -> Path:
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    modes = BASELINES if full else ["rule_only", "graph_only", "isynkgr_hybrid"]
    rows = []
    for mode in modes:
        out_dir = run_dir / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["DATASET_DIR"] = str(Path(dataset_dir).resolve())
        env["OUTPUT_DIR"] = str(out_dir.resolve())
        env["CONFIG_PATH"] = str((Path("benchmark") / "config.json").resolve())
        subprocess.run(["docker", "compose", "-f", compose_file, "run", "--rm", mode], check=True, env=env)
        summary = evaluate_run(out_dir)
        summary["baseline"] = mode
        rows.append(summary)
    with (run_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys() if k != "violation_counts"}))
        w.writeheader()
        for r in rows:
            r = r.copy(); r.pop("violation_counts", None)
            w.writerow(r)
    (run_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    write_report(run_dir, {r["baseline"]: r.get("f1", 0.0) for r in rows})
    return run_dir


if __name__ == "__main__":
    run_benchmark(full=os.getenv("FULL", "0") == "1")
