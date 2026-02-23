from __future__ import annotations

import json
import subprocess
from pathlib import Path


def main() -> int:
    out_dir = Path("results/sample_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-u",
        "-m",
        "benchmark.run",
        "--scenario",
        "baseline",
        "--config",
        "benchmark/config.json",
        "--out",
        str(out_dir),
        "--max-items",
        "5",
        "--model-name",
        "qwen3:0.6b",
        "--tier",
        "canonical",
    ]
    print("[SAMPLE] Starting sample validation with N=5", flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        return proc.returncode

    metrics = json.loads((out_dir / "metrics.json").read_text())
    print(f"[SAMPLE] GT path used: {metrics.get('gt_path_used')}", flush=True)
    print(f"[SAMPLE] Pred path used: {metrics.get('pred_path_used')}", flush=True)

    checks = {
        "gt_count == 5": metrics.get("gt_count") == 5,
        "pred_count == 5": metrics.get("pred_count") == 5,
        "dataset_count == 5": metrics.get("dataset_count") == 5,
        "f1 present": isinstance(metrics.get("f1"), (int, float)),
        "validity_pass_rate not always 1": metrics.get("validity_pass_rate", 0.0) < 1.0 or bool(metrics.get("violation_counts")),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        print("[SAMPLE] Validation failed:", ", ".join(failed), flush=True)
        return 1
    print("[SAMPLE] Validation passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
