from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from benchmark.report import generate_final_report

SCENARIOS = [
    "baseline",
    "full_framework",
    "ablation_no_graphrag",
    "ablation_no_parallel",
    "ablation_no_community",
    "ablation_no_reasoning",
]


def _run(cmd: list[str], step: str) -> None:
    print(step, flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"{step} failed (exit={proc.returncode})")


def main() -> int:
    try:
        _run([sys.executable, "-u", "-m", "benchmark.sample_validate"], "STEP 1/4: sample validation")
        print("STEP 2/4: run scenarios", flush=True)
        for scenario in SCENARIOS:
            _run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "benchmark.run",
                    "--scenario",
                    scenario,
                    "--config",
                    "benchmark/config.json",
                    "--out",
                    f"results/{scenario}",
                    "--max-items",
                    "100",
                ],
                f" - scenario={scenario}",
            )
        print("STEP 3/4: evaluate", flush=True)
        for scenario in SCENARIOS:
            metrics_path = Path(f"results/{scenario}/metrics.json")
            if not metrics_path.exists():
                raise RuntimeError(f"Missing metrics for {scenario}: {metrics_path}")
            metrics = json.loads(metrics_path.read_text())
            print(
                f"scenario={scenario} GT={metrics.get('gt_count')} Pred={metrics.get('pred_count')} "
                f"Matched={metrics.get('matched_count')} Coverage={metrics.get('coverage')}",
                flush=True,
            )
        print("STEP 4/4: final report", flush=True)
        final_dir = generate_final_report(Path("results"))
        print(f"Final report generated at {final_dir}", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Pipeline failed: {exc}", flush=True)
        print("Check logs under results/<scenario>/logs/run.log", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
