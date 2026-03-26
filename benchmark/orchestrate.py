from __future__ import annotations

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
        print("STEP 1/2: adaptive full workflow", flush=True)
        _run([sys.executable, "-u", "-m", "benchmark.full_workflow"], " - adaptive-full-workflow")
        print("STEP 2/2: final report export", flush=True)
        final_dir = generate_final_report(Path("results"))
        report_html = Path("results/final_report.html")
        latest_artifacts = sorted(Path("artifacts").glob("run_*"))
        if latest_artifacts:
            src = latest_artifacts[-1] / "report.html"
            if src.exists():
                report_html.write_text(src.read_text())
        print(f"Final report generated at {final_dir} and {report_html}", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Pipeline failed: {exc}", flush=True)
        print("Check logs under results/<scenario>/logs/run.log", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
