from __future__ import annotations

from pathlib import Path

from benchmark.full_workflow import run_full_workflow
from benchmark.report import generate_final_report


def main() -> int:
    try:
        print("STEP 1/2: canonical full workflow", flush=True)
        rc = run_full_workflow()
        if rc != 0:
            return rc
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
        print("Check logs under results/<pair>/<scenario>/seed*/", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
