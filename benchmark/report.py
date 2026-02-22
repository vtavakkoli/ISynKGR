from __future__ import annotations

import json
from pathlib import Path


def write_report(run_dir: Path, summary: dict) -> None:
    md = ["# ISynKGR Benchmark Report", "", "|Metric|Value|", "|---|---|"]
    for k, v in summary.items():
        md.append(f"|{k}|{v}|")
    (run_dir / "report.md").write_text("\n".join(md))
    html = "<html><body><h1>ISynKGR Benchmark Report</h1><pre>" + json.dumps(summary, indent=2) + "</pre></body></html>"
    (run_dir / "report.html").write_text(html)
