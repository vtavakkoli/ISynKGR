from __future__ import annotations

import html
import json
from pathlib import Path

CANONICAL_METRIC_KEYS = ("precision", "recall", "f1", "validity_pass_rate", "violation_counts")


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _markdown_table(rows: list[dict], columns: list[str]) -> str:
    header = "|" + "|".join(columns) + "|"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    body = []
    for row in rows:
        body.append("|" + "|".join(str(row.get(col, "")) for col in columns) + "|")
    return "\n".join([header, sep, *body])


def _import_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to generate PNG charts. "
            "Install matplotlib or run this in the benchmark Docker image."
        ) from exc
    return plt


def _write_placeholder_png(path: Path) -> None:
    # 1x1 transparent PNG
    path.write_bytes(bytes.fromhex("89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C4890000000A49444154789C6360000000020001E221BC330000000049454E44AE426082"))


def _bar_chart(path: Path, names: list[str], values: list[float], title: str, ylabel: str) -> None:
    try:
        plt = _import_matplotlib_pyplot()
    except RuntimeError:
        _write_placeholder_png(path)
        return
    plt.figure(figsize=(9, 4))
    plt.bar(names, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _scenario_name(row: dict) -> str:
    return str(row.get("baseline") or row.get("scenario") or "")


def _metric(row: dict, key: str) -> float:
    return float(row.get(key, 0.0))


def _aggregate_violations(rows: list[dict]) -> dict[str, int]:
    violations: dict[str, int] = {}
    for row in rows:
        violation_counts = row.get("violation_counts") or {}
        for key, value in violation_counts.items():
            violations[key] = violations.get(key, 0) + int(value)
    return violations


def _build_validity_breakdown(violations: dict[str, int]) -> list[dict]:
    return [
        {"reason": "mapping_type_invalid", "count": int(violations.get("mapping_type_invalid", 0))},
        {"reason": "target_id_format", "count": int(violations.get("target_id_format", 0))},
        {
            "reason": "target_validator_errors",
            "count": int(sum(v for k, v in violations.items() if str(k).startswith("target_"))),
        },
        {"reason": "confidence_low", "count": int(violations.get("confidence_low", 0))},
    ]


def write_report(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    canonical_rows = []
    for row in rows:
        canonical_rows.append(
            {
                "scenario": _scenario_name(row),
                "precision": _metric(row, "precision"),
                "recall": _metric(row, "recall"),
                "f1": _metric(row, "f1"),
                "validity_pass_rate": _metric(row, "validity_pass_rate"),
                "violation_counts": row.get("violation_counts") or {},
            }
        )

    ranked_f1 = sorted(canonical_rows, key=lambda r: r["f1"], reverse=True)
    ranked_validity = sorted(canonical_rows, key=lambda r: r["validity_pass_rate"], reverse=True)
    violations = _aggregate_violations(canonical_rows)
    violation_rows = [
        {"violation_type": k, "count": v}
        for k, v in sorted(violations.items(), key=lambda kv: kv[1], reverse=True)
    ]
    validity_breakdown = _build_validity_breakdown(violations)

    summary_rows = [
        {
            "scenario": r["scenario"],
            "f1": _fmt(r["f1"]),
            "validity_pass_rate": _fmt(r["validity_pass_rate"]),
        }
        for r in ranked_f1
    ]

    report_payload = {
        "canonical_metric_keys": list(CANONICAL_METRIC_KEYS),
        "summary_table": summary_rows,
        "why_validity_low": validity_breakdown,
        "top_violations": violation_rows,
        "scenarios": canonical_rows,
    }
    (run_dir / "report.json").write_text(json.dumps(report_payload, indent=2))

    md = [
        "# ISynKGR Benchmark Report",
        "",
        "Canonical metric keys consumed from evaluator: `precision`, `recall`, `f1`, `validity_pass_rate`, `violation_counts`.",
        "",
        "## Summary table (F1 + validity)",
        _markdown_table(summary_rows, ["scenario", "f1", "validity_pass_rate"]),
        "",
        "## Why validity is low",
        _markdown_table(validity_breakdown, ["reason", "count"]),
        "",
        "## Top violations",
        _markdown_table(violation_rows or [{"violation_type": "none", "count": 0}], ["violation_type", "count"]),
        "",
        "## Plots",
        "- `plots/f1_by_scenario.png`",
        "- `plots/validity_by_scenario.png`",
        "- `plots/top_violations.png`",
        "",
        "## Raw JSON details",
        "```json",
        json.dumps(report_payload, indent=2),
        "```",
    ]
    (run_dir / "report.md").write_text("\n".join(md))

    names = [r["scenario"] for r in ranked_f1]
    _bar_chart(plots_dir / "f1_by_scenario.png", names, [r["f1"] for r in ranked_f1], "F1 by Scenario", "f1")
    _bar_chart(
        plots_dir / "validity_by_scenario.png",
        [r["scenario"] for r in ranked_validity],
        [r["validity_pass_rate"] for r in ranked_validity],
        "Validity by Scenario",
        "validity_pass_rate",
    )
    top_violation_rows = violation_rows[:10] if violation_rows else [{"violation_type": "none", "count": 0}]
    _bar_chart(
        plots_dir / "top_violations.png",
        [r["violation_type"] for r in top_violation_rows],
        [float(r["count"]) for r in top_violation_rows],
        "Top Violations",
        "count",
    )

    summary_table = html.escape(_markdown_table(summary_rows, ["scenario", "f1", "validity_pass_rate"]))
    validity_table = html.escape(_markdown_table(validity_breakdown, ["reason", "count"]))
    raw_json = html.escape(json.dumps(report_payload, indent=2))
    html_content = f"""<html><body style="font-family:Arial,sans-serif;margin:24px">
<h1>ISynKGR Benchmark Report</h1>
<p>Canonical metric keys consumed from evaluator: <code>precision</code>, <code>recall</code>, <code>f1</code>, <code>validity_pass_rate</code>, <code>violation_counts</code>.</p>
<h2>Summary table (F1 + validity)</h2>
<pre>{summary_table}</pre>
<h2>Why validity is low</h2>
<pre>{validity_table}</pre>
<h2>Plots</h2>
<ul>
<li><img alt="F1 by scenario" src="plots/f1_by_scenario.png" style="max-width:100%;height:auto" /></li>
<li><img alt="Validity by scenario" src="plots/validity_by_scenario.png" style="max-width:100%;height:auto" /></li>
<li><img alt="Top violations" src="plots/top_violations.png" style="max-width:100%;height:auto" /></li>
</ul>
<h2>Raw JSON details</h2>
<details>
<summary>Expand raw JSON details</summary>
<pre>{raw_json}</pre>
</details>
</body></html>"""
    (run_dir / "report.html").write_text(html_content)


def generate_final_report(results_root: Path = Path("results")) -> Path:
    scenarios = [
        "baseline",
        "full_framework",
        "ablation_no_graphrag",
        "ablation_no_parallel",
        "ablation_no_community",
        "ablation_no_reasoning",
    ]
    rows = []
    for scenario in scenarios:
        metrics_path = results_root / scenario / "metrics.json"
        if metrics_path.exists():
            payload = json.loads(metrics_path.read_text())
            if isinstance(payload, list):
                for row in payload:
                    if isinstance(row, dict):
                        rows.append(row)
            elif isinstance(payload, dict):
                rows.append(payload)

    final_dir = results_root / "final"
    write_report(final_dir, rows)
    return final_dir
