from __future__ import annotations

import html
import json
import csv
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
    pair = str(row.get("pair") or "").strip()
    scenario = str(row.get("baseline") or row.get("scenario") or "")
    return f"{pair}::{scenario}" if pair else scenario


def _aggregate_rows(rows: list[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(_scenario_name(row), []).append(row)
    aggregated: list[dict] = []
    for scenario, items in buckets.items():
        f1_values = [float(i.get("f1", 0.0)) for i in items]
        validity_values = [float(i.get("validity_pass_rate", 0.0)) for i in items]
        aggregated.append(
            {
                "scenario": scenario,
                "pair": scenario.split("::", 1)[0] if "::" in scenario else "aggregate",
                "baseline": scenario.split("::", 1)[1] if "::" in scenario else scenario,
                "runs": len(items),
                "f1_mean": sum(f1_values) / max(len(f1_values), 1),
                "f1_std": (sum((x - (sum(f1_values) / max(len(f1_values), 1))) ** 2 for x in f1_values) / max(len(f1_values), 1)) ** 0.5 if f1_values else 0.0,
                "validity_mean": sum(validity_values) / max(len(validity_values), 1),
                "validity_std": (sum((x - (sum(validity_values) / max(len(validity_values), 1))) ** 2 for x in validity_values) / max(len(validity_values), 1)) ** 0.5 if validity_values else 0.0,
            }
        )
    return sorted(aggregated, key=lambda r: r["f1_mean"], reverse=True)


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
    aggregated_rows = _aggregate_rows(rows)
    violations = _aggregate_violations(canonical_rows)
    violation_rows = [
        {"violation_type": k, "count": v}
        for k, v in sorted(violations.items(), key=lambda kv: kv[1], reverse=True)
    ]
    validity_breakdown = _build_validity_breakdown(violations)

    summary_rows = [
        {
            "scenario": r["scenario"],
            "pair": r["scenario"].split("::", 1)[0] if "::" in r["scenario"] else "aggregate",
            "f1": _fmt(r["f1"]),
            "validity_pass_rate": _fmt(r["validity_pass_rate"]),
        }
        for r in ranked_f1
    ]

    report_payload = {
        "canonical_metric_keys": list(CANONICAL_METRIC_KEYS),
        "summary_table": summary_rows,
        "aggregated_scenario_summary": aggregated_rows,
        "why_validity_low": validity_breakdown,
        "top_violations": violation_rows,
        "scenarios": canonical_rows,
    }
    (run_dir / "report.json").write_text(json.dumps(report_payload, indent=2))
    metrics_dir = run_dir / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "main_comparison.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "scenario",
                "precision",
                "recall",
                "f1",
                "validity_pass_rate",
                "transform_correctness",
                "retrieval_recall_at_1",
                "retrieval_recall_at_5",
                "latency_per_sample_s",
                "runtime_per_scenario_s",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "scenario": _scenario_name(row),
                    "precision": row.get("precision", 0.0),
                    "recall": row.get("recall", 0.0),
                    "f1": row.get("f1", 0.0),
                    "validity_pass_rate": row.get("validity_pass_rate", 0.0),
                    "transform_correctness": row.get("transform_correctness", 0.0),
                    "retrieval_recall_at_1": row.get("retrieval_recall_at_1", 0.0),
                    "retrieval_recall_at_5": row.get("retrieval_recall_at_5", 0.0),
                    "latency_per_sample_s": row.get("latency_per_sample_s", 0.0),
                    "runtime_per_scenario_s": row.get("runtime_per_scenario_s", 0.0),
                }
            )

    md = [
        "# ISynKGR Benchmark Report",
        "",
        "## Benchmark setup",
        "This report is generated from per-scenario exported metrics across configured seeds.",
        "",
        "## Scenario definitions",
        "See `docs/SCENARIO_MATRIX.md` for component-level scenario toggles.",
        "",
        "Canonical metric keys consumed from evaluator: `precision`, `recall`, `f1`, `validity_pass_rate`, `violation_counts`.",
        "",
        "## Main results",
        _markdown_table(summary_rows, ["pair", "scenario", "f1", "validity_pass_rate"]),
        "",
        "## Aggregated per-scenario summary (mean/std across seeds)",
        _markdown_table(
            [
                {
                    "pair": r["pair"],
                    "scenario": r["baseline"],
                    "runs": r["runs"],
                    "f1_mean": _fmt(r["f1_mean"]),
                    "f1_std": _fmt(r["f1_std"]),
                    "validity_mean": _fmt(r["validity_mean"]),
                    "validity_std": _fmt(r["validity_std"]),
                }
                for r in aggregated_rows
            ],
            ["pair", "scenario", "runs", "f1_mean", "f1_std", "validity_mean", "validity_std"],
        ),
        "",
        "## Ablation study",
        "Ablation scenarios are those with names prefixed by `ablation_`.",
        "",
        "## Why validity is low",
        _markdown_table(validity_breakdown, ["reason", "count"]),
        "",
        "## Error analysis",
        "## Top violations",
        _markdown_table(violation_rows or [{"violation_type": "none", "count": 0}], ["violation_type", "count"]),
        "",
        "## Reproducibility notes",
        "- Seeds are fixed per run; metrics are exported from artifacts without manual post-editing.",
        "- See `docs/IMPLEMENTATION_DIAGNOSIS.md` for known limitations.",
        "",
        "## Plots",
        "- `plots/f1_by_scenario.png`",
        "- `plots/validity_by_scenario.png`",
        "- `plots/top_violations.png`",
        "- `plots/latency_by_scenario.png`",
        "- `plots/retrieval_recall_by_scenario.png`",
        "",
        "## Raw JSON details",
        "```json",
        json.dumps(report_payload, indent=2),
        "```",
    ]
    (run_dir / "report.md").write_text("\n".join(md))

    names = [r["scenario"] for r in aggregated_rows]
    _bar_chart(plots_dir / "f1_by_scenario.png", names, [r["f1_mean"] for r in aggregated_rows], "F1 by Scenario (mean)", "f1")
    _bar_chart(
        plots_dir / "validity_by_scenario.png",
        [r["scenario"] for r in aggregated_rows],
        [r["validity_mean"] for r in aggregated_rows],
        "Validity by Scenario (mean)",
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
    _bar_chart(
        plots_dir / "cost_vs_performance.png",
        [r["scenario"] for r in aggregated_rows],
        [float(sum(x.get("runtime_per_scenario_s", 0.0) for x in rows if _scenario_name(x) == r["scenario"]) / max(1, sum(1 for x in rows if _scenario_name(x) == r["scenario"]))) for r in aggregated_rows],
        "Runtime Cost by Scenario",
        "runtime_s",
    )
    _bar_chart(
        plots_dir / "latency_by_scenario.png",
        [r["scenario"] for r in aggregated_rows],
        [float(sum(x.get("latency_per_sample_s", 0.0) for x in rows if _scenario_name(x) == r["scenario"]) / max(1, sum(1 for x in rows if _scenario_name(x) == r["scenario"]))) for r in aggregated_rows],
        "Latency per Sample by Scenario",
        "seconds",
    )
    _bar_chart(
        plots_dir / "retrieval_recall_by_scenario.png",
        [r["scenario"] for r in aggregated_rows],
        [float(sum(x.get("retrieval_recall_at_5", 0.0) for x in rows if _scenario_name(x) == r["scenario"]) / max(1, sum(1 for x in rows if _scenario_name(x) == r["scenario"]))) for r in aggregated_rows],
        "Retrieval Recall@5 by Scenario",
        "recall@5",
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
<li><img alt="Cost vs performance" src="plots/cost_vs_performance.png" style="max-width:100%;height:auto" /></li>
<li><img alt="Latency by scenario" src="plots/latency_by_scenario.png" style="max-width:100%;height:auto" /></li>
<li><img alt="Retrieval recall by scenario" src="plots/retrieval_recall_by_scenario.png" style="max-width:100%;height:auto" /></li>
</ul>
<h2>Raw JSON details</h2>
<details>
<summary>Expand raw JSON details</summary>
<pre>{raw_json}</pre>
</details>
</body></html>"""
    (run_dir / "report.html").write_text(html_content)


def generate_final_report(results_root: Path = Path("results")) -> Path:
    rows = []
    for metrics_path in results_root.glob("**/metrics.json"):
        payload = json.loads(metrics_path.read_text())
        if isinstance(payload, list):
            rows.extend([row for row in payload if isinstance(row, dict)])
        elif isinstance(payload, dict):
            rows.append(payload)

    final_dir = results_root / "final"
    write_report(final_dir, rows)
    return final_dir
