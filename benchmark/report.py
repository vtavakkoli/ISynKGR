from __future__ import annotations

import html
import json
import csv
from pathlib import Path
from statistics import fmean

from benchmark.scenarios import COMPONENT_FLAGS

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


def _html_table(rows: list[dict], columns: list[str]) -> str:
    head = "".join(f"<th style=\"padding:6px 10px;border:1px solid #ddd\">{html.escape(col)}</th>" for col in columns)
    body_rows: list[str] = []
    for row in rows:
        cols = "".join(
            f"<td style=\"padding:6px 10px;border:1px solid #ddd\">{html.escape(str(row.get(col, '')))}</td>"
            for col in columns
        )
        body_rows.append(f"<tr>{cols}</tr>")
    return (
        "<table style=\"border-collapse:collapse;border:1px solid #ddd\">"
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


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
    plt.figure(figsize=(11, 5))
    plt.bar(names, values, color="#2563eb", edgecolor="#1e3a8a", linewidth=0.6)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _grouped_metric_chart(path: Path, names: list[str], precision: list[float], recall: list[float], f1: list[float]) -> None:
    try:
        plt = _import_matplotlib_pyplot()
    except RuntimeError:
        _write_placeholder_png(path)
        return
    x = list(range(len(names)))
    width = 0.25
    plt.figure(figsize=(12, 5.2))
    plt.bar([i - width for i in x], precision, width=width, label="precision", color="#2563eb")
    plt.bar(x, recall, width=width, label="recall", color="#0ea5e9")
    plt.bar([i + width for i in x], f1, width=width, label="f1", color="#16a34a")
    plt.title("Cumulative Precision / Recall / F1 by Scenario")
    plt.ylabel("score")
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _horizontal_bar_chart(path: Path, names: list[str], values: list[float], title: str, xlabel: str) -> None:
    try:
        plt = _import_matplotlib_pyplot()
    except RuntimeError:
        _write_placeholder_png(path)
        return
    plt.figure(figsize=(10, 5))
    plt.barh(names, values, color="#2563eb")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _scatter_quality_latency_chart(path: Path, rows: list[dict]) -> None:
    try:
        plt = _import_matplotlib_pyplot()
    except RuntimeError:
        _write_placeholder_png(path)
        return
    if not rows:
        _write_placeholder_png(path)
        return
    x = [float(r.get("latency_mean", 0.0)) for r in rows]
    y = [float(r.get("f1_mean", 0.0)) for r in rows]
    labels = [str(r.get("scenario", "unknown")) for r in rows]
    marker_sizes = [60 + 20 * int(r.get("runs", 1)) for r in rows]
    plt.figure(figsize=(11, 5.2))
    plt.scatter(x, y, alpha=0.9, color="#2563eb", s=marker_sizes, edgecolors="#1e3a8a", linewidth=0.6)
    for idx, label in enumerate(labels):
        plt.annotate(label, (x[idx], y[idx]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.title("Cumulative Quality vs Latency by Scenario")
    plt.xlabel("latency_per_sample_s")
    plt.ylabel("f1_mean")
    plt.grid(alpha=0.25, linestyle="--")
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _scenario_name(row: dict) -> str:
    pair = str(row.get("pair") or "").strip()
    scenario = str(row.get("baseline") or row.get("scenario") or row.get("variant") or "").strip()
    if pair and scenario:
        return f"{pair}::{scenario}"
    if scenario:
        return scenario
    if pair:
        return f"{pair}::default"
    return "unknown"


def _scenario_baseline(row: dict) -> str:
    scenario = str(row.get("baseline") or row.get("scenario") or row.get("variant") or "").strip()
    if scenario and "::" in scenario:
        return scenario.split("::", 1)[1]
    return scenario or "unknown"


def _scenario_workflow(scenario: str) -> str:
    flags = COMPONENT_FLAGS.get(scenario, {})
    disabled = sorted(k for k, enabled in flags.items() if enabled is False)
    if not disabled:
        return "All major components enabled"
    return "Disabled: " + ", ".join(disabled)


def _scenario_type(scenario: str) -> str:
    key = scenario.lower()
    ablation_breakdown = {
        "ablation_no_retrieval": "ablation_no_retrieval",
        "ablation_no_rules": "ablation_no_rules",
        "ablation_no_llm": "ablation_no_llm",
    }
    if key in ablation_breakdown:
        return ablation_breakdown[key]
    if key.startswith("ablation_"):
        return "ablation_other"
    if "hybrid" in key or "adaptive" in key or "full_framework" in key:
        return "full framework"
    if "rule" in key:
        return "rules-focused"
    if "graph" in key or "rag" in key or "retrieval" in key:
        return "retrieval-focused"
    if "llm" in key:
        return "llm-focused"
    return "other"


def _aggregate_rows_by_scenario_type(rows: list[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        scenario = _scenario_baseline(row)
        scenario_type = _scenario_type(scenario)
        buckets.setdefault(scenario_type, []).append(row)

    aggregated: list[dict] = []
    for scenario_type, items in buckets.items():
        f1_values = [float(i.get("f1", 0.0)) for i in items]
        validity_values = [float(i.get("validity_pass_rate", 0.0)) for i in items]
        precision_values = [float(i.get("precision", 0.0)) for i in items]
        recall_values = [float(i.get("recall", 0.0)) for i in items]
        latency_values = [float(i.get("latency_per_sample_s", 0.0)) for i in items]
        runtime_values = [float(i.get("runtime_per_scenario_s", 0.0)) for i in items]
        scenario_set = sorted({_scenario_baseline(i) for i in items})
        pair_set = sorted({str(i.get("pair", "aggregate")) for i in items})
        aggregated.append(
            {
                "scenario_type": scenario_type,
                "scenario_count": len(scenario_set),
                "scenarios": ", ".join(scenario_set),
                "pairs_covered": ", ".join(pair_set),
                "runs": len(items),
                "precision_mean": fmean(precision_values) if precision_values else 0.0,
                "recall_mean": fmean(recall_values) if recall_values else 0.0,
                "f1_mean": fmean(f1_values) if f1_values else 0.0,
                "validity_mean": fmean(validity_values) if validity_values else 0.0,
                "latency_mean": fmean(latency_values) if latency_values else 0.0,
                "runtime_mean": fmean(runtime_values) if runtime_values else 0.0,
            }
        )

    return sorted(aggregated, key=lambda r: r["f1_mean"], reverse=True)


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


def _aggregate_rows_by_scenario(rows: list[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(_scenario_baseline(row), []).append(row)

    aggregated: list[dict] = []
    for scenario, items in buckets.items():
        f1_values = [float(i.get("f1", 0.0)) for i in items]
        validity_values = [float(i.get("validity_pass_rate", 0.0)) for i in items]
        pair_set = sorted({str(i.get("pair", "aggregate")) for i in items})
        mean_f1 = sum(f1_values) / max(len(f1_values), 1)
        mean_validity = sum(validity_values) / max(len(validity_values), 1)
        aggregated.append(
            {
                "scenario": scenario,
                "workflow": _scenario_workflow(scenario),
                "pairs_covered": ", ".join(pair_set),
                "runs": len(items),
                "f1_mean": mean_f1,
                "f1_std": (sum((x - mean_f1) ** 2 for x in f1_values) / max(len(f1_values), 1)) ** 0.5 if f1_values else 0.0,
                "validity_mean": mean_validity,
                "validity_std": (sum((x - mean_validity) ** 2 for x in validity_values) / max(len(validity_values), 1)) ** 0.5 if validity_values else 0.0,
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
    if not violations:
        return [{"reason": "none", "count": 0}]

    breakdown = dict(violations)
    target_validator_errors = sum(int(count) for reason, count in violations.items() if str(reason).startswith("target_"))
    if target_validator_errors:
        breakdown["target_validator_errors"] = target_validator_errors

    return [
        {"reason": str(reason), "count": int(count)}
        for reason, count in sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)
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
    scenario_grouped_rows = _aggregate_rows_by_scenario(rows)
    scenario_type_rows = _aggregate_rows_by_scenario_type(rows)
    violations = _aggregate_violations(canonical_rows)
    violation_rows = [
        {"violation_type": k, "count": v}
        for k, v in sorted(violations.items(), key=lambda kv: kv[1], reverse=True)
    ]
    validity_breakdown = _build_validity_breakdown(violations)

    summary_rows = [
        {
            "scenario": r["scenario"].split("::", 1)[1] if "::" in r["scenario"] else r["scenario"],
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
        "scenario_grouped_summary": scenario_grouped_rows,
        "scenario_type_summary": scenario_type_rows,
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
    with (metrics_dir / "scenario_type_summary.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "scenario_type",
                "scenario_count",
                "runs",
                "precision_mean",
                "recall_mean",
                "f1_mean",
                "validity_mean",
                "latency_mean",
                "runtime_mean",
                "pairs_covered",
                "scenarios",
            ],
        )
        writer.writeheader()
        for row in scenario_type_rows:
            writer.writerow(row)

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
        "## Cumulative scenario-type summary (all scenario families)",
        _markdown_table(
            [
                {
                    "scenario_type": r["scenario_type"],
                    "scenario_count": r["scenario_count"],
                    "runs": r["runs"],
                    "precision_mean": _fmt(r["precision_mean"]),
                    "recall_mean": _fmt(r["recall_mean"]),
                    "f1_mean": _fmt(r["f1_mean"]),
                    "validity_mean": _fmt(r["validity_mean"]),
                    "latency_mean": _fmt(r["latency_mean"]),
                }
                for r in scenario_type_rows
            ],
            ["scenario_type", "scenario_count", "runs", "precision_mean", "recall_mean", "f1_mean", "validity_mean", "latency_mean"],
        ),
        "",
        "## Scenario grouped summary (all pairs/seeds)",
        _markdown_table(
            [
                {
                    "scenario": r["scenario"],
                    "workflow": r["workflow"],
                    "pairs_covered": r["pairs_covered"],
                    "runs": r["runs"],
                    "f1_mean": _fmt(r["f1_mean"]),
                    "validity_mean": _fmt(r["validity_mean"]),
                }
                for r in scenario_grouped_rows
            ],
            ["scenario", "workflow", "pairs_covered", "runs", "f1_mean", "validity_mean"],
        ),
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
        "- `plots/precision_recall_f1_by_scenario.png`",
        "- `plots/quality_vs_latency_scatter.png`",
        "- `plots/latency_by_scenario.png`",
        "- `plots/retrieval_recall_by_scenario.png`",
        "- `plots/f1_by_scenario_grouped.png`",
        "- `plots/validity_by_scenario_grouped.png`",
        "- `plots/f1_by_scenario_type.png`",
        "- `plots/latency_by_scenario_type.png`",
        "",
        "## Raw JSON details",
        "```json",
        json.dumps(report_payload, indent=2),
        "```",
    ]
    (run_dir / "report.md").write_text("\n".join(md))

    rows_by_grouped_scenario: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_grouped_scenario.setdefault(_scenario_baseline(row), []).append(row)

    def _grouped_mean(scenario: str, key: str) -> float:
        values = [float(item.get(key, 0.0)) for item in rows_by_grouped_scenario.get(scenario, [])]
        return fmean(values) if values else 0.0

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
    _grouped_metric_chart(
        plots_dir / "precision_recall_f1_by_scenario.png",
        [r["scenario"] for r in scenario_grouped_rows],
        [_grouped_mean(r["scenario"], "precision") for r in scenario_grouped_rows],
        [_grouped_mean(r["scenario"], "recall") for r in scenario_grouped_rows],
        [r["f1_mean"] for r in scenario_grouped_rows],
    )
    _scatter_quality_latency_chart(plots_dir / "quality_vs_latency_scatter.png", scenario_grouped_rows)
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
        [r["scenario"] for r in scenario_grouped_rows],
        [_grouped_mean(r["scenario"], "retrieval_recall_at_5") for r in scenario_grouped_rows],
        "Cumulative Retrieval Recall@5 by Scenario",
        "recall@5",
    )
    _bar_chart(
        plots_dir / "f1_by_scenario_grouped.png",
        [r["scenario"] for r in scenario_grouped_rows],
        [r["f1_mean"] for r in scenario_grouped_rows],
        "F1 by Scenario (grouped across pairs)",
        "f1",
    )
    _bar_chart(
        plots_dir / "validity_by_scenario_grouped.png",
        [r["scenario"] for r in scenario_grouped_rows],
        [r["validity_mean"] for r in scenario_grouped_rows],
        "Validity by Scenario (grouped across pairs)",
        "validity_pass_rate",
    )
    _horizontal_bar_chart(
        plots_dir / "f1_by_scenario_type.png",
        [r["scenario_type"] for r in scenario_type_rows],
        [r["f1_mean"] for r in scenario_type_rows],
        "Cumulative F1 by Scenario Type",
        "f1",
    )
    _horizontal_bar_chart(
        plots_dir / "latency_by_scenario_type.png",
        [r["scenario_type"] for r in scenario_type_rows],
        [r["latency_mean"] for r in scenario_type_rows],
        "Cumulative Latency by Scenario Type",
        "seconds",
    )

    summary_table_html = _html_table(summary_rows, ["pair", "scenario", "f1", "validity_pass_rate"])
    validity_table = html.escape(_markdown_table(validity_breakdown, ["reason", "count"]))
    raw_json = html.escape(json.dumps(report_payload, indent=2))
    scenario_type_table_html = _html_table(
        [
            {
                "scenario_type": r["scenario_type"],
                "scenario_count": r["scenario_count"],
                "runs": r["runs"],
                "f1_mean": _fmt(r["f1_mean"]),
                "validity_mean": _fmt(r["validity_mean"]),
                "latency_mean": _fmt(r["latency_mean"]),
            }
            for r in scenario_type_rows
        ],
        ["scenario_type", "scenario_count", "runs", "f1_mean", "validity_mean", "latency_mean"],
    )
    html_content = f"""<html><head><style>
body{{font-family:Inter,Segoe UI,Arial,sans-serif;margin:0;background:#f8fafc;color:#0f172a;}}
.container{{max-width:1200px;margin:0 auto;padding:28px;}}
h1,h2{{margin:0 0 12px 0;}}
.card{{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:18px 20px;margin-bottom:18px;box-shadow:0 2px 10px rgba(15,23,42,.04);}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;}}
img{{max-width:100%;height:auto;border-radius:10px;border:1px solid #cbd5e1;background:white;}}
small{{color:#475569;}}
</style></head><body><div class="container">
<h1>ISynKGR Benchmark Report</h1>
<p><small>Canonical metric keys: <code>precision</code>, <code>recall</code>, <code>f1</code>, <code>validity_pass_rate</code>, <code>violation_counts</code>.</small></p>
<div class="card">
<h2>Executive Summary</h2>
{summary_table_html}
</div>
<div class="card">
<h2>Cumulative Results by Scenario Type</h2>
<p><small>All charts and tables in this section aggregate run data by scenario family.</small></p>
{scenario_type_table_html}
</div>
<div class="card">
<h2>Quality and Reliability Diagnostics</h2>
<pre>{validity_table}</pre>
</div>
<div class="card">
<h2>Visual Comparison Dashboard</h2>
<div class="grid">
<div><img alt="F1 by scenario grouped across pairs" src="plots/f1_by_scenario_grouped.png" /></div>
<div><img alt="Validity by scenario grouped across pairs" src="plots/validity_by_scenario_grouped.png" /></div>
<div><img alt="Cumulative F1 by scenario type" src="plots/f1_by_scenario_type.png" /></div>
<div><img alt="Cumulative latency by scenario type" src="plots/latency_by_scenario_type.png" /></div>
<div><img alt="Precision, recall, F1 by scenario" src="plots/precision_recall_f1_by_scenario.png" /></div>
<div><img alt="Quality vs latency scatter" src="plots/quality_vs_latency_scatter.png" /></div>
<div><img alt="Top violations" src="plots/top_violations.png" /></div>
<div><img alt="Retrieval recall by scenario" src="plots/retrieval_recall_by_scenario.png" /></div>
</div>
</div>
<div class="card">
<h2>Raw JSON details</h2>
<details>
<summary>Expand raw JSON details</summary>
<pre>{raw_json}</pre>
</details>
</div>
</div></body></html>"""
    (run_dir / "report.html").write_text(html_content)


def generate_final_report(results_root: Path = Path("results")) -> Path:
    def _enrich_from_path(row: dict, metrics_path: Path) -> dict:
        enriched = dict(row)
        rel_parts = metrics_path.relative_to(results_root).parts
        # Common layout: results/<PAIR>/<SCENARIO>/seed*/metrics.json
        if len(rel_parts) >= 4:
            enriched.setdefault("pair", str(enriched.get("pair") or rel_parts[-4]))
            enriched.setdefault("baseline", str(enriched.get("baseline") or rel_parts[-3]))
        return enriched

    rows = []
    for metrics_path in results_root.glob("**/metrics.json"):
        payload = json.loads(metrics_path.read_text())
        if isinstance(payload, list):
            for row in payload:
                if not isinstance(row, dict):
                    continue
                enriched = _enrich_from_path(row, metrics_path)
                if _scenario_name(enriched) != "unknown":
                    rows.append(enriched)
        elif isinstance(payload, dict):
            enriched = _enrich_from_path(payload, metrics_path)
            if _scenario_name(enriched) != "unknown":
                rows.append(enriched)

    final_dir = results_root / "final"
    write_report(final_dir, rows)
    return final_dir
