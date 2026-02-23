from __future__ import annotations

import html
import json
from pathlib import Path


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _markdown_table(rows: list[dict], columns: list[str]) -> str:
    header = "|" + "|".join(columns) + "|"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    body = []
    for row in rows:
        body.append("|" + "|".join(str(row.get(col, "")) for col in columns) + "|")
    return "\n".join([header, sep, *body])


def _bar_svg(items: list[tuple[str, float]], title: str, width: int = 800, row_h: int = 28) -> str:
    if not items:
        return ""
    max_v = max(v for _, v in items) or 1.0
    left = 180
    chart_w = width - left - 20
    height = 60 + row_h * len(items)
    lines = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        f'<text x="10" y="24" font-size="18" font-family="Arial">{html.escape(title)}</text>',
    ]
    y = 50
    for name, value in items:
        bar_w = int((value / max_v) * chart_w)
        safe = html.escape(name)
        lines.append(f'<text x="10" y="{y+14}" font-size="12" font-family="Arial">{safe}</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{bar_w}" height="16" fill="#4e79a7" />')
        lines.append(f'<text x="{left + bar_w + 6}" y="{y+13}" font-size="12" font-family="Arial">{value:.3f}</text>')
        y += row_h
    lines.append("</svg>")
    return "\n".join(lines)


def write_report(run_dir: Path, rows: list[dict]) -> None:
    ranked_f1 = sorted(rows, key=lambda r: r.get("f1", 0.0), reverse=True)
    ranked_validity = sorted(rows, key=lambda r: r.get("validity_pass_rate", 0.0), reverse=True)

    compact_rows = []
    for row in ranked_f1:
        compact_rows.append(
            {
                "baseline": row.get("baseline", row.get("scenario", "")),
                "precision": _fmt(row.get("precision", 0.0)),
                "recall": _fmt(row.get("recall", 0.0)),
                "f1": _fmt(row.get("f1", 0.0)),
                "validity_pass_rate": _fmt(row.get("validity_pass_rate", 0.0)),
            }
        )

    violations: dict[str, int] = {}
    for row in rows:
        for k, v in row.get("violation_counts", {}).items():
            violations[k] = violations.get(k, 0) + int(v)

    violation_rows = [{"violation_type": k, "count": v} for k, v in sorted(violations.items(), key=lambda kv: kv[1], reverse=True)]

    md = [
        "# ISynKGR Benchmark Report",
        "",
        "## Ranking by F1",
        _markdown_table(compact_rows, ["baseline", "precision", "recall", "f1", "validity_pass_rate"]),
        "",
        "## Ranking by Validity Pass Rate",
        _markdown_table(
            [
                {
                    "baseline": row.get("baseline", row.get("scenario", "")),
                    "validity_pass_rate": _fmt(row.get("validity_pass_rate", 0.0)),
                    "f1": _fmt(row.get("f1", 0.0)),
                }
                for row in ranked_validity
            ],
            ["baseline", "validity_pass_rate", "f1"],
        ),
        "",
        "## Error Taxonomy",
        _markdown_table(violation_rows or [{"violation_type": "none", "count": 0}], ["violation_type", "count"]),
        "",
        "## Raw Results",
        "```json",
        json.dumps(rows, indent=2),
        "```",
    ]
    (run_dir / "report.md").write_text("\n".join(md))

    f1_items = [(r.get("baseline", r.get("scenario", "")), float(r.get("f1", 0.0))) for r in ranked_f1]
    validity_items = [(r.get("baseline", r.get("scenario", "")), float(r.get("validity_pass_rate", 0.0))) for r in ranked_validity]
    violation_items = [(v["violation_type"], float(v["count"])) for v in violation_rows[:8]]

    summary_table = html.escape(_markdown_table(compact_rows, ["baseline", "precision", "recall", "f1", "validity_pass_rate"]))
    raw_json = html.escape(json.dumps(rows, indent=2))
    html_content = f"""<html><body style="font-family:Arial,sans-serif;margin:24px">
<h1>ISynKGR Benchmark Report</h1>
<h2>Summary Table</h2>
<pre>{summary_table}</pre>
<h2>Plots</h2>
{_bar_svg(f1_items, "F1 by Baseline")}
{_bar_svg(validity_items, "Validity Pass Rate by Baseline")}
{_bar_svg(violation_items, "Top Violation Counts")}
<h2>Raw JSON</h2>
<pre>{raw_json}</pre>
</body></html>"""
    (run_dir / "report.html").write_text(html_content)


def _import_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to generate PNG charts. "
            "Install matplotlib or run this in the benchmark Docker image."
        ) from exc
    return plt


def _bar_chart(path: Path, names: list[str], values: list[float], title: str, ylabel: str) -> None:
    plt = _import_matplotlib_pyplot()
    plt.figure(figsize=(9, 4))
    plt.bar(names, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


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
            rows.append(json.loads(metrics_path.read_text()))

    final_dir = results_root / "final"
    charts_dir = final_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    (final_dir / "metrics_merged.json").write_text(json.dumps(rows, indent=2))
    write_report(final_dir, rows)

    names = [r.get("scenario", r.get("baseline", "")) for r in rows]
    _bar_chart(charts_dir / "f1_by_scenario.png", names, [float(r.get("f1", 0.0)) for r in rows], "F1 by Scenario", "F1")
    _bar_chart(
        charts_dir / "validity_by_scenario.png",
        names,
        [float(r.get("validity_pass_rate", 0.0)) for r in rows],
        "Validity Pass Rate by Scenario",
        "validity_pass_rate",
    )

    plt = _import_matplotlib_pyplot()
    plt.figure(figsize=(6, 4))
    plt.scatter([float(r.get("time_s", 0.0)) for r in rows], [float(r.get("f1", 0.0)) for r in rows])
    plt.title("time_s vs F1")
    plt.xlabel("time_s")
    plt.ylabel("f1")
    plt.tight_layout()
    plt.savefig(charts_dir / "time_vs_f1.png")
    plt.close()
    return final_dir
