from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def _markdown_table(rows: list[dict], columns: list[str]) -> str:
    header = "|" + "|".join(columns) + "|"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    body = ["|" + "|".join(str(r.get(c, "")) for c in columns) + "|" for r in rows]
    return "\n".join([header, sep, *body])


def write_report(run_dir: Path, rows: list[dict]) -> None:
    ranked_f1 = sorted(rows, key=lambda r: r.get("f1", 0.0), reverse=True)
    ranked_validity = sorted(rows, key=lambda r: r.get("validity_pass_rate", 0.0), reverse=True)
    violations: dict[str, int] = {}
    for row in rows:
        for k, v in (row.get("violation_counts") or {}).items():
            violations[k] = violations.get(k, 0) + int(v)

    report_rows = [
        {
            "scenario": r.get("scenario", r.get("baseline", "")),
            "f1": f"{r.get('f1', 0.0):.3f}",
            "validity_pass_rate": f"{r.get('validity_pass_rate', 0.0):.3f}",
            "coverage": f"{r.get('coverage', 0.0):.3f}",
        }
        for r in ranked_f1
    ]

    coverage_rows = [
        {
            "scenario": r.get("scenario", r.get("baseline", "")),
            "gt_count": r.get("gt_count", 0),
            "pred_count": r.get("pred_count", 0),
            "matched_count": r.get("matched_count", 0),
            "coverage": f"{r.get('coverage', 0.0):.3f}",
        }
        for r in rows
    ]

    md = [
        "# ISynKGR Final Benchmark Report",
        "",
        "## Setup + Dataset Details",
        f"- model: {rows[0].get('model', 'unknown') if rows else 'unknown'}",
        f"- seed: {rows[0].get('seed', 'unknown') if rows else 'unknown'}",
        f"- items: {rows[0].get('items', 'unknown') if rows else 'unknown'}",
        f"- tier: {rows[0].get('tier', 'unknown') if rows else 'unknown'}",
        "- standards_run: [\"OPCUA->AAS\"]",
        "- standards_declared_but_skipped: [{\"standard\": \"others\", \"reason\": \"not implemented in current benchmark path\"}]",
        "",
        "## Ranking by F1",
        _markdown_table(report_rows, ["scenario", "f1", "validity_pass_rate", "coverage"]),
        "",
        "## Ranking by Validity",
        _markdown_table(
            [
                {"scenario": r.get("scenario", r.get("baseline", "")), "validity_pass_rate": f"{r.get('validity_pass_rate', 0.0):.3f}", "f1": f"{r.get('f1', 0.0):.3f}"}
                for r in ranked_validity
            ],
            ["scenario", "validity_pass_rate", "f1"],
        ),
        "",
        "## Coverage",
        _markdown_table(coverage_rows, ["scenario", "gt_count", "pred_count", "matched_count", "coverage"]),
        "",
        "## Error Taxonomy Summary",
        _markdown_table(
            [{"violation": k, "count": v} for k, v in sorted(violations.items(), key=lambda kv: kv[1], reverse=True)] or [{"violation": "none", "count": 0}],
            ["violation", "count"],
        ),
        "",
        "## Per-Scenario Metrics",
    ]
    for row in ranked_f1:
        md.extend(
            [
                f"### {row.get('scenario', row.get('baseline', 'unknown'))}",
                "```json",
                json.dumps(row, indent=2),
                "```",
            ]
        )
    md.extend(["", "## Limitations & Next Steps", "- Only OPCUA->AAS path executed in this run.", "- Extend standards and validators for additional pairs."])
    (run_dir / "report.md").write_text("\n".join(md))


def _bar_chart(path: Path, names: list[str], values: list[float], title: str, ylabel: str) -> None:
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
    plt.figure(figsize=(6, 4))
    plt.scatter([float(r.get("time_s", 0.0)) for r in rows], [float(r.get("f1", 0.0)) for r in rows])
    plt.title("time_s vs F1")
    plt.xlabel("time_s")
    plt.ylabel("f1")
    plt.tight_layout()
    plt.savefig(charts_dir / "time_vs_f1.png")
    plt.close()
    return final_dir
