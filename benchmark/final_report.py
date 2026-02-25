from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmark.evaluate import evaluate_run
from benchmark.report import write_report
from isynkgr.pipeline.hybrid import TranslatorConfig
from isynkgr.translator import Translator

BASELINES = ["rule_only", "graph_only", "isynkgr_hybrid", "rag_only", "llm_only"]


def _run_local_baseline(mode: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    translator = Translator(TranslatorConfig(seed=42))
    mapping_lines = []
    validations = []
    opc_files = sorted(Path("datasets/v1/opcua/synthetic").glob("*.xml"))[:10]
    for f in opc_files:
        idx = int(f.stem.split("_")[-1])
        result = translator.translate("opcua", "aas", str(f), mode=mode if mode != "isynkgr_hybrid" else "hybrid")
        for m in result.mappings:
            mapping_lines.append(json.dumps(m.model_dump()))
        if not result.mappings:
            mapping_lines.append(
                json.dumps(
                    {
                        "source_path": f"opcua://ns=2;i={1000+idx}",
                        "target_path": "",
                        "mapping_type": "no_match",
                        "transform": None,
                        "confidence": 0.0,
                        "rationale": "No mappings produced by report runner.",
                        "evidence": [],
                    }
                )
            )
        validations.append(result.validation_report.model_dump())
    (out_dir / "mappings.jsonl").write_text("\n".join(mapping_lines) + "\n")
    (out_dir / "validation.json").write_text(json.dumps(validations, indent=2))
    (out_dir / "provenance.json").write_text(json.dumps({"mode": mode, "runner": "local-final-report"}, indent=2))


def generate_final_report() -> Path:
    final_dir = Path("results/final")
    final_dir.mkdir(parents=True, exist_ok=True)

    gt = Path("datasets/v1/crosswalk/gt_mappings.jsonl")
    (final_dir / "gt_mappings.jsonl").write_text(gt.read_text())

    rows = []
    for mode in BASELINES:
        out_dir = final_dir / mode
        _run_local_baseline(mode, out_dir)
        metrics = evaluate_run(out_dir)
        metrics["baseline"] = mode
        rows.append(metrics)

    fieldnames = sorted({k for r in rows for k in r.keys() if k != "violation_counts"})
    with (final_dir / "metrics.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            copy = row.copy()
            copy.pop("violation_counts", None)
            writer.writerow(copy)

    (final_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    write_report(final_dir, rows)
    return final_dir


if __name__ == "__main__":
    out = generate_final_report()
    print(f"final-report-generated: {out}")
