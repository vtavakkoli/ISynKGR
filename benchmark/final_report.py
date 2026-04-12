from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmark.data_gen.pipeline import _synthetic_id_for_standard
from benchmark.evaluate import evaluate_run
from benchmark.report import write_report
from isynkgr.pipeline.adaptive_candidate_ranker import TranslatorConfig
from isynkgr.translator import Translator

BASELINES = ["rule_only", "graph_only", "adaptive_candidate_ranker", "rag_only", "llm_only"]


def normalize_target_path(path: str | None) -> str | None:
    raw = str(path or "").strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in {"<none>", "no_match"}:
        return None
    return raw


def _component_attribution(evidence: list[str] | None) -> str:
    evidence_items = [str(item).lower() for item in (evidence or [])]
    parts: list[str] = []
    if any("rules" in item for item in evidence_items):
        parts.append("rules")
    if any("retrieval" in item for item in evidence_items):
        parts.append("retrieval")
    if any("llm" in item for item in evidence_items):
        parts.append("llm")
    if not parts:
        return "unknown"
    if len(parts) == 1:
        return f"{parts[0]}_only"
    return "+".join(parts)


def _load_gt_subset(limit: int) -> list[dict]:
    gt = Path("datasets/v1/crosswalk/gt_mappings.jsonl")
    rows: list[dict] = []
    for idx, line in enumerate(gt.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        source_standard = str(row.get("source_standard") or "OPCUA")
        target_standard = str(row.get("target_standard") or "AAS")
        source_raw = str(row.get("source_path") or row.get("source_id") or "")
        target_raw = str(row.get("target_path") or row.get("target_id") or "")
        source_path = _synthetic_id_for_standard(source_standard, idx, source_raw)
        target_path = _synthetic_id_for_standard(target_standard, idx, target_raw)
        rows.append(
            {
                **row,
                "source_path": source_path,
                "target_path": target_path,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _run_local_baseline(mode: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    translator = Translator(TranslatorConfig(seed=42))
    mapping_lines = []
    validations = []
    opc_files = sorted(Path("datasets/v1/opcua/synthetic").glob("*.xml"))[:10]
    for f in opc_files:
        idx = int(f.stem.split("_")[-1])
        result = translator.translate("opcua", "aas", str(f), mode=mode if mode != "adaptive_candidate_ranker" else "adaptive_candidate_ranker")
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

    gt_subset = _load_gt_subset(limit=10)
    gt_subset_text = "\n".join(json.dumps(row) for row in gt_subset) + "\n"
    (final_dir / "gt_mappings.jsonl").write_text(gt_subset_text)
    (final_dir / "ground_truth.jsonl").write_text(gt_subset_text)

    rows = []
    comparison_rows: list[dict] = []
    for mode in BASELINES:
        out_dir = final_dir / mode
        _run_local_baseline(mode, out_dir)
        metrics = evaluate_run(out_dir)
        metrics["baseline"] = mode
        pred_rows: dict[str, dict] = {}
        for line in (out_dir / "mappings.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            pred_rows[str(row.get("source_path") or "")] = row

        attribution_counts: dict[str, int] = {}
        for idx, gt_row in enumerate(gt_subset):
            source_path = str(gt_row.get("source_path") or "")
            expected_raw = normalize_target_path(gt_row.get("target_path"))
            pred_row = pred_rows.get(source_path, {})
            predicted_raw = normalize_target_path(str(pred_row.get("target_path") or ""))
            expected = normalize_target_path(
                _synthetic_id_for_standard(str(gt_row.get("target_standard") or "AAS"), idx, expected_raw or "")
            )
            predicted = normalize_target_path(
                _synthetic_id_for_standard(str(gt_row.get("target_standard") or "AAS"), idx, predicted_raw or "")
            )
            matched = predicted == expected
            attribution = _component_attribution(pred_row.get("evidence") or [])
            attribution_counts[attribution] = attribution_counts.get(attribution, 0) + 1
            comparison_rows.append(
                {
                    "baseline": mode,
                    "source_path": source_path,
                    "expected_target_path": expected or "<none>",
                    "predicted_target_path": predicted or "<none>",
                    "match": matched,
                    "component_attribution": attribution,
                }
            )
        metrics["component_attribution"] = ";".join(
            f"{name}:{count}" for name, count in sorted(attribution_counts.items())
        )
        rows.append(metrics)

    fieldnames = sorted({k for r in rows for k in r.keys() if k != "violation_counts"})
    with (final_dir / "metrics.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            copy = row.copy()
            copy.pop("violation_counts", None)
            writer.writerow(copy)
    with (final_dir / "comparison.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "baseline",
                "source_path",
                "expected_target_path",
                "predicted_target_path",
                "match",
                "component_attribution",
            ],
        )
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow(row)

    (final_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    write_report(final_dir, rows)
    return final_dir


if __name__ == "__main__":
    out = generate_final_report()
    print(f"final-report-generated: {out}")
