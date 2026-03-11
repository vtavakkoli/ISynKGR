from __future__ import annotations

import json
from pathlib import Path

from benchmark import final_report


def test_final_report_uses_gt_subset_matching_processed_samples(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "datasets" / "v1" / "crosswalk").mkdir(parents=True)
    gt_rows = [
        {
            "source_path": f"opcua://ns=2;i={1000+i}",
            "target_path": f"aas://aas-{i}/submodel/default/element/value",
            "mapping_type": "equivalent",
            "transform": None,
            "confidence": 1.0,
            "rationale": "Synthetic ground truth row.",
            "evidence": [],
        }
        for i in range(12)
    ]
    (tmp_path / "datasets" / "v1" / "crosswalk" / "gt_mappings.jsonl").write_text("\n".join(json.dumps(r) for r in gt_rows) + "\n")

    def _fake_run_local_baseline(_mode: str, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        pred = gt_rows[:10]
        (out_dir / "mappings.jsonl").write_text("\n".join(json.dumps(r) for r in pred) + "\n")
        (out_dir / "validation.json").write_text("[]")

    monkeypatch.setattr(final_report, "_run_local_baseline", _fake_run_local_baseline)

    out = final_report.generate_final_report()
    subset = [json.loads(line) for line in (out / "ground_truth.jsonl").read_text().splitlines() if line.strip()]
    assert len(subset) == 10

