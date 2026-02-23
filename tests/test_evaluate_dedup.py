import json

from benchmark.evaluate import evaluate_run


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_evaluate_run_deduplicates_and_reports_counts(tmp_path):
    out_dir = tmp_path / "predictions"
    out_dir.mkdir(parents=True)

    pred_rows = [
        {
            "source_path": "opcua://ns=2;i=1000",
            "target_path": "aas://aas-0/submodel/default/element/value",
            "mapping_type": "equivalent",
            "confidence": 0.9,
            "rationale": "rationale-a",
            "evidence": [],
        },
        {
            "source_path": "opcua://ns=2;i=1000",
            "target_path": "aas://aas-0/submodel/default/element/value",
            "mapping_type": "equivalent",
            "confidence": 0.7,
            "rationale": "rationale-dup",
            "evidence": [],
        },
    ]
    gt_rows = [pred_rows[0]]

    _write_jsonl(out_dir / "mappings.jsonl", pred_rows)
    _write_jsonl(tmp_path / "ground_truth.jsonl", gt_rows)

    metrics = evaluate_run(out_dir)

    assert metrics["pred_count"] == 1
    assert metrics["gt_count"] == 1
    assert metrics["counts"]["predictions"] == {"raw": 2, "deduplicated": 1}
    assert metrics["counts"]["ground_truth"] == {"raw": 1, "deduplicated": 1}


def test_evaluate_run_mismatch_diagnostics_section(tmp_path):
    out_dir = tmp_path / "predictions"
    out_dir.mkdir(parents=True)

    pred_rows = [
        {
            "source_path": "opcua://ns=2;i=1000",
            "target_path": "aas://aas-0/submodel/default/element/value",
            "mapping_type": "equivalent",
            "confidence": 0.9,
            "rationale": "rationale-a",
            "evidence": [],
        }
    ]
    gt_rows = [
        {
            "source_path": "opcua://ns=2;i=1001",
            "target_path": "aas://aas-1/submodel/default/element/value",
            "mapping_type": "equivalent",
            "confidence": 0.9,
            "rationale": "rationale-b",
            "evidence": [],
        }
    ]

    _write_jsonl(out_dir / "mappings.jsonl", pred_rows)
    _write_jsonl(tmp_path / "ground_truth.jsonl", gt_rows)

    metrics = evaluate_run(out_dir)

    assert "mismatch_diagnostics" in metrics
    assert metrics["mismatch_diagnostics"]["pred_only_count"] == 1
    assert metrics["mismatch_diagnostics"]["gt_only_count"] == 1
