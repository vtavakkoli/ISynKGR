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


def test_evaluate_run_treats_label_match_as_equivalent(tmp_path):
    out_dir = tmp_path / "predictions"
    out_dir.mkdir(parents=True)

    pred_rows = [
        {
            "source_path": "opcua://ns=2;i=1000",
            "target_path": "aas://aas-0/submodel/default/element/value",
            "mapping_type": "label_match",
            "confidence": 0.95,
            "rationale": "rule style mapping",
            "evidence": [],
        }
    ]
    gt_rows = [
        {
            "source_path": "opcua://ns=2;i=1000",
            "target_path": "aas://aas-0/submodel/default/element/value",
            "mapping_type": "equivalent",
            "confidence": 1.0,
            "rationale": "ground truth",
            "evidence": [],
        }
    ]

    _write_jsonl(out_dir / "mappings.jsonl", pred_rows)
    _write_jsonl(tmp_path / "ground_truth.jsonl", gt_rows)

    metrics = evaluate_run(out_dir)
    assert metrics["matched_count"] == 1
    assert metrics["f1"] == 1.0


def test_evaluate_exports_difficulty_and_error_summary(tmp_path):
    out_dir = tmp_path / "predictions"
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True)
    pred_rows = [
        {
            "source_path": "opcua://ns=2;i=1000",
            "target_path": "",
            "mapping_type": "no_match",
            "confidence": 0.2,
            "rationale": "no match generated",
            "evidence": [],
        }
    ]
    gt_rows = pred_rows
    _write_jsonl(out_dir / "mappings.jsonl", pred_rows)
    _write_jsonl(tmp_path / "ground_truth.jsonl", gt_rows)
    (out_dir / "validation.json").write_text(json.dumps([{"valid": False, "violations": [{"type": "schema_invalid"}]}]))
    _write_jsonl(pred_dir / "sample_results.jsonl", [{"sample": "s1", "tier": "noisy", "pair": "OPCUA->AAS", "difficulty": "hard", "matched": True}])
    _write_jsonl(pred_dir / "decision_trace.jsonl", [{"selected_strategy": "llm", "pair": "OPCUA->AAS", "tier": "noisy", "difficulty": "hard", "matched": True}])

    metrics = evaluate_run(out_dir)
    assert metrics["per_difficulty"]["hard"]["count"] == 1
    assert metrics["adaptive_strategy_usage"]["llm"] == 1
    summary = json.loads((out_dir / "error_summary.json").read_text())
    assert "schema_invalid" in summary
