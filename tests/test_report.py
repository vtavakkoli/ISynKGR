import json

from benchmark.report import write_report


def test_write_report_emits_canonical_outputs(tmp_path):
    rows = [
        {
            "baseline": "baseline",
            "precision": 0.4,
            "recall": 0.5,
            "f1": 0.444,
            "validity_pass_rate": 0.8,
            "violation_counts": {
                "mapping_type_invalid": 2,
                "target_id_format": 1,
                "target_missing": 3,
                "confidence_low": 4,
            },
        },
        {
            "baseline": "full_framework",
            "precision": 0.7,
            "recall": 0.6,
            "f1": 0.646,
            "validity_pass_rate": 0.9,
            "violation_counts": {},
        },
    ]

    write_report(tmp_path, rows)

    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "plots" / "f1_by_scenario.png").exists()
    assert (tmp_path / "plots" / "validity_by_scenario.png").exists()
    assert (tmp_path / "plots" / "top_violations.png").exists()

    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["canonical_metric_keys"] == ["precision", "recall", "f1", "validity_pass_rate", "violation_counts"]

    reasons = {r["reason"]: r["count"] for r in payload["why_validity_low"]}
    assert reasons["mapping_type_invalid"] == 2
    assert reasons["target_id_format"] == 1
    assert reasons["target_validator_errors"] == 4
    assert reasons["confidence_low"] == 4

    html = (tmp_path / "report.html").read_text()
    assert "<details>" in html
    assert "Expand raw JSON details" in html
