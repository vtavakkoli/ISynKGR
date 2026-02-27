from benchmark.run_sut import _enforce_cardinality


def test_enforce_cardinality_trims_to_expected_count() -> None:
    violations = []
    sample_mappings = [
        {"source_path": "opcua://a", "target_path": "aas://x", "mapping_type": "equivalent", "confidence": 0.6},
        {"source_path": "opcua://b", "target_path": "aas://y", "mapping_type": "equivalent", "confidence": 0.9},
    ]
    out = _enforce_cardinality(sample_mappings, {"mode": "one_to_one", "expected_count": 1, "grouped_1": False}, violations)
    assert len(out) == 1
    assert out[0]["source_path"] == "opcua://b"
    assert violations and violations[0]["type"] == "cardinality_trimmed"


def test_enforce_cardinality_keeps_grouped_contract() -> None:
    violations = []
    sample_mappings = [
        {"source_path": "opcua://a", "target_path": "aas://x", "mapping_type": "equivalent", "confidence": 0.6},
        {"source_path": "opcua://b", "target_path": "aas://y", "mapping_type": "equivalent", "confidence": 0.9},
    ]
    out = _enforce_cardinality(sample_mappings, {"mode": "grouped_1", "expected_count": 0, "grouped_1": True}, violations)
    assert len(out) == 2
    assert violations == []
