from pathlib import Path

from benchmark.validate_dataset import validate_or_generate


def test_validate_dataset_counts_exist():
    counts = validate_or_generate(Path("datasets/v1"))
    assert counts["opcua_synthetic"] >= 100
    assert counts["aas_synthetic"] >= 100
    assert counts["opcua_semi_real"] >= 10
    assert counts["aas_semi_real"] >= 10
