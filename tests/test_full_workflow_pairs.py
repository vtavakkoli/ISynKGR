from pathlib import Path

from benchmark.full_workflow import _build_pair_dataset, _pair_supported, _synthetic_id_for_standard
from isynkgr.icr.path_validation import is_valid_protocol_path


def test_pair_dataset_loader_creates_pair_specific_files(tmp_path: Path):
    pair_dir = _build_pair_dataset(tmp_path, "OPCUA", "AAS", max_rows=3)
    assert (pair_dir / "dataset.jsonl").exists()
    assert (pair_dir / "ground_truth.jsonl").exists()
    rows = (pair_dir / "dataset.jsonl").read_text().strip().splitlines()
    assert len(rows) == 3


def test_unsupported_pair_reporting():
    ok, reason = _pair_supported("UNKNOWN_SRC", "AAS")
    assert ok is False
    assert "source adapter" in reason


def test_synthetic_ids_remain_protocol_valid():
    assert is_valid_protocol_path(_synthetic_id_for_standard("AAS", 3, ""))
    assert is_valid_protocol_path(_synthetic_id_for_standard("OPCUA", 3, ""))
    assert is_valid_protocol_path(_synthetic_id_for_standard("IEEE1451", 3, ""))
    assert is_valid_protocol_path(_synthetic_id_for_standard("IEC61499", 3, ""))
