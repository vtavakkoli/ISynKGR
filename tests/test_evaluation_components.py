from pathlib import Path

import pytest

from isynkgr.evaluation import components


def test_load_standards_from_config(tmp_path: Path) -> None:
    config = tmp_path / "standards.json"
    config.write_text('{"standards": ["s1", "s2"]}')
    assert components.load_standards(config) == ["s1", "s2"]


def test_load_standards_raises_for_empty_list(tmp_path: Path) -> None:
    config = tmp_path / "standards.json"
    config.write_text('{"standards": []}')
    with pytest.raises(ValueError):
        components.load_standards(config)


def test_build_pairs_excludes_self_edges() -> None:
    assert components.build_pairs(["a", "b"]) == [("a", "b"), ("b", "a")]


def test_build_graph_and_predict_and_score() -> None:
    sample = {
        "sample_id": "ieee1451_7",
        "standard": "ieee1451",
        "terms": ["temp"],
        "entities": [{"id": "ieee1451.Sensor"}],
        "properties": [{"name": "reading"}],
    }
    graph = components.build_graph(sample)
    assert graph["nodes"][1]["id"] == "ieee1451.Sensor:reading"
    assert components.predict_name(sample, "opcua62541", "rag").endswith("_rag")
    assert components.predict_name(sample, "opcua62541", "llm_only") == "opcua62541_guess_7"
    assert components.score("x", "x")["f1"] == 1.0


def test_save_outputs_writes_latest_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(components, "_plot", lambda rows, path: None)
    rows = [{"method": "isynkgr", "f1": 1.0, "source": "a", "target": "b"}]
    out = components.save_outputs(rows, tmp_path)
    assert (out / "metrics.json").exists()
    assert (tmp_path / "latest" / "metrics.csv").exists()
