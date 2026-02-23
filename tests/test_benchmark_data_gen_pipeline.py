import json
from pathlib import Path

from benchmark.data_gen.pipeline import SCENARIOS, generate_pipeline
from isynkgr.icr.mapping_schema import ingest_mapping_payload


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_pipeline_generates_all_scenarios_and_cardinality(tmp_path):
    out_dir = tmp_path / "data_gen"
    summary = generate_pipeline(out_dir, seed=7, sample_size=3)

    assert summary["scenario_count"] == 6
    assert summary["dataset_rows"] == 18
    assert summary["gt_rows"] == 21  # structural_map scenarios produce one-to-many

    for scenario in SCENARIOS:
        slug = scenario.replace("->", "_to_")
        assert (out_dir / "scenarios" / slug / "source").exists()
        assert (out_dir / "scenarios" / slug / "target").exists()

    dataset_rows = _read_jsonl(out_dir / "dataset.jsonl")
    gt_rows = _read_jsonl(out_dir / "ground_truth.jsonl")

    counts: dict[str, int] = {}
    for row in gt_rows:
        validated = ingest_mapping_payload(row, migrate_legacy=False).model_dump()
        key = validated["source_path"]
        counts[key] = counts.get(key, 0) + 1

    for row in dataset_rows:
        source_path = row["mapping_source_path"]
        expected = row["cardinality_contract"]["expected_count"]
        assert expected == counts[source_path]


def test_pipeline_includes_required_transform_kinds(tmp_path):
    out_dir = tmp_path / "data_gen"
    generate_pipeline(out_dir, seed=3, sample_size=2)
    gt_rows = _read_jsonl(out_dir / "ground_truth.jsonl")

    transform_kinds = {
        (row.get("transform") or {}).get("args", {}).get("kind")
        for row in gt_rows
        if row.get("transform")
    }
    assert {"unit_convert", "cast", "id_normalize", "structural_map"}.issubset(transform_kinds)
