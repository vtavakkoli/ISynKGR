import json
from benchmark.full_workflow import run_full_workflow


def test_full_workflow_fast_mode_generates_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "benchmark").mkdir()
    cfg = {
        "seed": 42,
        "run_id_prefix": "testrun",
        "standards": ["IEEE1451", "ISO15926", "IEC61499", "OPCUA", "AAS"],
        "pairs": [["OPCUA", "AAS"]],
        "default_samples_per_standard": 20,
        "fast_samples_per_standard": 5,
        "tiers": ["canonical", "noisy", "realistic"],
        "variants": [
            {"name": "full_framework", "graph_rag": True, "parallel_retrieval": True, "community_detection": True, "reasoning_steps": True},
            {"name": "baseline", "graph_rag": False, "parallel_retrieval": False, "community_detection": False, "reasoning_steps": False},
        ],
    }
    (tmp_path / "benchmark" / "benchmark_full.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("BENCHMARK_CONFIG", str(tmp_path / "benchmark" / "benchmark_full.json"))
    monkeypatch.setenv("PROFILE", "fast")
    monkeypatch.setenv("RUN_ID", "testrun_001")

    rc = run_full_workflow()
    assert rc == 0

    root = tmp_path / "artifacts" / "testrun_001"
    assert (root / "dataset.jsonl").exists()
    assert (root / "ground_truth.jsonl").exists()
    assert (root / "metrics.json").exists()
    assert (root / "metrics" / "advanced_analysis.json").exists()
    assert (root / "report.md").exists()
    assert (root / "report.html").exists()
    assert (root / "predictions" / "full_framework.jsonl").exists()
