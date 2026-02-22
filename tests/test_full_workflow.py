from pathlib import Path
import shutil

from benchmark.full_workflow import run_full_workflow


def test_full_workflow_fast_mode_generates_artifacts(monkeypatch):
    run_id = "testrun_001"
    monkeypatch.setenv("BENCHMARK_CONFIG", "benchmark/benchmark_full.json")
    monkeypatch.setenv("PROFILE", "fast")
    monkeypatch.setenv("RUN_ID", run_id)

    try:
        rc = run_full_workflow()
        assert rc == 0

        root = Path("artifacts") / run_id
        assert (root / "dataset.jsonl").read_text().strip()
        assert (root / "ground_truth.jsonl").read_text().strip()
        assert (root / "metrics.json").read_text().strip()
        assert (root / "metrics" / "advanced_analysis.json").read_text().strip()
        assert (root / "report.md").read_text().strip()
        assert (root / "report.html").read_text().strip()
        assert (root / "predictions" / "full_framework" / "mappings.jsonl").read_text().strip()
    finally:
        for p in [Path("artifacts") / run_id, Path("results") / run_id]:
            if p.is_symlink() or p.is_file():
                p.unlink()
            elif p.exists():
                shutil.rmtree(p)
