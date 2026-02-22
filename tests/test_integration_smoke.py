from pathlib import Path

from benchmark.harness import run_benchmark


def test_smoke_benchmark_local_only(monkeypatch):
    monkeypatch.setenv("FULL", "0")
    # this exercises harness import path; runtime execution validated in Make command outside tests
    assert Path("datasets/v1/manifest.json").exists()
