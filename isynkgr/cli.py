from __future__ import annotations

from pathlib import Path

from benchmark.data_gen.pipeline import generate_pipeline
from benchmark.orchestrate import main as run_bench_main


def gen_samples_main() -> None:
    generate_pipeline(Path("benchmark/data_gen/out"), sample_size=100)


def run_bench_cli() -> None:
    run_bench_main()
