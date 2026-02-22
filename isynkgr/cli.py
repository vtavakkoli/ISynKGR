from __future__ import annotations

from pathlib import Path

from isynkgr.evaluation.benchmark_runner import main as run_bench_main
from isynkgr.evaluation.data_gen.generate_ground_truth import generate as gen_gt
from isynkgr.evaluation.data_gen.generate_samples import generate as gen_samples
from isynkgr.evaluation.data_gen.validate_data import main as validate


def gen_samples_main() -> None:
    samples = Path("data/samples")
    gt = Path("data/ground_truth")
    gen_samples(samples, 100)
    gen_gt(samples, gt)
    validate(samples, gt)


def run_bench_cli() -> None:
    run_bench_main()
