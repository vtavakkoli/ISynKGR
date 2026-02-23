from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from benchmark.data_gen.pipeline import generate_pipeline


def generate(samples_dir: Path, out_dir: Path) -> None:
    del samples_dir
    warnings.warn(
        "isynkgr.evaluation.data_gen.generate_ground_truth is deprecated; use benchmark.data_gen.pipeline.",
        DeprecationWarning,
        stacklevel=2,
    )
    generate_pipeline(out_dir.parent if out_dir.name else out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, default=Path("benchmark/data_gen/out"))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark/data_gen/out/ground_truth"))
    args = parser.parse_args()
    generate(args.samples_dir, args.out_dir)
    print("Generated benchmark ground truth artifacts")
