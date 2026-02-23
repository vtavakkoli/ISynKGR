from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from benchmark.data_gen.pipeline import generate_pipeline


def generate(out_dir: Path, count: int = 100) -> None:
    warnings.warn(
        "isynkgr.evaluation.data_gen.generate_samples is deprecated; use benchmark.data_gen.pipeline.",
        DeprecationWarning,
        stacklevel=2,
    )
    generate_pipeline(out_dir, sample_size=count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark/data_gen/out"))
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    generate(args.out_dir, args.count)
    print(f"Generated benchmark data at {args.out_dir}")
