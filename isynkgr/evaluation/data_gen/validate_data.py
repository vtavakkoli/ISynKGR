from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from benchmark.data_gen.pipeline import generate_pipeline


def main(samples_dir: Path, gt_dir: Path) -> None:
    del samples_dir, gt_dir
    warnings.warn(
        "isynkgr.evaluation.data_gen.validate_data is deprecated; use benchmark.data_gen.pipeline validation.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Ensure canonical benchmark data/validation path is invoked.
    generate_pipeline(Path("benchmark/data_gen/out"))
    print("Validation successful")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples-dir", type=Path, default=Path("benchmark/data_gen/out"))
    p.add_argument("--gt-dir", type=Path, default=Path("benchmark/data_gen/out/ground_truth"))
    args = p.parse_args()
    main(args.samples_dir, args.gt_dir)
