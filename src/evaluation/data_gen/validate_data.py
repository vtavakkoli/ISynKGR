from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

from src.common import STANDARDS, read_jsonl


def main(samples_dir: Path, gt_dir: Path) -> None:
    for sid in STANDARDS:
        rows = read_jsonl(samples_dir / sid / "samples_100.jsonl")
        assert len(rows) == 100, f"{sid}: expected 100 rows"
        assert len({r['sample_id'] for r in rows}) == 100, f"{sid}: duplicate ids"
    for s, t in permutations(STANDARDS.keys(), 2):
        rows = read_jsonl(gt_dir / f"{s}__to__{t}" / "gt.jsonl")
        assert len(rows) == 100, f"{s}->{t}: expected 100 rows"
    print("Validation successful")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples-dir", type=Path, default=Path("data/samples"))
    p.add_argument("--gt-dir", type=Path, default=Path("data/ground_truth"))
    args = p.parse_args()
    main(args.samples_dir, args.gt_dir)
