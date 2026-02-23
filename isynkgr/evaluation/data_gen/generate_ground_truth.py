from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

from isynkgr.common import STANDARDS, read_jsonl, write_jsonl


def map_row(row: dict, source: str, target: str) -> dict:
    source_entity = row["entities"][0]["id"]
    target_entity = source_entity.replace(source, target)
    return {
        "sample_id": row["sample_id"],
        "source_standard": source,
        "target_standard": target,
        "source_path": source_entity,
        "target_path": target_entity,
        "mapping_type": "equivalent",
        "transform": None,
        "confidence": 1.0,
        "rationale": "Deterministic synthetic GT mapping.",
        "evidence": ["generator:deterministic"],
    }


def validate(gt_rows: list[dict]) -> None:
    ids = {(r["sample_id"], r["source_standard"], r["target_standard"], r["source_path"], r["target_path"]) for r in gt_rows}
    if len(ids) != len(gt_rows):
        raise ValueError("Duplicate GT rows")


def generate(samples_dir: Path, out_dir: Path) -> None:
    for source, target in permutations(STANDARDS.keys(), 2):
        source_rows = read_jsonl(samples_dir / source / "samples_100.jsonl")
        gt_rows = [map_row(r, source, target) for r in source_rows]
        validate(gt_rows)
        write_jsonl(out_dir / f"{source}__to__{target}" / "gt.jsonl", gt_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, default=Path("data/samples"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/ground_truth"))
    args = parser.parse_args()
    generate(args.samples_dir, args.out_dir)
    print("Generated deterministic ground truth mappings")
