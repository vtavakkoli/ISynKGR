from __future__ import annotations

import argparse
from pathlib import Path

from src.common import STANDARDS, seeded_rng, write_jsonl

DOMAIN_TOKENS = [
    "temperature", "pressure", "flow", "vibration", "energy", "status", "speed", "torque",
    "alarm", "maintenance", "batch", "quality", "setpoint", "sensor", "actuator", "controller",
]


def make_sample(standard_id: str, i: int, difficulty: str) -> dict:
    rng = seeded_rng(standard_id, str(i), difficulty)
    entity = f"{standard_id}_entity_{i:03d}"
    tokens = rng.sample(DOMAIN_TOKENS, 4)
    rel = rng.choice(["hasProperty", "controlledBy", "linkedTo", "requires"])
    constraints = [f"{tokens[0]} within {rng.randint(1, 20)}..{rng.randint(21, 100)}"]
    if difficulty in {"medium", "hard"}:
        constraints.append(f"alias:{tokens[1]}::{tokens[1]}_{standard_id}")
    if difficulty == "hard":
        constraints.append(f"multi-hop:{tokens[2]}->{tokens[3]}->{tokens[0]}")
    return {
        "sample_id": f"{standard_id}_{i:03d}",
        "standard": standard_id,
        "terms": tokens,
        "entities": [{"id": entity, "type": "AssetComponent"}],
        "properties": [{"name": t, "datatype": rng.choice(["float", "int", "string", "bool"])} for t in tokens[:3]],
        "relationships": [{"source": entity, "predicate": rel, "target": f"{standard_id}_node_{i:03d}"}],
        "constraints": constraints,
        "difficulty": difficulty,
    }


def generate(out_dir: Path, count: int = 100) -> None:
    tiers = ["easy", "medium", "hard"]
    for sid in STANDARDS:
        rows = [make_sample(sid, i, tiers[i % 3]) for i in range(1, count + 1)]
        write_jsonl(out_dir / sid / "samples_100.jsonl", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/samples"))
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    generate(args.out_dir, args.count)
    print(f"Generated samples for {len(STANDARDS)} standards at {args.out_dir}")
