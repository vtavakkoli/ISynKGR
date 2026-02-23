from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path("datasets/v1/crosswalk")
    root.mkdir(parents=True, exist_ok=True)
    gt = root / "gt_mappings.jsonl"
    lines = []
    for i in range(100):
        lines.append(
            json.dumps(
                {
                    "source_path": f"ns=2;i={1000+i}",
                    "target_path": f"aas-{i}",
                    "mapping_type": "equivalent",
                    "transform": None,
                    "confidence": 1.0,
                    "rationale": "Synthetic deterministic ground-truth mapping.",
                    "evidence": ["generator:gen_crosswalk"],
                }
            )
        )
    gt.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
