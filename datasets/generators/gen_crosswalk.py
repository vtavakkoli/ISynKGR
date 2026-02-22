from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path("datasets/v1/crosswalk")
    root.mkdir(parents=True, exist_ok=True)
    gt = root / "gt_mappings.jsonl"
    lines = []
    for i in range(100):
        lines.append(json.dumps({"source_id": f"ns=2;i={1000+i}", "target_id": f"aas-{i}", "relation": "equivalent", "confidence": 1.0}))
    gt.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
