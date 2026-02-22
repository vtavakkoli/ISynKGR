from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

SEED = 145162578
STANDARDS = {
    "ieee1451": "IEEE 1451",
    "iso15926": "ISO 15926",
    "iec61499": "IEC 61499",
    "opcua62541": "OPC UA (IEC 62541)",
    "aas63278": "Asset Administration Shell (IEC 63278)",
}


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def seeded_rng(*parts: str) -> random.Random:
    token = "::".join(parts)
    seed = int(hashlib.sha256(f"{SEED}:{token}".encode()).hexdigest()[:16], 16)
    return random.Random(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
