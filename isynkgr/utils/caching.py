from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonCache:
    def __init__(self, root: str = "cache/llm") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict[str, Any] | None:
        p = self.root / f"{key}.json"
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def set(self, key: str, value: dict[str, Any]) -> None:
        (self.root / f"{key}.json").write_text(json.dumps(value, indent=2, sort_keys=True))
