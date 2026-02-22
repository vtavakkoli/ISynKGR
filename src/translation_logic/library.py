from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TranslationLogicLibrary:
    path: Path = Path("output/translation_logic/library.json")
    version: str = "1.0.0"

    def save_rule(
        self,
        source_standard: str,
        target_standard: str,
        source_entity: str,
        target_entity: str,
        confidence: float,
        provenance: dict[str, Any],
    ) -> None:
        db = self._read()
        db.setdefault("version", self.version)
        db.setdefault("rules", [])
        db["rules"].append(
            {
                "source_standard": source_standard,
                "target_standard": target_standard,
                "source_entity": source_entity,
                "target_entity": target_entity,
                "confidence": round(confidence, 4),
                "provenance": {
                    **provenance,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(db, indent=2, sort_keys=True))

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text())
