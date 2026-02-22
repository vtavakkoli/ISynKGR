from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from isynkgr.common_model import SimpleModel


@dataclass
class Rule(SimpleModel):
    rule_id: str
    condition: str
    action: str
    confidence: float = 1.0
    priority: int = 100
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_used: str | None = None
    sources: list[str] = field(default_factory=list)


@dataclass
class RuleStore(SimpleModel):
    version: str = "1.0"
    rules: list[Rule] = field(default_factory=list)

    def export_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, payload: str) -> "RuleStore":
        data = json.loads(payload)
        return cls(version=data.get("version", "1.0"), rules=[Rule(**r) for r in data.get("rules", [])])
