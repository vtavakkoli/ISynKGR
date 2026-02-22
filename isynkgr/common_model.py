from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class SimpleModel:
    def model_dump(self):
        return asdict(self)

    def model_dump_json(self, indent: int | None = None):
        return json.dumps(self.model_dump(), indent=indent)
