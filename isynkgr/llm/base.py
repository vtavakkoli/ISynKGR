from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    model: str

    def complete_json(self, prompt: str, schema_name: str, seed: int) -> dict: ...
