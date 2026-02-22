from __future__ import annotations

from typing import Any, Protocol

from isynkgr.canonical.model import CanonicalModel
from isynkgr.canonical.schemas import ValidationReport


class StandardAdapter(Protocol):
    name: str

    def parse(self, raw: str | bytes | dict[str, Any]) -> CanonicalModel: ...

    def serialize(self, model: CanonicalModel, mappings: list[dict[str, Any]] | None = None) -> dict[str, Any] | str: ...

    def validate(self, raw: str | bytes | dict[str, Any]) -> ValidationReport: ...
