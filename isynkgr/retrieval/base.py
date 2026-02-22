from __future__ import annotations

from typing import Protocol

from isynkgr.canonical.model import CanonicalModel
from isynkgr.canonical.schemas import EvidenceItem


class Retriever(Protocol):
    def retrieve(self, source: CanonicalModel, target_schema_hint: str) -> list[EvidenceItem]: ...
