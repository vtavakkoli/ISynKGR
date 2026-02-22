from __future__ import annotations

from isynkgr.canonical.model import CanonicalModel
from isynkgr.canonical.schemas import EvidenceItem


class SqliteFTSRetriever:
    def retrieve(self, source: CanonicalModel, target_schema_hint: str) -> list[EvidenceItem]:
        return [EvidenceItem(id="fts:hint", kind="text", text=target_schema_hint, score=0.1)]
