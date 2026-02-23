from __future__ import annotations

from isynkgr.canonical.model import CanonicalModel
from isynkgr.canonical.schemas import EvidenceItem
from isynkgr.icr.entities import build_endpoint_path, normalize_path


class SqliteFTSRetriever:
    def retrieve(self, source: CanonicalModel, target_schema_hint: str) -> list[EvidenceItem]:
        base_path = normalize_path(build_endpoint_path(source.standard, "hint"))
        return [EvidenceItem(id=f"fts:{base_path}", kind="text", text=target_schema_hint, score=0.1)]
