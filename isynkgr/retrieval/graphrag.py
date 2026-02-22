from __future__ import annotations

from isynkgr.canonical.model import CanonicalModel
from isynkgr.canonical.schemas import EvidenceItem


class GraphRAGRetriever:
    def retrieve(self, source: CanonicalModel, target_schema_hint: str) -> list[EvidenceItem]:
        out = []
        for n in source.nodes[:20]:
            out.append(EvidenceItem(id=f"node:{n.id}", kind="node", text=f"{n.type}:{n.label or n.id}", score=0.5))
        return out
