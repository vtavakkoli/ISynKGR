from __future__ import annotations

from isynkgr.canonical.model import CanonicalModel
from isynkgr.canonical.schemas import EvidenceItem
from isynkgr.icr.entities import build_endpoint_path, normalize_path


class GraphRAGRetriever:
    def retrieve(self, source: CanonicalModel, target_schema_hint: str) -> list[EvidenceItem]:
        out = []
        for n in source.nodes[:20]:
            node_path = normalize_path(n.id if "://" in n.id else build_endpoint_path(source.standard, n.id))
            out.append(EvidenceItem(id=f"node:{node_path}", kind="node", text=f"{n.type}:{n.label or n.id}", score=0.5))
        return out
