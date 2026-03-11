from __future__ import annotations

from isynkgr.canonical.model import CanonicalModel
from isynkgr.canonical.schemas import EvidenceItem
from isynkgr.icr.entities import build_endpoint_path, normalize_path


class GraphRAGRetriever:
    def retrieve(self, source: CanonicalModel, target_schema_hint: str) -> list[EvidenceItem]:
        scored: list[EvidenceItem] = []
        for n in source.nodes:
            node_path = normalize_path(n.id if "://" in n.id else build_endpoint_path(source.standard, n.id))
            lexical = (n.label or n.id or "").lower()
            score = 0.2
            if "temp" in lexical or "temperature" in lexical:
                score += 0.5
            if target_schema_hint.lower() in {"aas", "opcua"}:
                target_hint = f"{target_schema_hint.lower()}://candidate/{(n.label or n.id).replace(' ', '_')}"
                score += 0.2
            else:
                target_hint = ""
            scored.append(
                EvidenceItem(
                    id=f"node:{node_path}",
                    kind="node",
                    text=f"{n.type}:{n.label or n.id}",
                    score=min(score, 1.0),
                    payload={"source_node": node_path, "target_hint": target_hint},
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:20]
