from __future__ import annotations

from isynkgr.canonical.model import CanonicalModel
from isynkgr.canonical.schemas import EvidenceItem
from isynkgr.icr.entities import build_endpoint_path, normalize_path


def _candidate_pool(target_schema_hint: str) -> list[str]:
    hint = target_schema_hint.lower()
    if hint == "aas":
        return [
            "aas://asset/submodel/default/element/temperature/value",
            "aas://asset/submodel/default/element/pressure/value",
            "aas://asset/submodel/default/element/flow/value",
            "aas://asset/submodel/default/element/state/value",
            "aas://asset/submodel/default/element/value",
        ]
    if hint == "iec61499":
        return ["iec61499://device/res/fb/out_temp", "iec61499://device/res/fb/out_value"]
    return [f"{hint}://candidate/default"]


class GraphRAGRetriever:
    def retrieve(self, source: CanonicalModel, target_schema_hint: str) -> list[EvidenceItem]:
        scored: list[EvidenceItem] = []
        pool = _candidate_pool(target_schema_hint)
        for n in source.nodes:
            node_path = normalize_path(n.id if "://" in n.id else build_endpoint_path(source.standard, n.id))
            lexical = (n.label or n.id or "").lower()
            score = 0.2
            if "temp" in lexical or "temperature" in lexical:
                score += 0.5
            label = lexical.replace(" ", "_")
            for idx, candidate in enumerate(pool):
                boost = 0.0
                if label and label in candidate.lower():
                    boost += 0.4
                if "value" in candidate.lower():
                    boost += 0.05
                cand_score = min(score + boost - (idx * 0.03), 1.0)
                scored.append(
                    EvidenceItem(
                        id=f"node:{node_path}:cand:{idx}",
                        kind="target_candidate",
                        text=f"{n.type}:{n.label or n.id}",
                        score=cand_score,
                        payload={"source_node": node_path, "target_hint": candidate, "candidate_path": candidate},
                    )
                )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:20]
