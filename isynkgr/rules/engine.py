from __future__ import annotations

from isynkgr.canonical.model import CanonicalModel
from isynkgr.canonical.schemas import Mapping
from isynkgr.rules.store import RuleStore


class RuleEngine:
    def __init__(self, store: RuleStore | None = None) -> None:
        self.store = store or RuleStore()

    def apply_rules(self, source: CanonicalModel, target: CanonicalModel | None = None) -> list[Mapping]:
        mappings: list[Mapping] = []
        target_nodes = {n.label: n for n in (target.nodes if target else [])}
        for node in source.nodes:
            if node.label and node.label in target_nodes:
                mappings.append(Mapping(source_id=node.id, target_id=target_nodes[node.label].id, relation_type="label_match", confidence=0.9, evidence_ids=["rule:label_match"]))
        return mappings
