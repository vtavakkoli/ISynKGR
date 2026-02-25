from __future__ import annotations

from isynkgr.canonical.model import CanonicalModel
from isynkgr.canonical.schemas import Mapping
from isynkgr.icr.mapping_output_contract import normalize_mapping_item
from isynkgr.rules.store import RuleStore


class RuleEngine:
    def __init__(self, store: RuleStore | None = None) -> None:
        self.store = store or RuleStore()

    def apply_rules(
        self,
        source: CanonicalModel,
        target_protocol: str,
        target: CanonicalModel | None = None,
    ) -> list[Mapping]:
        mappings: list[Mapping] = []
        target_nodes = {n.label: n for n in (target.nodes if target else []) if n.label}

        for node in source.nodes:
            if node.label and node.label in target_nodes:
                target_node = target_nodes[node.label]
                payload = {
                    "source_path": node.id,
                    "target_path": target_node.id,
                    "mapping_type": "label_match",
                    "confidence": 0.9,
                    "rationale": "Rule-based label match.",
                    "evidence": ["rule:label_match"],
                }
            else:
                payload = {
                    "source_path": node.id,
                    "target_path": "",
                    "mapping_type": "no_match",
                    "confidence": 0.0,
                    "rationale": "Rule engine found no deterministic target match.",
                    "evidence": ["rule:no_match"],
                }
            mappings.append(normalize_mapping_item(payload, source.standard, target_protocol))
        return mappings
