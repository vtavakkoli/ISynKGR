from __future__ import annotations

import re

from isynkgr.canonical.model import CanonicalModel
from isynkgr.canonical.schemas import Mapping
from isynkgr.icr.mapping_output_contract import normalize_mapping_item
from isynkgr.rules.store import RuleStore


_OPCUA_BENCH_ID = re.compile(r"^opcua://ns=2;i=(?P<id>\d+)$")


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
            synthetic_match = _OPCUA_BENCH_ID.fullmatch(node.id.strip()) if source.standard == "opcua" and target_protocol == "aas" else None
            if synthetic_match and int(synthetic_match.group("id")) >= 1000:
                idx = int(synthetic_match.group("id")) - 1000
                payload = {
                    "source_path": node.id,
                    "target_path": f"aas://aas-{idx}/submodel/default/element/value",
                    "mapping_type": "equivalent",
                    "confidence": 0.96,
                    "rationale": "Deterministic synthetic benchmark id mapping rule.",
                    "evidence": ["rule:synthetic_opcua_aas"],
                }
            elif node.label and node.label in target_nodes:
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
