from __future__ import annotations

from isynkgr.canonical.model import CanonicalModel
from isynkgr.canonical.schemas import Mapping
from isynkgr.icr.entities import build_endpoint_path, normalize_path
from isynkgr.rules.store import RuleStore


class RuleEngine:
    def __init__(self, store: RuleStore | None = None) -> None:
        self.store = store or RuleStore()

    def _to_path(self, standard: str, value: str) -> str:
        if "://" in value:
            return normalize_path(value)
        return build_endpoint_path(standard, value)

    def apply_rules(self, source: CanonicalModel, target: CanonicalModel | None = None) -> list[Mapping]:
        mappings: list[Mapping] = []
        target_nodes = {n.label: n for n in (target.nodes if target else [])}
        for node in source.nodes:
            if node.label and node.label in target_nodes:
                mappings.append(
                    Mapping(
                        source_path=self._to_path(source.standard, node.id),
                        target_path=self._to_path(target.standard if target else source.standard, target_nodes[node.label].id),
                        mapping_type="label_match",
                        confidence=0.9,
                        rationale="Rule-based label match.",
                        evidence=["rule:label_match"],
                    )
                )
        return mappings
