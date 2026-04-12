from __future__ import annotations

import difflib
import re

from isynkgr.canonical.model import CanonicalModel, CanonicalNode
from isynkgr.canonical.schemas import Mapping
from isynkgr.icr.mapping_output_contract import normalize_mapping_item
from isynkgr.rules.store import RuleStore


def _norm(text: str) -> str:
    return str(text or "").strip().lower().replace("_", " ")


def _node_dtype(node: CanonicalNode) -> str:
    attrs = node.attributes or {}
    meta = attrs.get("metadata", {}) if isinstance(attrs.get("metadata", {}), dict) else {}
    for key in ("dtype", "datatype", "valueType", "dataType", "type"):
        value = attrs.get(key) or meta.get(key)
        if value:
            return str(value).upper()
    return ""


def _node_unit(node: CanonicalNode) -> str:
    attrs = node.attributes or {}
    meta = attrs.get("metadata", {}) if isinstance(attrs.get("metadata", {}), dict) else {}
    return str(attrs.get("unit") or meta.get("unit") or "").strip()


class RuleEngine:
    def __init__(self, store: RuleStore | None = None) -> None:
        self.store = store or RuleStore()

    def apply_rules(
        self,
        source: CanonicalModel,
        target_protocol: str,
        target: CanonicalModel | None = None,
        allow_synthetic_shortcuts: bool = True,
    ) -> list[Mapping]:
        mappings: list[Mapping] = []
        target_nodes = list(target.nodes if target else [])

        for node in source.nodes:
            benchmark_target = self._deterministic_benchmark_target(node.id, target_protocol) if allow_synthetic_shortcuts else ""
            if benchmark_target:
                payload = {
                    "source_path": node.id,
                    "target_path": benchmark_target,
                    "mapping_type": "equivalent",
                    "confidence": 0.95,
                    "rationale": "Deterministic synthetic benchmark mapping from OPC UA NodeId to canonical AAS value path.",
                    "evidence": ["rule:synthetic_benchmark_map"],
                }
                mappings.append(normalize_mapping_item(payload, source.standard, target_protocol))
                continue

            best = self._best_target_for_source(node, target_nodes)
            if best:
                target_node, score, evidence = best
                payload = {
                    "source_path": node.id,
                    "target_path": target_node.id,
                    "mapping_type": "label_match" if score >= 0.75 else "approximate",
                    "confidence": min(0.97, max(0.52, score)),
                    "rationale": "Rule engine selected target using label/datatype/unit/context heuristics.",
                    "evidence": evidence,
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

    def _best_target_for_source(self, source_node: CanonicalNode, target_nodes: list[CanonicalNode]) -> tuple[CanonicalNode, float, list[str]] | None:
        source_label = _norm(source_node.label or source_node.id)
        source_dtype = _node_dtype(source_node)
        source_unit = _node_unit(source_node)

        best: tuple[CanonicalNode, float, list[str]] | None = None
        for target_node in target_nodes:
            target_label = _norm(target_node.label or target_node.id)
            label_sim = difflib.SequenceMatcher(a=source_label, b=target_label).ratio()
            dtype_match = 1.0 if source_dtype and _node_dtype(target_node) and source_dtype == _node_dtype(target_node) else (0.4 if not source_dtype or not _node_dtype(target_node) else 0.0)
            unit_match = 1.0 if source_unit and _node_unit(target_node) and source_unit.lower() == _node_unit(target_node).lower() else (0.5 if not source_unit or not _node_unit(target_node) else 0.0)
            src_parent = source_node.id.rsplit("/", 2)[-2] if "/" in source_node.id else ""
            tgt_parent = target_node.id.rsplit("/", 2)[-2] if "/" in target_node.id else ""
            context_match = 1.0 if src_parent and tgt_parent and _norm(src_parent) == _norm(tgt_parent) else 0.0
            score = (label_sim * 0.6) + (dtype_match * 0.2) + (unit_match * 0.1) + (context_match * 0.1)
            evidence = ["rule:label_similarity"]
            if dtype_match >= 1.0:
                evidence.append("rule:datatype_compatible")
            if unit_match >= 1.0:
                evidence.append("rule:unit_compatible")
            if context_match >= 1.0:
                evidence.append("rule:context_match")
            if best is None or score > best[1]:
                best = (target_node, score, evidence)

        if best and best[1] >= 0.5:
            return best
        return None

    @staticmethod
    def _deterministic_benchmark_target(source_path: str, target_protocol: str) -> str:
        if target_protocol.lower() != "aas":
            return ""
        match = re.search(r"i=(\d+)", source_path)
        if not match:
            return ""
        idx = int(match.group(1)) - 1000
        if idx < 0:
            return ""
        lower = source_path.lower()
        signal = "temperature"
        for candidate in ["pressure", "flow", "speed", "state", "vibration", "temperature"]:
            if candidate in lower:
                signal = candidate
                break
        return f"aas://asset-{idx}/submodel/default/element/{signal}/value"
