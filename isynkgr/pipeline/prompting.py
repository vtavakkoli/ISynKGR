from __future__ import annotations

import json
from typing import Any

from isynkgr.canonical.model import CanonicalModel
from isynkgr.canonical.schemas import EvidenceItem


def _node_attr(node: Any, *keys: str) -> str:
    attrs = getattr(node, "attributes", {}) or {}
    metadata = attrs.get("metadata", {}) if isinstance(attrs.get("metadata", {}), dict) else {}
    for key in keys:
        value = attrs.get(key) or metadata.get(key)
        if value:
            return str(value)
    return ""


def _node_summary(model: CanonicalModel, max_items: int = 30) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in model.nodes[:max_items]:
        rows.append(
            {
                "id": node.id,
                "path": node.id,
                "name": node.label,
                "datatype": _node_attr(node, "datatype", "dtype", "valueType", "dataType", "type"),
                "unit": _node_attr(node, "unit"),
                "description": node.attributes.get("description") or _node_attr(node, "DisplayName"),
            }
        )
    return rows


def _target_summary(evidence: list[EvidenceItem], target_protocol: str, max_items: int = 30) -> list[dict[str, Any]]:
    prefix = f"{target_protocol.lower()}://"
    exact: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in evidence:
        payload = item.payload or {}
        candidate_path = str(payload.get("candidate_path") or payload.get("target_hint") or "").strip()
        if candidate_path.startswith(prefix):
            if candidate_path in seen:
                continue
            seen.add(candidate_path)
            exact.append(
                {
                    "path": candidate_path,
                    "name": item.text,
                    "description": item.kind,
                    "datatype": payload.get("datatype", ""),
                    "unit": payload.get("unit", ""),
                    "parent": payload.get("parent", ""),
                    "retrieval_score": item.score,
                    "exact_candidate": True,
                }
            )
            continue
        path = str(item.id)
        if path in seen:
            continue
        seen.add(path)
        fallback.append({"path": path, "name": item.text, "description": item.kind, "exact_candidate": False})
    rows = exact if exact else fallback
    return rows[:max_items]


def build_mapping_prompt(
    source_protocol: str,
    target_protocol: str,
    source_schema_summary: dict[str, Any],
    target_schema_summary: dict[str, Any],
    source_model: CanonicalModel,
    evidence: list[EvidenceItem],
    use_reasoning_prompt: bool = True,
) -> str:
    contract = {
        "mappings": [
            {
                "source_path": f"{source_protocol.lower()}://...",
                "target_path": f"{target_protocol.lower()}://... or '' when mapping_type=='no_match'",
                "mapping_type": "equivalent|approximate|label_match|transform|no_match",
                "transform": {"op": "identity|concat|cast|format|regex_extract", "args": {}},
                "confidence": 0.0,
                "rationale": "string (8..1000 chars)",
                "evidence": ["string"],
            }
        ]
    }
    payload = {
        "SOURCE_PROTOCOL": source_protocol,
        "TARGET_PROTOCOL": target_protocol,
        "SOURCE_SCHEMA": source_schema_summary,
        "TARGET_SCHEMA": target_schema_summary,
        "SOURCE_VARIABLES": _node_summary(source_model),
        "TARGET_VARIABLES": _target_summary(evidence, target_protocol),
    }
    base_prompt = (
        "You are an industrial protocol mapping assistant.\n"
        "Return JSON only and no markdown, comments, XML tags, or prose.\n"
        "Do not output hidden reasoning or thinking. Put only concise rationale/evidence text in final JSON.\n"
        "Return exactly one top-level JSON object and nothing else.\n"
        "CRITICAL: You MUST return exactly one mapping object in the mappings array for EVERY item in the SOURCE_VARIABLES list. "
        "The length of the output array must perfectly match the length of the input variables.\n"
        "The response MUST match this contract exactly:\n"
        f"{json.dumps(contract, ensure_ascii=False)}\n"
        "Few-shot guardrail example:\n"
        '{"SOURCE_VARIABLES":[{"path":"opcua://ns=2;s=Pump01","name":"Pump01","description":"equipment label only"}],'
        '"TARGET_VARIABLES":[{"path":"aas://asset-0/submodel/default/element/temperature/value","name":"temperature"}],'
        '"EXPECTED":{"mappings":[{"source_path":"opcua://ns=2;s=Pump01","target_path":"","mapping_type":"no_match","transform":null,"confidence":0.0,'
        '"rationale":"Pump01 is an equipment identifier, not a measurement variable, so there is no valid measurement mapping.",'
        '"evidence":["equipment_identifier_not_measurement"]}]}}\n'
        "Rules:\n"
        "1) mapping_type must be one of equivalent, approximate, label_match, transform, no_match.\n"
        "2) transform must be null unless mapping_type == 'transform'.\n"
        f"3) source_path must start with '{source_protocol.lower()}://'.\n"
        f"4) target_path must start with '{target_protocol.lower()}://' or be empty string only when mapping_type == 'no_match'.\n"
        "5) confidence must be numeric between 0 and 1.\n"
        "6) Choose target_path exactly from TARGET_VARIABLES when possible.\n"
        "7) Do not invent a new target_path when an exact TARGET_VARIABLES candidate applies.\n"
        "8) Prefer one high-confidence mapping per source variable when possible.\n"
        "9) Prefer semantic matches: variable meaning first, then datatype/unit/context compatibility.\n"
        "10) Avoid arbitrary index-based choices when candidates are generic; choose no_match if evidence is insufficient.\n"
        "11) Equipment labels (Pump, Motor, Line, Device) are not measurements by themselves.\n"
        "12) Do not map generic labels to pressure/temperature/current/etc. unless explicit evidence exists in source metadata.\n"
        "13) If multiple semantically-equivalent candidates remain unresolved by context, output mapping_type='no_match' with empty target_path.\n"
        "14) Never invent target paths outside provided target candidates.\n"
        "Input context:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    if not use_reasoning_prompt:
        return base_prompt
    return (
        f"{base_prompt}\n"
        "Before finalizing, do a short internal consistency check over source_path prefixes, target_path prefixes, and mapping_type/transform validity."
    )
