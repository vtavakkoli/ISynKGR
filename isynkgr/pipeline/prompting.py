from __future__ import annotations

import json
from typing import Any

from isynkgr.canonical.model import CanonicalModel
from isynkgr.canonical.schemas import EvidenceItem


def _node_summary(model: CanonicalModel, max_items: int = 30) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in model.nodes[:max_items]:
        rows.append(
            {
                "id": node.id,
                "path": node.id,
                "name": node.label,
                "datatype": node.attributes.get("datatype"),
                "unit": node.attributes.get("unit"),
                "description": node.attributes.get("description"),
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
            exact.append({"path": candidate_path, "name": item.text, "description": item.kind, "exact_candidate": True})
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
    return (
        "You are an industrial protocol mapping assistant.\n"
        "Return JSON only and no markdown, comments, XML tags, or prose.\n"
        "Do not output hidden reasoning or thinking. Put only concise rationale/evidence text in final JSON.\n"
        "Return exactly one top-level JSON object and nothing else.\n"
        "The response MUST match this contract exactly:\n"
        f"{json.dumps(contract, ensure_ascii=False)}\n"
        "Rules:\n"
        "1) mapping_type must be one of equivalent, approximate, label_match, transform, no_match.\n"
        "2) transform must be null unless mapping_type == 'transform'.\n"
        f"3) source_path must start with '{source_protocol.lower()}://'.\n"
        f"4) target_path must start with '{target_protocol.lower()}://' or be empty string only when mapping_type == 'no_match'.\n"
        "5) confidence must be numeric between 0 and 1.\n"
        "6) Choose target_path exactly from TARGET_VARIABLES when possible.\n"
        "7) Do not invent a new target_path when an exact TARGET_VARIABLES candidate applies.\n"
        "8) Prefer one high-confidence mapping per source variable when possible.\n"
        "Input context:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
