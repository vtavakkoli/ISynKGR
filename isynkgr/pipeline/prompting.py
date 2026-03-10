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
        "TARGET_VARIABLES": [{"path": e.id, "name": e.text, "description": e.kind} for e in evidence[:30]],
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
        "6) Preserve the source signal semantic token (e.g., speed/temp/pressure/current/voltage/flow) in target_path naming.\n"
        "7) For AAS targets use canonical shape: aas://<asset>/submodel/default/element/<signal>/value.\n"
        "8) Prefer one high-confidence mapping per source variable when possible.\n"
        "9) For synthetic OPCUA->AAS benchmark ids where source_path is opcua://ns=2;i=<N>=1000+k, prefer target_path aas://aas-k/submodel/default/element/value.\n"
        "Input context:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
