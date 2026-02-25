from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from isynkgr.icr.entities import build_endpoint_path
from isynkgr.icr.mapping_schema import MappingRecord, MappingType
from isynkgr.icr.entities import normalize_path
from isynkgr.icr.path_validation import detect_path_protocol, validate_protocol_path

_ALLOWED_MAPPING_TYPES = {m.value for m in MappingType}


@dataclass
class MappingContractError:
    method: str
    reason: str
    item: dict[str, Any]
    source_protocol: str
    target_protocol: str
    exception: str | None = None

    def model_dump(self) -> dict[str, Any]:
        payload = {
            "method": self.method,
            "reason": self.reason,
            "item": self.item,
            "source_protocol": self.source_protocol,
            "target_protocol": self.target_protocol,
        }
        if self.exception:
            payload["exception"] = self.exception
        return payload


@dataclass
class MappingNormalizationReport:
    accepted: list[MappingRecord] = field(default_factory=list)
    rejected: list[MappingContractError] = field(default_factory=list)


def _coerce_confidence(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _protocol_path_or_build(path: str, protocol: str, fallback_id: str) -> str:
    raw = normalize_path(path or "")
    if not raw:
        return build_endpoint_path(protocol, fallback_id)
    if "://" not in raw:
        return build_endpoint_path(protocol, raw)
    return raw


def normalize_mapping_item(item: dict[str, Any], source_protocol: str, target_protocol: str) -> MappingRecord:
    payload = dict(item or {})
    source_protocol = source_protocol.lower().strip()
    target_protocol = target_protocol.lower().strip()

    mapping_type = str(payload.get("mapping_type") or "approximate").lower().strip()
    if mapping_type not in _ALLOWED_MAPPING_TYPES:
        mapping_type = "approximate"

    source_path = _protocol_path_or_build(
        str(payload.get("source_path") or payload.get("source_id") or ""),
        source_protocol,
        "unknown_source",
    )

    if mapping_type == MappingType.NO_MATCH.value:
        target_path = ""
    else:
        target_path = _protocol_path_or_build(
            str(payload.get("target_path") or payload.get("target_id") or ""),
            target_protocol,
            "unknown_target",
        )
        target_proto = detect_path_protocol(target_path)
        if target_proto != target_protocol:
            target_path = build_endpoint_path(target_protocol, target_path.split("://", 1)[-1])

    transform = payload.get("transform")
    if mapping_type != MappingType.TRANSFORM.value:
        transform = None
    elif transform is None:
        transform = {"op": "identity", "args": {}}

    rationale = str(payload.get("rationale") or "")
    if len(rationale) < 8:
        rationale = (rationale + " Auto-normalized mapping rationale.").strip()
    rationale = rationale[:1000]

    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        evidence = []

    normalized_payload = {
        "source_path": source_path,
        "target_path": target_path,
        "mapping_type": mapping_type,
        "transform": transform,
        "confidence": _coerce_confidence(payload.get("confidence", 0.0)),
        "rationale": rationale,
        "evidence": [str(x) for x in evidence],
    }
    return MappingRecord.model_validate(normalized_payload)


def validate_mapping_item(item: dict[str, Any], source_protocol: str, target_protocol: str) -> tuple[bool, str]:
    try:
        mapping = normalize_mapping_item(item, source_protocol=source_protocol, target_protocol=target_protocol)
    except Exception as exc:
        return False, str(exc)

    src_proto = detect_path_protocol(mapping.source_path)
    if src_proto != source_protocol.lower().strip():
        return False, f"source_path protocol mismatch: expected '{source_protocol}', got '{src_proto}'"

    if mapping.mapping_type == MappingType.NO_MATCH:
        if mapping.target_path:
            return False, "target_path must be empty for no_match"
        return True, ""

    tgt_proto = detect_path_protocol(mapping.target_path)
    if tgt_proto != target_protocol.lower().strip():
        return False, f"target_path protocol mismatch: expected '{target_protocol}', got '{tgt_proto}'"
    try:
        validate_protocol_path(mapping.source_path, "source_path")
        validate_protocol_path(mapping.target_path, "target_path")
    except Exception as exc:
        return False, str(exc)
    return True, ""


def normalize_mapping_items(
    items: list[dict[str, Any]],
    source_protocol: str,
    target_protocol: str,
    method: str,
) -> MappingNormalizationReport:
    report = MappingNormalizationReport()
    for raw_item in items:
        try:
            normalized = normalize_mapping_item(raw_item, source_protocol=source_protocol, target_protocol=target_protocol)
            ok, error = validate_mapping_item(normalized.model_dump(), source_protocol=source_protocol, target_protocol=target_protocol)
            if not ok:
                report.rejected.append(
                    MappingContractError(
                        method=method,
                        reason="validation_failed",
                        item=raw_item,
                        source_protocol=source_protocol,
                        target_protocol=target_protocol,
                        exception=error,
                    )
                )
                continue
            report.accepted.append(normalized)
        except Exception as exc:
            report.rejected.append(
                MappingContractError(
                    method=method,
                    reason="normalization_failed",
                    item=raw_item,
                    source_protocol=source_protocol,
                    target_protocol=target_protocol,
                    exception=str(exc),
                )
            )
    return report
