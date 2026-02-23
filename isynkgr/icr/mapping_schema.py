from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from isynkgr.icr.entities import normalize_path
from isynkgr.icr.path_validation import validate_protocol_path


class MappingType(str, Enum):
    EQUIVALENT = "equivalent"
    APPROXIMATE = "approximate"
    FALLBACK = "fallback"
    LABEL_MATCH = "label_match"
    TRANSFORM = "transform"


class MappingTransformOp(str, Enum):
    IDENTITY = "identity"
    CONCAT = "concat"
    CAST = "cast"
    FORMAT = "format"
    REGEX_EXTRACT = "regex_extract"


def normalize_mapping_path(path: str) -> str:
    return normalize_path(path)



@dataclass
class MappingTransform:
    op: MappingTransformOp
    args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | "MappingTransform") -> "MappingTransform":
        if isinstance(payload, MappingTransform):
            return payload
        if not isinstance(payload, dict):
            raise ValueError("transform must be an object")
        op = payload.get("op")
        if op is None:
            raise ValueError("transform.op is required")
        try:
            op_enum = MappingTransformOp(str(op))
        except ValueError as exc:
            raise ValueError(f"transform.op must be one of {[e.value for e in MappingTransformOp]}") from exc
        args = payload.get("args", {})
        if not isinstance(args, dict):
            raise ValueError("transform.args must be an object")
        return cls(op=op_enum, args=args)

    def model_dump(self) -> dict[str, Any]:
        return {"op": self.op.value, "args": self.args}


@dataclass
class MappingRecord:
    source_path: str
    target_path: str
    mapping_type: MappingType
    transform: MappingTransform | None = None
    confidence: float = 1.0
    rationale: str = ""
    evidence: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.source_path = normalize_mapping_path(self.source_path)
        self.target_path = normalize_mapping_path(self.target_path)

        if not self.source_path:
            raise ValueError("source_path is required")
        if not self.target_path:
            raise ValueError("target_path is required")

        validate_protocol_path(self.source_path, "source_path")
        validate_protocol_path(self.target_path, "target_path")

        if not isinstance(self.mapping_type, MappingType):
            self.mapping_type = MappingType(str(self.mapping_type))

        self.confidence = float(self.confidence)
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0 and 1")

        if not isinstance(self.rationale, str):
            raise ValueError("rationale must be a string")
        if not (8 <= len(self.rationale) <= 1000):
            raise ValueError("rationale must be 8..1000 chars")

        if not isinstance(self.evidence, list):
            raise ValueError("evidence must be a list")
        self.evidence = [str(x) for x in self.evidence]

        if self.mapping_type == MappingType.TRANSFORM:
            if self.transform is None:
                raise ValueError("transform is required when mapping_type='transform'")
            if not isinstance(self.transform, MappingTransform):
                self.transform = MappingTransform.from_payload(self.transform)
        elif self.transform is not None:
            raise ValueError("transform must be null unless mapping_type='transform'")

    @classmethod
    def model_validate(cls, payload: dict[str, Any] | "MappingRecord") -> "MappingRecord":
        if isinstance(payload, MappingRecord):
            return payload
        if not isinstance(payload, dict):
            raise ValueError("mapping payload must be an object")

        allowed = {"source_path", "target_path", "mapping_type", "transform", "confidence", "rationale", "evidence"}
        extras = sorted(set(payload) - allowed)
        if extras:
            raise ValueError(f"extra fields are not permitted: {extras}")

        return cls(
            source_path=payload.get("source_path", ""),
            target_path=payload.get("target_path", ""),
            mapping_type=payload.get("mapping_type"),
            transform=payload.get("transform"),
            confidence=payload.get("confidence", 1.0),
            rationale=payload.get("rationale", ""),
            evidence=payload.get("evidence", []),
        )

    def model_dump(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "target_path": self.target_path,
            "mapping_type": self.mapping_type.value,
            "transform": self.transform.model_dump() if self.transform else None,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "evidence": self.evidence,
        }


LEGACY_FIELD_MAP = {
    "source_id": "source_path",
    "target_id": "target_path",
    "relation_type": "mapping_type",
    "relation": "mapping_type",
}


def ingest_mapping_payload(payload: dict[str, Any], migrate_legacy: bool = False) -> MappingRecord:
    has_legacy = any(k in payload for k in LEGACY_FIELD_MAP) or "evidence_ids" in payload
    if has_legacy and not migrate_legacy:
        legacy_keys = [k for k in [*LEGACY_FIELD_MAP, "evidence_ids"] if k in payload]
        raise ValueError(f"Legacy mapping keys are not accepted: {legacy_keys}")

    normalized = payload.copy()
    if has_legacy and migrate_legacy:
        for old_key, new_key in LEGACY_FIELD_MAP.items():
            if old_key not in normalized:
                continue
            if new_key in normalized:
                raise ValueError(f"Cannot migrate '{old_key}' because '{new_key}' already exists")
            normalized[new_key] = normalized.pop(old_key)

    if "evidence_ids" in normalized and "evidence" not in normalized:
        normalized["evidence"] = normalized.pop("evidence_ids")

    if migrate_legacy:
        normalized.setdefault("confidence", 1.0)
        normalized.setdefault("rationale", "Migrated legacy mapping record.")
        normalized.setdefault("evidence", [])

    return MappingRecord.model_validate(normalized)
