from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class MappingTransform(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: MappingTransformOp
    args: dict[str, Any] = Field(default_factory=dict)


def normalize_mapping_path(path: str) -> str:
    value = (path or "").strip().replace("\\", "/")
    value = re.sub(r"/+", "/", value)
    if value.startswith("./"):
        value = value[2:]
    return value.rstrip("/")


class MappingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str
    target_path: str
    mapping_type: MappingType
    transform: MappingTransform | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=8, max_length=1000)
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_model(self) -> "MappingRecord":
        self.source_path = normalize_mapping_path(self.source_path)
        self.target_path = normalize_mapping_path(self.target_path)

        if self.mapping_type == MappingType.TRANSFORM and self.transform is None:
            raise ValueError("transform is required when mapping_type='transform'")
        if self.mapping_type != MappingType.TRANSFORM and self.transform is not None:
            raise ValueError("transform must be null unless mapping_type='transform'")
        return self


LEGACY_FIELD_MAP = {
    "source_id": "source_path",
    "target_id": "target_path",
    "relation_type": "mapping_type",
}


def ingest_mapping_payload(payload: dict[str, Any], migrate_legacy: bool = False) -> MappingRecord:
    has_legacy = any(k in payload for k in LEGACY_FIELD_MAP)
    if has_legacy and not migrate_legacy:
        legacy_keys = [k for k in LEGACY_FIELD_MAP if k in payload]
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

    return MappingRecord.model_validate(normalized)
