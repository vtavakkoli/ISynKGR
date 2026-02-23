from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from isynkgr.common_model import SimpleModel
from isynkgr.icr.mapping_schema import MappingRecord


Mapping = MappingRecord


@dataclass
class EvidenceItem(SimpleModel):
    id: str
    kind: str
    text: str
    score: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationViolation(SimpleModel):
    type: str
    message: str
    path: str | None = None


@dataclass
class ValidationReport(SimpleModel):
    valid: bool
    violations: list[ValidationViolation] = field(default_factory=list)


@dataclass
class Provenance(SimpleModel):
    model_name: str
    prompt_hash: str
    seed: int
    git_commit: str
    adapter_versions: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslationResult(SimpleModel):
    target_artifact: dict[str, Any] | str
    mappings: list[Mapping] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    provenance: Provenance | None = None
    validation_report: ValidationReport | None = None
    repair_iterations: int = 0
