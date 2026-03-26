from __future__ import annotations

import json
from typing import Any

from isynkgr.canonical.model import CanonicalEdge, CanonicalModel, CanonicalNode
from isynkgr.canonical.schemas import ValidationReport, ValidationViolation


class ISO15926Adapter:
    name = "iso15926"

    def _load(self, raw: str | bytes | dict[str, Any]) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, bytes):
            return json.loads(raw.decode())
        return json.loads(raw)

    def parse(self, raw: str | bytes | dict[str, Any]) -> CanonicalModel:
        doc = self._load(raw)
        model = CanonicalModel(standard=self.name)
        for cls in doc.get("classes", []):
            cid = str(cls.get("id", "")).strip()
            if not cid:
                continue
            model.nodes.append(CanonicalNode(id=f"iso15926://{cid}", type="ClassOfIndividual", label=cls.get("label"), attributes=cls))
        for rel in doc.get("relations", []):
            src = str(rel.get("source", "")).strip()
            dst = str(rel.get("target", "")).strip()
            if src and dst:
                model.edges.append(CanonicalEdge(source=f"iso15926://{src}", target=f"iso15926://{dst}", relation=str(rel.get("type", "relatedTo"))))
        return model

    def serialize(self, model: CanonicalModel, mappings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {"standard": self.name, "nodes": [n.model_dump() for n in model.nodes], "edges": [e.model_dump() for e in model.edges], "mappings": mappings or []}

    def validate(self, raw: str | bytes | dict[str, Any]) -> ValidationReport:
        try:
            doc = self._load(raw)
        except Exception as exc:
            return ValidationReport(valid=False, violations=[ValidationViolation(type="json", message=str(exc))])
        violations: list[ValidationViolation] = []
        if not isinstance(doc.get("classes", []), list):
            violations.append(ValidationViolation(type="required", message="classes must be a list"))
        if not isinstance(doc.get("relations", []), list):
            violations.append(ValidationViolation(type="required", message="relations must be a list"))
        return ValidationReport(valid=not violations, violations=violations)
