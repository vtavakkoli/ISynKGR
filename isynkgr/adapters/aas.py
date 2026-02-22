from __future__ import annotations

import json
import zipfile
from io import BytesIO
from typing import Any

from isynkgr.canonical.model import CanonicalEdge, CanonicalModel, CanonicalNode
from isynkgr.canonical.schemas import ValidationReport, ValidationViolation


class AASAdapter:
    name = "aas"

    def _load(self, raw: str | bytes | dict[str, Any]) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, bytes):
            if raw[:2] == b"PK":
                with zipfile.ZipFile(BytesIO(raw)) as zf:
                    candidate = next((n for n in zf.namelist() if n.endswith(".json")), None)
                    if not candidate:
                        raise ValueError("AASX contains no JSON")
                    return json.loads(zf.read(candidate).decode())
            return json.loads(raw.decode())
        return json.loads(raw)

    def parse(self, raw: str | bytes | dict[str, Any]) -> CanonicalModel:
        doc = self._load(raw)
        m = CanonicalModel(standard=self.name)
        for aas in doc.get("assetAdministrationShells", []):
            aid = aas.get("id", "")
            if aid:
                m.nodes.append(CanonicalNode(id=aid, type="AssetAdministrationShell", label=aas.get("idShort")))
            for ref in aas.get("submodels", []):
                sid = ref.get("keys", [{}])[-1].get("value")
                if aid and sid:
                    m.edges.append(CanonicalEdge(source=aid, target=sid, relation="hasSubmodel"))
        for sm in doc.get("submodels", []):
            sid = sm.get("id", "")
            m.nodes.append(CanonicalNode(id=sid, type="Submodel", label=sm.get("idShort")))
            for elem in sm.get("submodelElements", []):
                eid = f"{sid}:{elem.get('idShort','element')}"
                m.nodes.append(CanonicalNode(id=eid, type=elem.get("modelType", "Property"), label=elem.get("idShort"), attributes={"valueType": elem.get("valueType"), "value": elem.get("value")}))
                m.edges.append(CanonicalEdge(source=sid, target=eid, relation="hasElement"))
        return m

    def serialize(self, model: CanonicalModel, mappings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        shells = [n for n in model.nodes if n.type == "AssetAdministrationShell"]
        submodels = [n for n in model.nodes if n.type == "Submodel"]
        return {"assetAdministrationShells": [{"id": s.id, "idShort": s.label or s.id} for s in shells], "submodels": [{"id": sm.id, "idShort": sm.label or sm.id, "submodelElements": []} for sm in submodels], "mappings": mappings or []}

    def validate(self, raw: str | bytes | dict[str, Any]) -> ValidationReport:
        violations: list[ValidationViolation] = []
        try:
            doc = self._load(raw)
        except Exception as exc:
            return ValidationReport(valid=False, violations=[ValidationViolation(type="json", message=str(exc))])
        sm_ids = {sm.get("id") for sm in doc.get("submodels", []) if sm.get("id")}
        for aas in doc.get("assetAdministrationShells", []):
            if "id" not in aas:
                violations.append(ValidationViolation(type="required", message="AAS.id missing"))
            for ref in aas.get("submodels", []):
                key = (ref.get("keys") or [{}])[-1].get("value")
                if key and key not in sm_ids:
                    violations.append(ValidationViolation(type="integrity", message=f"Submodel reference missing: {key}"))
        for sm in doc.get("submodels", []):
            sid = sm.get("semanticId")
            if sid is not None and not isinstance(sid, dict):
                violations.append(ValidationViolation(type="semanticId", message=f"Submodel semanticId invalid: {sm.get('id')}"))
        return ValidationReport(valid=not violations, violations=violations)
