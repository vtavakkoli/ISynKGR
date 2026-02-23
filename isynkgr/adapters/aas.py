from __future__ import annotations

import json
import zipfile
from io import BytesIO
from typing import Any

from isynkgr.canonical.model import CanonicalEdge, CanonicalModel, CanonicalNode
from isynkgr.canonical.schemas import ValidationReport, ValidationViolation
from isynkgr.icr.entities import Asset, Relationship, Sensor, build_asset_path, build_sensor_path


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
                asset = Asset(id=aid, path=build_asset_path(self.name, aid), protocol=self.name, label=aas.get("idShort"), metadata={"raw_id": aid})
                m.nodes.append(CanonicalNode(id=asset.path, type="AssetAdministrationShell", label=asset.label, attributes=asset.model_dump()))
            for ref in aas.get("submodels", []):
                sid = ref.get("keys", [{}])[-1].get("value")
                if aid and sid:
                    rel = Relationship(source_path=build_asset_path(self.name, aid), target_path=build_sensor_path(self.name, aid, sid), relation="hasSubmodel")
                    m.edges.append(CanonicalEdge(source=rel.source_path, target=rel.target_path, relation=rel.relation))
        for sm in doc.get("submodels", []):
            sid = sm.get("id", "")
            sm_path = build_sensor_path(self.name, sid or "default", sid or "submodel")
            m.nodes.append(CanonicalNode(id=sm_path, type="Submodel", label=sm.get("idShort"), attributes={"raw_id": sid}))
            for elem in sm.get("submodelElements", []):
                eid = elem.get("idShort", "element")
                sensor = Sensor(id=eid, path=build_sensor_path(self.name, sid or "default", eid), protocol=self.name, label=eid, metadata={"valueType": elem.get("valueType"), "value": elem.get("value"), "submodel_id": sid})
                m.nodes.append(CanonicalNode(id=sensor.path, type=elem.get("modelType", "Property"), label=sensor.label, attributes=sensor.model_dump()))
                m.edges.append(CanonicalEdge(source=sm_path, target=sensor.path, relation="hasElement"))
        return m

    def serialize(self, model: CanonicalModel, mappings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        shells = [n for n in model.nodes if n.type == "AssetAdministrationShell"]
        submodels = [n for n in model.nodes if n.type == "Submodel"]
        return {
            "assetAdministrationShells": [{"id": (s.attributes or {}).get("raw_id", s.id), "idShort": s.label or s.id} for s in shells],
            "submodels": [{"id": (sm.attributes or {}).get("raw_id", sm.id), "idShort": sm.label or sm.id, "submodelElements": []} for sm in submodels],
            "mappings": mappings or [],
        }

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
