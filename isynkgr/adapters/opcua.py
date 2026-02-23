from __future__ import annotations

from typing import Any
from xml.etree import ElementTree as ET

from isynkgr.canonical.model import CanonicalEdge, CanonicalModel, CanonicalNode
from isynkgr.canonical.schemas import ValidationReport, ValidationViolation
from isynkgr.icr.entities import Endpoint, Relationship, build_endpoint_path

UA_TYPES = {"UAObjectType", "UAVariable", "UADataType"}


class OPCUAAdapter:
    name = "opcua"

    def parse(self, raw: str | bytes | dict[str, Any]) -> CanonicalModel:
        xml = raw.decode() if isinstance(raw, bytes) else raw
        if isinstance(raw, dict):
            xml = raw.get("xml", "")
        root = ET.fromstring(xml)
        model = CanonicalModel(standard=self.name)
        refs: list[tuple[str, str, str]] = []
        for elem in root.iter():
            tag = elem.tag.split("}")[-1]
            if tag in UA_TYPES:
                nid = elem.attrib.get("NodeId", "")
                if not nid:
                    continue
                endpoint = Endpoint(id=nid, path=build_endpoint_path(self.name, nid), protocol=self.name, label=elem.attrib.get("BrowseName"), metadata={"DisplayName": (elem.findtext("{*}DisplayName") or ""), "raw_id": nid})
                model.nodes.append(CanonicalNode(id=endpoint.path, type=tag, label=endpoint.label, attributes=endpoint.model_dump()))
                for r in elem.findall("{*}References/{*}Reference"):
                    if r.text:
                        refs.append((build_endpoint_path(self.name, nid), build_endpoint_path(self.name, r.text.strip()), r.attrib.get("ReferenceType", "References")))
        node_ids = {n.id for n in model.nodes}
        for s, t, rel in refs:
            if t in node_ids:
                model.edges.append(CanonicalEdge(source=s, target=t, relation=rel))
        return model

    def serialize(self, model: CanonicalModel, mappings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {
            "format": "opcua-rdfish-json",
            "nodes": [n.model_dump() for n in model.nodes],
            "edges": [e.model_dump() for e in model.edges],
            "mappings": mappings or [],
        }

    def validate(self, raw: str | bytes | dict[str, Any]) -> ValidationReport:
        violations: list[ValidationViolation] = []
        try:
            xml = raw.decode() if isinstance(raw, bytes) else raw
            if isinstance(raw, dict):
                xml = raw.get("xml", "")
            root = ET.fromstring(xml)
        except Exception as exc:
            return ValidationReport(valid=False, violations=[ValidationViolation(type="xml", message=str(exc))])
        nodes = set()
        refs = []
        for elem in root.iter():
            tag = elem.tag.split("}")[-1]
            if tag in UA_TYPES:
                nid = elem.attrib.get("NodeId")
                if not nid:
                    violations.append(ValidationViolation(type="required", message="NodeId missing"))
                    continue
                nodes.add(nid)
                for r in elem.findall("{*}References/{*}Reference"):
                    if r.text:
                        refs.append(r.text.strip())
        for r in refs:
            if r not in nodes:
                violations.append(ValidationViolation(type="integrity", message=f"Reference target missing: {r}"))
        return ValidationReport(valid=not violations, violations=violations)
