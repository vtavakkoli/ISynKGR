from __future__ import annotations

import json
from typing import Any

from isynkgr.canonical.model import CanonicalEdge, CanonicalModel, CanonicalNode
from isynkgr.canonical.schemas import ValidationReport, ValidationViolation

_ALLOWED_DTYPES = {"BOOL", "INT", "FLOAT", "STRING"}
_NUMERIC_DTYPES = {"INT", "FLOAT"}


class IEC61499Adapter:
    name = "iec61499"

    def _load(self, raw: str | bytes | dict[str, Any]) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, bytes):
            return json.loads(raw.decode())
        return json.loads(raw)

    def parse(self, raw: str | bytes | dict[str, Any]) -> CanonicalModel:
        doc = self._load(raw)
        model = CanonicalModel(standard=self.name)
        for device in doc.get("devices", []):
            did = device.get("id", "")
            if not did:
                continue
            model.nodes.append(CanonicalNode(id=did, type="Device", label=device.get("name")))
            for resource in device.get("resources", []):
                rid = resource.get("id", "")
                if not rid:
                    continue
                resource_node_id = f"{did}/{rid}"
                model.nodes.append(CanonicalNode(id=resource_node_id, type="Resource", label=resource.get("name"), attributes={"resource_id": rid}))
                model.edges.append(CanonicalEdge(source=did, target=resource_node_id, relation="hasResource"))
                for fb in resource.get("function_blocks", []):
                    fbid = fb.get("id", "")
                    if not fbid:
                        continue
                    fb_node_id = f"{resource_node_id}/{fbid}"
                    model.nodes.append(CanonicalNode(id=fb_node_id, type="FunctionBlock", label=fb.get("name"), attributes={"fb_type": fb.get("type"), "fb_id": fbid}))
                    model.edges.append(CanonicalEdge(source=resource_node_id, target=fb_node_id, relation="hasFunctionBlock"))
                    for direction in ("inputs", "outputs"):
                        for signal in fb.get(direction, []):
                            sid = signal.get("id", "")
                            if not sid:
                                continue
                            signal_node_id = f"{fb_node_id}/{sid}"
                            attrs = {
                                "direction": direction[:-1],
                                "signal_id": sid,
                                "dtype": signal.get("dtype"),
                                "unit": signal.get("unit"),
                                "range": signal.get("range"),
                            }
                            model.nodes.append(CanonicalNode(id=signal_node_id, type="Signal", label=signal.get("name"), attributes=attrs))
                            model.edges.append(CanonicalEdge(source=fb_node_id, target=signal_node_id, relation=f"has{direction[:-1].capitalize()}"))
        return model

    def serialize(self, model: CanonicalModel, mappings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        devices: dict[str, dict[str, Any]] = {}
        for node in model.nodes:
            if node.type == "Device":
                devices[node.id] = {"id": node.id, "name": node.label, "resources": []}

        resource_index: dict[str, tuple[str, dict[str, Any]]] = {}
        for edge in model.edges:
            if edge.relation == "hasResource" and edge.source in devices:
                resource = next((n for n in model.nodes if n.id == edge.target and n.type == "Resource"), None)
                if resource is None:
                    continue
                rid = (resource.attributes or {}).get("resource_id") or resource.id.rsplit("/", 1)[-1]
                resource_doc = {"id": rid, "name": resource.label, "function_blocks": []}
                devices[edge.source]["resources"].append(resource_doc)
                resource_index[resource.id] = (edge.source, resource_doc)

        fb_index: dict[str, dict[str, Any]] = {}
        for edge in model.edges:
            if edge.relation == "hasFunctionBlock" and edge.source in resource_index:
                fb = next((n for n in model.nodes if n.id == edge.target and n.type == "FunctionBlock"), None)
                if fb is None:
                    continue
                fbid = (fb.attributes or {}).get("fb_id") or fb.id.rsplit("/", 1)[-1]
                fb_doc = {
                    "id": fbid,
                    "name": fb.label,
                    "type": (fb.attributes or {}).get("fb_type"),
                    "inputs": [],
                    "outputs": [],
                }
                _, resource_doc = resource_index[edge.source]
                resource_doc["function_blocks"].append(fb_doc)
                fb_index[fb.id] = fb_doc

        for edge in model.edges:
            if edge.source not in fb_index:
                continue
            if edge.relation not in {"hasInput", "hasOutput"}:
                continue
            signal = next((n for n in model.nodes if n.id == edge.target and n.type == "Signal"), None)
            if signal is None:
                continue
            attrs = signal.attributes or {}
            sig_doc = {
                "id": attrs.get("signal_id") or signal.id.rsplit("/", 1)[-1],
                "name": signal.label,
                "dtype": attrs.get("dtype"),
                "unit": attrs.get("unit"),
                "range": attrs.get("range"),
            }
            if edge.relation == "hasInput":
                fb_index[edge.source]["inputs"].append(sig_doc)
            else:
                fb_index[edge.source]["outputs"].append(sig_doc)

        return {"standard": self.name, "devices": list(devices.values()), "mappings": mappings or []}

    def _validate_signal(self, signal: dict[str, Any], location: str) -> list[ValidationViolation]:
        violations: list[ValidationViolation] = []
        for field in ("id", "dtype"):
            if not signal.get(field):
                violations.append(ValidationViolation(type="required", message=f"{location}.{field} missing"))

        dtype = signal.get("dtype")
        if dtype and dtype not in _ALLOWED_DTYPES:
            violations.append(ValidationViolation(type="dtype", message=f"{location}.dtype invalid: {dtype}"))

        unit = signal.get("unit")
        if unit is not None and (not isinstance(unit, str) or not unit.strip()):
            violations.append(ValidationViolation(type="unit", message=f"{location}.unit must be a non-empty string when provided"))

        rng = signal.get("range")
        if dtype in _NUMERIC_DTYPES:
            if not isinstance(rng, dict):
                violations.append(ValidationViolation(type="range", message=f"{location}.range required for numeric dtype"))
            else:
                min_v = rng.get("min")
                max_v = rng.get("max")
                if not isinstance(min_v, (int, float)) or not isinstance(max_v, (int, float)):
                    violations.append(ValidationViolation(type="range", message=f"{location}.range min/max must be numeric"))
                elif min_v > max_v:
                    violations.append(ValidationViolation(type="range", message=f"{location}.range min cannot be greater than max"))
        elif rng is not None:
            violations.append(ValidationViolation(type="range", message=f"{location}.range allowed only for numeric dtype"))

        return violations

    def validate(self, raw: str | bytes | dict[str, Any]) -> ValidationReport:
        violations: list[ValidationViolation] = []
        try:
            doc = self._load(raw)
        except Exception as exc:
            return ValidationReport(valid=False, violations=[ValidationViolation(type="json", message=str(exc))])

        devices = doc.get("devices")
        if not isinstance(devices, list):
            violations.append(ValidationViolation(type="required", message="devices must be a list"))
            return ValidationReport(valid=False, violations=violations)

        device_ids: set[str] = set()
        for d_idx, device in enumerate(devices):
            did = device.get("id")
            if not did:
                violations.append(ValidationViolation(type="required", message=f"devices[{d_idx}].id missing"))
                continue
            if did in device_ids:
                violations.append(ValidationViolation(type="id", message=f"duplicate device id: {did}"))
            device_ids.add(did)

            resource_ids: set[str] = set()
            for r_idx, resource in enumerate(device.get("resources", [])):
                rid = resource.get("id")
                if not rid:
                    violations.append(ValidationViolation(type="required", message=f"devices[{d_idx}].resources[{r_idx}].id missing"))
                    continue
                if rid in resource_ids:
                    violations.append(ValidationViolation(type="id", message=f"duplicate resource id in {did}: {rid}"))
                resource_ids.add(rid)

                fb_ids: set[str] = set()
                for f_idx, fb in enumerate(resource.get("function_blocks", [])):
                    fbid = fb.get("id")
                    if not fbid:
                        violations.append(ValidationViolation(type="required", message=f"devices[{d_idx}].resources[{r_idx}].function_blocks[{f_idx}].id missing"))
                        continue
                    if fbid in fb_ids:
                        violations.append(ValidationViolation(type="id", message=f"duplicate function block id in {did}/{rid}: {fbid}"))
                    fb_ids.add(fbid)

                    signal_ids: set[str] = set()
                    for direction in ("inputs", "outputs"):
                        for s_idx, signal in enumerate(fb.get(direction, [])):
                            sid = signal.get("id")
                            if sid in signal_ids:
                                violations.append(ValidationViolation(type="id", message=f"duplicate signal id in {did}/{rid}/{fbid}: {sid}"))
                            if sid:
                                signal_ids.add(sid)
                            location = f"devices[{d_idx}].resources[{r_idx}].function_blocks[{f_idx}].{direction}[{s_idx}]"
                            violations.extend(self._validate_signal(signal, location))

        return ValidationReport(valid=not violations, violations=violations)
