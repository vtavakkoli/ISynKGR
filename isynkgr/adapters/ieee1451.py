from __future__ import annotations

import json
from typing import Any

from isynkgr.canonical.model import CanonicalEdge, CanonicalModel, CanonicalNode
from isynkgr.canonical.schemas import ValidationReport, ValidationViolation
from isynkgr.icr.entities import Asset, Relationship, Sensor, build_asset_path, build_sensor_path

_ALLOWED_DTYPES = {"BOOL", "INT", "FLOAT", "STRING"}
_NUMERIC_DTYPES = {"INT", "FLOAT"}


class IEEE1451Adapter:
    name = "ieee1451"

    def _load(self, raw: str | bytes | dict[str, Any]) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, bytes):
            return json.loads(raw.decode())
        return json.loads(raw)

    def parse(self, raw: str | bytes | dict[str, Any]) -> CanonicalModel:
        doc = self._load(raw)
        model = CanonicalModel(standard=self.name)
        for teds in doc.get("teds", []):
            tid = teds.get("id", "")
            if not tid:
                continue
            ted_asset = Asset(id=tid, path=build_asset_path(self.name, tid), protocol=self.name, label=teds.get("name"), metadata={"meta": teds.get("meta", {}), "raw_id": tid})
            model.nodes.append(CanonicalNode(id=ted_asset.path, type="TEDS", label=teds.get("name"), attributes=ted_asset.model_dump()))
            for channel in teds.get("channels", []):
                cid = channel.get("id", "")
                if not cid:
                    continue
                channel_node_id = build_sensor_path(self.name, tid, cid)
                attrs = {
                    "channel_id": cid,
                    "dtype": channel.get("dtype"),
                    "unit": channel.get("unit"),
                    "range": channel.get("range"),
                }
                sensor = Sensor(id=cid, path=channel_node_id, protocol=self.name, label=channel.get("name"), metadata=attrs)
                model.nodes.append(CanonicalNode(id=sensor.path, type="Channel", label=sensor.label, attributes=sensor.model_dump()))
                rel = Relationship(source_path=ted_asset.path, target_path=channel_node_id, relation="hasChannel")
                model.edges.append(CanonicalEdge(source=rel.source_path, target=rel.target_path, relation=rel.relation))
        return model

    def serialize(self, model: CanonicalModel, mappings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        teds_docs: dict[str, dict[str, Any]] = {}
        for node in model.nodes:
            if node.type == "TEDS":
                raw_id = (node.attributes or {}).get("metadata", {}).get("raw_id") or (node.attributes or {}).get("raw_id") or node.id
                teds_docs[node.id] = {
                    "id": raw_id,
                    "name": node.label,
                    "meta": (node.attributes or {}).get("meta", {}),
                    "channels": [],
                }

        for edge in model.edges:
            if edge.relation != "hasChannel" or edge.source not in teds_docs:
                continue
            channel = next((n for n in model.nodes if n.id == edge.target and n.type == "Channel"), None)
            if channel is None:
                continue
            attrs = channel.attributes or {}
            meta = attrs.get("metadata", attrs)
            teds_docs[edge.source]["channels"].append(
                {
                    "id": meta.get("channel_id") or channel.id.rsplit("/", 2)[-2],
                    "name": channel.label,
                    "dtype": meta.get("dtype"),
                    "unit": meta.get("unit"),
                    "range": meta.get("range"),
                }
            )

        return {"standard": self.name, "teds": list(teds_docs.values()), "mappings": mappings or []}

    def _validate_channel(self, channel: dict[str, Any], location: str) -> list[ValidationViolation]:
        violations: list[ValidationViolation] = []
        for field in ("id", "dtype"):
            if not channel.get(field):
                violations.append(ValidationViolation(type="required", message=f"{location}.{field} missing"))

        dtype = channel.get("dtype")
        if dtype and dtype not in _ALLOWED_DTYPES:
            violations.append(ValidationViolation(type="dtype", message=f"{location}.dtype invalid: {dtype}"))

        unit = channel.get("unit")
        if unit is not None and (not isinstance(unit, str) or not unit.strip()):
            violations.append(ValidationViolation(type="unit", message=f"{location}.unit must be a non-empty string when provided"))

        rng = channel.get("range")
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

        teds_list = doc.get("teds")
        if not isinstance(teds_list, list):
            violations.append(ValidationViolation(type="required", message="teds must be a list"))
            return ValidationReport(valid=False, violations=violations)

        ted_ids: set[str] = set()
        for t_idx, teds in enumerate(teds_list):
            tid = teds.get("id")
            if not tid:
                violations.append(ValidationViolation(type="required", message=f"teds[{t_idx}].id missing"))
                continue
            if tid in ted_ids:
                violations.append(ValidationViolation(type="id", message=f"duplicate teds id: {tid}"))
            ted_ids.add(tid)

            channel_ids: set[str] = set()
            for c_idx, channel in enumerate(teds.get("channels", [])):
                cid = channel.get("id")
                if cid in channel_ids:
                    violations.append(ValidationViolation(type="id", message=f"duplicate channel id in {tid}: {cid}"))
                if cid:
                    channel_ids.add(cid)
                location = f"teds[{t_idx}].channels[{c_idx}]"
                violations.extend(self._validate_channel(channel, location))

        return ValidationReport(valid=not violations, violations=violations)
