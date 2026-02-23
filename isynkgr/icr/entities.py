from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator
except Exception:  # pragma: no cover
    BaseModel = object
    ConfigDict = dict

    def Field(default_factory=None):
        return default_factory()

    def field_validator(*_args, **_kwargs):
        def dec(fn):
            return fn

        return dec


def normalize_path(path: str) -> str:
    value = (path or "").strip().replace("\\", "/")
    if value.startswith("./"):
        value = value[2:]

    protocol, sep, rest = value.partition("://")
    if sep:
        rest = re.sub(r"/+", "/", rest)
        value = f"{protocol.lower()}://{rest}"
    else:
        value = re.sub(r"/+", "/", value)

    value = value.rstrip("/")
    value = re.sub(r"(^|/)(?:ns|namespace):", r"\1", value)
    return value


def _clean(segment: str) -> str:
    return normalize_path(str(segment)).strip("/")


def build_asset_path(protocol: str, asset_id: str) -> str:
    p = protocol.lower()
    sid = _clean(asset_id)
    if p == "aas":
        return normalize_path(f"aas://{sid}/submodel/default/element/asset")
    if p == "opcua":
        return normalize_path(f"opcua://{sid}")
    if p == "iec61499":
        return normalize_path(f"iec61499://{sid}/default/default/asset")
    if p == "ieee1451":
        return normalize_path(f"ieee1451://{sid}/default/asset")
    return normalize_path(f"{p}://{sid}")


def build_sensor_path(protocol: str, parent_id: str, sensor_id: str) -> str:
    p = protocol.lower()
    parent = _clean(parent_id)
    sid = _clean(sensor_id)
    if p == "aas":
        return normalize_path(f"aas://{parent}/submodel/default/element/{sid}")
    if p == "opcua":
        return normalize_path(f"opcua://{sid}")
    if p == "iec61499":
        parts = [x for x in parent.split("/") if x]
        while len(parts) < 3:
            parts.append("default")
        return normalize_path(f"iec61499://{parts[0]}/{parts[1]}/{parts[2]}/{sid}")
    if p == "ieee1451":
        return normalize_path(f"ieee1451://{parent}/{sid}/value")
    return normalize_path(f"{p}://{parent}/{sid}")


def build_signal_path(protocol: str, parent_id: str, signal_id: str) -> str:
    return build_sensor_path(protocol, parent_id, signal_id)


def build_endpoint_path(protocol: str, endpoint_id: str, namespace: str | None = None) -> str:
    eid = _clean(endpoint_id)
    if protocol.lower() == "opcua" and namespace:
        ns = _clean(namespace)
        if not eid.startswith("ns="):
            if eid.isdigit():
                eid = f"ns={ns};i={eid}"
            else:
                eid = f"ns={ns};s={eid}"
    return normalize_path(f"{protocol.lower()}://{eid}")


@dataclass
class ICREntity:
    id: str
    path: str
    protocol: str
    label: str | None = None
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = normalize_path(self.path)

    def model_dump(self) -> dict:
        return {
            "id": self.id,
            "path": self.path,
            "protocol": self.protocol,
            "label": self.label,
            "metadata": self.metadata,
            "kind": getattr(self, "kind", None),
        }


@dataclass
class Asset(ICREntity):
    kind: Literal["asset"] = "asset"


@dataclass
class Sensor(ICREntity):
    kind: Literal["sensor"] = "sensor"


@dataclass
class Signal(ICREntity):
    kind: Literal["signal"] = "signal"


@dataclass
class Endpoint(ICREntity):
    kind: Literal["endpoint"] = "endpoint"


@dataclass
class Relationship:
    source_path: str
    target_path: str
    relation: str

    def __post_init__(self) -> None:
        self.source_path = normalize_path(self.source_path)
        self.target_path = normalize_path(self.target_path)
