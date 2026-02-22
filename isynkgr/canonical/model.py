from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from isynkgr.common_model import SimpleModel


@dataclass
class CanonicalNode(SimpleModel):
    id: str
    type: str
    label: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    namespace: str | None = None


@dataclass
class CanonicalEdge(SimpleModel):
    source: str
    target: str
    relation: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalModel(SimpleModel):
    standard: str
    nodes: list[CanonicalNode] = field(default_factory=list)
    edges: list[CanonicalEdge] = field(default_factory=list)
    namespaces: dict[str, str] = field(default_factory=dict)
    identifiers: dict[str, str] = field(default_factory=dict)

    def node_index(self) -> dict[str, CanonicalNode]:
        return {n.id: n for n in self.nodes}
