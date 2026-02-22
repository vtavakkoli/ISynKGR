from __future__ import annotations

from isynkgr.canonical.schemas import Mapping


def mapping_validity(mappings: list[Mapping]) -> tuple[bool, list[str]]:
    errs = []
    for m in mappings:
        if not (0 <= m.confidence <= 1):
            errs.append(f"invalid confidence {m.confidence} for {m.source_id}")
    return (not errs, errs)
