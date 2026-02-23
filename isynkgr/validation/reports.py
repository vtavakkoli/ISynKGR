from __future__ import annotations

from isynkgr.canonical.schemas import Mapping
from isynkgr.icr.path_validation import validate_protocol_path


def mapping_validity(mappings: list[Mapping]) -> tuple[bool, list[str]]:
    errs = []
    for m in mappings:
        if not (0 <= m.confidence <= 1):
            errs.append(f"invalid confidence {m.confidence} for {m.source_path}")

        try:
            validate_protocol_path(m.source_path, "source_path")
        except ValueError as exc:
            errs.append(str(exc))

        try:
            validate_protocol_path(m.target_path, "target_path")
        except ValueError as exc:
            errs.append(str(exc))
    return (not errs, errs)
