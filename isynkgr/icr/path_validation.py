from __future__ import annotations

import re

# NOTE: keep patterns linear-time and fully anchored for deterministic validation.
# We intentionally use explicit character classes and avoid nested unbounded groups.
_SEGMENT = r"[A-Za-z0-9._~-]+"

AAS_PATH_PATTERN = re.compile(
    rf"^aas://(?P<aas_id>{_SEGMENT})/submodel/(?P<sm_idshort>{_SEGMENT})/element/(?P<path>{_SEGMENT}(?:/{_SEGMENT})*)$"
)
OPCUA_PATH_PATTERN = re.compile(
    r"^opcua://ns=(?P<ns>[0-9]+);(?:(?:s=(?P<string_id>[A-Za-z0-9._~:-]+))|(?:i=(?P<int_id>[0-9]+)))$"
)
IEC61499_PATH_PATTERN = re.compile(
    rf"^iec61499://(?P<device>{_SEGMENT})/(?P<resource>{_SEGMENT})/(?P<fb>{_SEGMENT})/(?P<var>{_SEGMENT})$"
)
IEEE1451_PATH_PATTERN = re.compile(
    rf"^ieee1451://(?P<ted_id>{_SEGMENT})/(?P<channel>{_SEGMENT})/(?P<field>{_SEGMENT})$"
)

PROTOCOL_PATH_PATTERNS: dict[str, re.Pattern[str]] = {
    "aas": AAS_PATH_PATTERN,
    "opcua": OPCUA_PATH_PATTERN,
    "iec61499": IEC61499_PATH_PATTERN,
    "ieee1451": IEEE1451_PATH_PATTERN,
}


def get_supported_path_protocols() -> tuple[str, ...]:
    return tuple(PROTOCOL_PATH_PATTERNS.keys())


def detect_path_protocol(path: str) -> str | None:
    value = (path or "").strip()
    protocol, sep, _ = value.partition("://")
    if not sep:
        return None
    return protocol.lower()


def is_valid_protocol_path(path: str) -> bool:
    protocol = detect_path_protocol(path)
    if protocol is None:
        return False
    pattern = PROTOCOL_PATH_PATTERNS.get(protocol)
    if pattern is None:
        return False
    return bool(pattern.fullmatch(path.strip()))


def validate_protocol_path(path: str, field_name: str = "path") -> None:
    value = (path or "").strip()
    protocol = detect_path_protocol(value)
    if protocol is None:
        raise ValueError(
            f"{field_name} must include protocol prefix (supported: {', '.join(get_supported_path_protocols())})"
        )
    pattern = PROTOCOL_PATH_PATTERNS.get(protocol)
    if pattern is None:
        raise ValueError(
            f"{field_name} protocol '{protocol}' is unsupported (supported: {', '.join(get_supported_path_protocols())})"
        )
    if not pattern.fullmatch(value):
        raise ValueError(f"{field_name} does not match required '{protocol}' path format")
