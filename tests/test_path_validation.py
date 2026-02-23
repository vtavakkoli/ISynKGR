import pytest

from isynkgr.icr.mapping_schema import ingest_mapping_payload, normalize_mapping_path
from isynkgr.icr.path_validation import is_valid_protocol_path


@pytest.mark.parametrize(
    "path",
    [
        "aas://machineA/submodel/sm_main/element/temp/value",
        "opcua://ns=2;s=Machine.Speed",
        "opcua://ns=2;i=1000",
        "iec61499://deviceA/resource1/fbController/output",
        "ieee1451://ted42/ch1/temperature",
    ],
)
def test_supported_protocol_paths_are_valid(path: str) -> None:
    assert is_valid_protocol_path(path)


def test_normalize_preserves_scheme_separator() -> None:
    assert normalize_mapping_path("opcua://ns=2;i=1000") == "opcua://ns=2;i=1000"


def test_ingest_rejects_non_protocol_paths() -> None:
    with pytest.raises(ValueError, match="source_path must include protocol prefix"):
        ingest_mapping_payload(
            {
                "source_path": "ns=2;i=1000",
                "target_path": "aas://aas-1/submodel/default/element/value",
                "mapping_type": "equivalent",
                "confidence": 0.8,
                "rationale": "sufficient rationale",
                "evidence": [],
            }
        )
