from __future__ import annotations

import pytest

from isynkgr.icr.mapping_output_contract import normalize_mapping_items


@pytest.mark.parametrize(
    ("sample", "expected_status", "expected_detail"),
    [
        (
            {
                "source_path": "opcua://ns=2;i=1001",
                "target_path": "aas://motorA/submodel/default/element/speed/value",
                "mapping_type": "equivalent",
                "confidence": "0.91",
                "rationale": "Motor speed points are semantically equivalent in both models.",
                "evidence": ["name-match", "unit-rpm"],
            },
            "accepted",
            "equivalent",
        ),
        (
            {
                "source_path": "opcua://ns=2;s=Machine.Temperature",
                "target_path": "aas://machineA/submodel/default/element/temperature/value",
                "mapping_type": "transform",
                "transform": {"op": "cast", "args": {"to": "float"}},
                "confidence": "0.78",
                "rationale": "Temperature arrives as text and must be converted to a numeric value.",
            },
            "accepted",
            "transform",
        ),
        (
            {
                "source_path": "ns=2;i=2000",
                "target_path": "motorA/submodel/default/element/voltage/value",
                "mapping_type": "equivalent",
                "confidence": 0.80,
                "rationale": "Raw ids without protocol should be auto-expanded using protocol defaults.",
            },
            "accepted",
            "opcua://ns=2;i=2000",
        ),
        (
            {
                "source_path": "opcua://ns=2;s=Unmapped.Signal",
                "mapping_type": "no_match",
                "confidence": 0.22,
                "rationale": "No meaningful AAS counterpart for this internal diagnostic tag.",
            },
            "accepted",
            "no_match",
        ),
        (
            {
                "source_path": "opcua://ns=2;i=3333",
                "target_path": "aas://machineA/submodel/default/element/pressure/value",
                "mapping_type": "weird_type",
                "confidence": 0.56,
                "rationale": "Unknown mapping type from LLM should fall back to approximate.",
            },
            "accepted",
            "approximate",
        ),
        (
            {
                "source_path": "opcua://ns=2;s=Machine.Current",
                "target_path": "opcua://ns=2;s=StillOpcua",
                "mapping_type": "equivalent",
                "confidence": 0.67,
                "rationale": "Wrong target protocol should be converted to target protocol namespace.",
            },
            "rejected",
            "target_path does not match required 'aas' path format",
        ),
        (
            {
                "source_path": "",
                "target_path": "aas://machineA/submodel/default/element/temp/value",
                "mapping_type": "equivalent",
                "confidence": 0.30,
                "rationale": "Missing source path should not pass protocol validation.",
            },
            "rejected",
            "source_path does not match required 'opcua' path format",
        ),
        (
            {
                "source_path": "opcua://ns=2;s=Machine.Flow",
                "target_path": "",
                "mapping_type": "equivalent",
                "confidence": 0.44,
                "rationale": "Equivalent mapping with empty target should fail AAS path validation.",
            },
            "rejected",
            "target_path does not match required 'aas' path format",
        ),
        (
            {
                "source_path": "opcua://ns=2;i=1010",
                "target_path": "aas://machineA/submodel/default/element/current/value",
                "mapping_type": "equivalent",
                "confidence": "high",
                "rationale": "Non-numeric confidence should trigger normalization failure.",
            },
            "rejected",
            "could not convert string to float",
        ),
        (
            {
                "source_path": "opcua://ns=2;i=2020",
                "target_path": "aas://machineA/submodel/default/element/temp/value",
                "mapping_type": "transform",
                "transform": {"args": {"to": "float"}},
                "confidence": 0.88,
                "rationale": "Transform object without operation should fail schema validation.",
            },
            "rejected",
            "transform.op is required",
        ),
    ],
)
def test_llm_conversion_samples(sample: dict, expected_status: str, expected_detail: str) -> None:
    report = normalize_mapping_items([sample], source_protocol="opcua", target_protocol="aas", method="llm")

    if expected_status == "accepted":
        assert len(report.accepted) == 1, f"Input={sample} Rejected={ [r.model_dump() for r in report.rejected] }"
        normalized = report.accepted[0].model_dump()
        snapshot = {"input": sample, "output": normalized}
        print(snapshot)
        joined = " ".join(str(v) for v in normalized.values())
        assert expected_detail in joined
    else:
        assert len(report.rejected) == 1, f"Input={sample} Accepted={ [a.model_dump() for a in report.accepted] }"
        rejection = report.rejected[0].model_dump()
        snapshot = {"input": sample, "error": rejection}
        print(snapshot)
        assert expected_detail in (rejection.get("exception") or "")
