from __future__ import annotations

from dataclasses import dataclass

from isynkgr.canonical.model import CanonicalModel, CanonicalNode
from isynkgr.icr.mapping_output_contract import normalize_mapping_item, validate_mapping_item
from isynkgr.pipeline.hybrid import HybridPipeline, TranslatorConfig


@dataclass
class _FakeAdapter:
    standard: str

    def parse(self, _raw):
        node_id = {
            "opcua": "ns=2;i=100",
            "aas": "asset1/submodel/default/element/temp",
            "iec61499": "device1/resource1/fb1/temperature",
        }.get(self.standard, "src_1")
        return CanonicalModel(
            standard=self.standard,
            nodes=[CanonicalNode(id=node_id, type="signal", label="temperature")],
            edges=[],
        )

    def serialize(self, _model, mappings):
        return {"mappings": mappings}

    def validate(self, _artifact):
        return {"valid": True, "violations": []}


class _FakeRetriever:
    def retrieve(self, *_args, **_kwargs):
        return []


class _FakeLLM:
    def complete_json(self, *_args, **_kwargs):
        return {
            "mappings": [
                {
                    "source_path": "bad-source",
                    "target_path": "opcua://wrong-target",
                    "mapping_type": "bad_enum",
                    "transform": {"op": "cast", "args": {}},
                    "confidence": "0.75",
                    "rationale": "short",
                    "evidence": "bad",
                }
            ]
        }


def test_mapping_output_contract_normalizes_and_validates_protocol_pairs():
    mapping = normalize_mapping_item(
        {
            "source_path": "device1/res1/fb1/temp_sensor",
            "target_path": "ted1/ch1/target_temp",
            "mapping_type": "transform",
            "transform": {"op": "cast", "args": {"to": "float"}},
            "confidence": "0.8",
            "rationale": "Convert string to float for interoperability",
            "evidence": ["synthetic"],
        },
        source_protocol="iec61499",
        target_protocol="ieee1451",
    )
    ok, err = validate_mapping_item(mapping.model_dump(), "iec61499", "ieee1451")
    assert ok, err


def test_rule_engine_outputs_schema_valid_no_match_for_unresolved(monkeypatch):
    from isynkgr.rules.engine import RuleEngine

    source = CanonicalModel(standard="opcua", nodes=[CanonicalNode(id="ns=2;i=10", type="signal", label="unknown")], edges=[])
    mappings = RuleEngine().apply_rules(source, target_protocol="aas", target=None)
    assert mappings[0].mapping_type.value == "no_match"
    ok, err = validate_mapping_item(mappings[0].model_dump(), "opcua", "aas")
    assert ok, err


def test_hybrid_modes_emit_schema_valid_mappings(monkeypatch):
    from isynkgr.pipeline import hybrid as hybrid_mod
    from isynkgr.rules.engine import RuleEngine

    monkeypatch.setattr(hybrid_mod, "ADAPTERS", {"opcua": _FakeAdapter("opcua"), "aas": _FakeAdapter("aas"), "iec61499": _FakeAdapter("iec61499")})
    pipeline = HybridPipeline(llm=_FakeLLM(), retriever=_FakeRetriever(), rules=RuleEngine())
    cfg = TranslatorConfig(seed=7)

    for mode, pair in (("llm_only", ("opcua", "aas")), ("rule_only", ("opcua", "aas")), ("hybrid", ("iec61499", "aas"))):
        result = pipeline.run(pair[0], pair[1], source_raw="x", mode=mode, config=cfg)
        assert result.mappings
        for mapping in result.mappings:
            ok, err = validate_mapping_item(mapping.model_dump(), pair[0], pair[1])
            assert ok, f"{mode}: {err}"
        assert isinstance(result.provenance.metadata.get("rejected_mappings", []), list)
