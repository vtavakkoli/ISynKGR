from __future__ import annotations

from dataclasses import dataclass

from benchmark.run_sut import _enforce_generation_cardinality
from isynkgr.canonical.model import CanonicalModel, CanonicalNode
from isynkgr.canonical.schemas import EvidenceItem
from isynkgr.pipeline.hybrid import HybridPipeline, TranslatorConfig
from isynkgr.rules.engine import RuleEngine


@dataclass
class _Adapter:
    standard: str

    def parse(self, _raw):
        return CanonicalModel(standard=self.standard, nodes=[CanonicalNode(id="opcua://ns=2;i=1", type="signal", label="temperature")], edges=[])

    def serialize(self, _model, mappings):
        return {"mappings": mappings}

    def validate(self, _artifact):
        return {"valid": True, "violations": []}


class _Retriever:
    def retrieve(self, *_args, **_kwargs):
        return [
            EvidenceItem(
                id="cand:1",
                kind="target_candidate",
                text="candidate",
                score=0.9,
                payload={"candidate_path": "aas://asset/submodel/default/element/temperature/value"},
            )
        ]


class _HallucinatingLLM:
    def complete_json(self, *_args, **_kwargs):
        return {
            "mappings": [
                {
                    "source_path": "opcua://ns=2;i=1",
                    "target_path": "aas://invented/path",
                    "mapping_type": "equivalent",
                    "transform": None,
                    "confidence": 0.9,
                    "rationale": "hallucinated",
                    "evidence": ["llm"],
                }
            ]
        }


def test_generation_cardinality_enforced_for_one_to_one():
    mappings, applied = _enforce_generation_cardinality(
        [
            {"source_path": "opcua://a", "target_path": "aas://x", "mapping_type": "equivalent", "confidence": 0.4},
            {"source_path": "opcua://a", "target_path": "aas://y", "mapping_type": "equivalent", "confidence": 0.9},
        ],
        {"mode": "one_to_one", "expected_count": 1, "grouped_1": False},
    )
    assert applied is True
    assert len(mappings) == 1
    assert mappings[0]["target_path"] == "aas://y"


def test_rule_shortcut_can_be_disabled_in_benchmark_mode():
    source = CanonicalModel(standard="opcua", nodes=[CanonicalNode(id="opcua://ns=2;i=1003", type="signal", label="Pump3")], edges=[])
    mappings = RuleEngine().apply_rules(source, target_protocol="aas", allow_synthetic_shortcuts=False)
    assert mappings[0].mapping_type.value == "no_match"
    assert mappings[0].target_path == ""


def test_llm_is_constrained_when_retrieval_is_high_confidence(monkeypatch):
    from isynkgr.pipeline import hybrid as hybrid_mod

    monkeypatch.setattr(hybrid_mod, "ADAPTERS", {"opcua": _Adapter("opcua"), "aas": _Adapter("aas")})
    pipeline = HybridPipeline(llm=_HallucinatingLLM(), retriever=_Retriever(), rules=RuleEngine())
    result = pipeline.run("opcua", "aas", source_raw="x", mode="hybrid", config=TranslatorConfig(component_flags={"rules": False}))
    assert result.mappings[0].target_path != "aas://invented/path"
    assert result.mappings[0].target_path in {"", "aas://asset/submodel/default/element/temperature/value"}


def test_scenarios_have_distinct_component_activation(monkeypatch):
    from isynkgr.pipeline import hybrid as hybrid_mod

    monkeypatch.setattr(hybrid_mod, "ADAPTERS", {"opcua": _Adapter("opcua"), "aas": _Adapter("aas")})
    pipeline = HybridPipeline(llm=_HallucinatingLLM(), retriever=_Retriever(), rules=RuleEngine())
    full = pipeline.run("opcua", "aas", source_raw="x", mode="hybrid", config=TranslatorConfig(component_flags={}))
    no_retrieval = pipeline.run("opcua", "aas", source_raw="x", mode="hybrid", config=TranslatorConfig(component_flags={"retrieval": False}))
    assert full.provenance.metadata["execution"]["retrieval_ran"] is True
    assert no_retrieval.provenance.metadata["execution"]["retrieval_ran"] is False
