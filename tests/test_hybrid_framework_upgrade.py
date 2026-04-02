from __future__ import annotations

from dataclasses import dataclass

from benchmark.run_sut import _validate_mapping
from isynkgr.canonical.model import CanonicalModel, CanonicalNode
from isynkgr.canonical.schemas import EvidenceItem, Mapping
from isynkgr.pipeline.hybrid import HybridPipeline, TranslatorConfig
from isynkgr.retrieval.graphrag import GraphRAGRetriever
from isynkgr.rules.engine import RuleEngine


@dataclass
class _Adapter:
    standard: str

    def parse(self, _raw):
        return CanonicalModel(
            standard=self.standard,
            nodes=[
                CanonicalNode(id="opcua://ns=2;i=1", type="signal", label="temperature", attributes={"dtype": "FLOAT", "unit": "C"}),
                CanonicalNode(id="opcua://ns=2;i=2", type="signal", label="pressure", attributes={"dtype": "FLOAT", "unit": "bar"}),
            ],
            edges=[],
        )

    def serialize(self, _model, mappings):
        return {"mappings": mappings}

    def validate(self, _artifact):
        return {"valid": True, "violations": []}


class _Retriever:
    def retrieve(self, *_args, **_kwargs):
        return [
            EvidenceItem(id="node:1:c1", kind="target_candidate", text="temperature", score=0.95, payload={"source_node": "opcua://ns=2;i=1", "candidate_path": "aas://asset/submodel/default/element/temperature/value", "label": "temperature"}),
            EvidenceItem(id="node:2:c1", kind="target_candidate", text="pressure", score=0.92, payload={"source_node": "opcua://ns=2;i=2", "candidate_path": "aas://asset/submodel/default/element/pressure/value", "label": "pressure"}),
        ]


class _Rules:
    def apply_rules(self, *_args, **_kwargs):
        return [
            Mapping(source_path="opcua://ns=2;i=1", target_path="aas://asset/submodel/default/element/temperature/value", mapping_type="label_match", confidence=0.85, rationale="rule temperature", evidence=["rule"]),
            Mapping(source_path="opcua://ns=2;i=2", target_path="aas://asset/submodel/default/element/pressure/value", mapping_type="label_match", confidence=0.85, rationale="rule pressure", evidence=["rule"]),
        ]


class _LLM:
    def complete_json(self, *_args, **_kwargs):
        return {
            "mappings": [
                {"source_path": "opcua://ns=2;i=1", "target_path": "aas://asset/submodel/default/element/temperature/value", "mapping_type": "equivalent", "transform": None, "confidence": 0.7, "rationale": "llm temperature", "evidence": ["llm"]},
                {"source_path": "opcua://ns=2;i=2", "target_path": "aas://asset/submodel/default/element/pressure/value", "mapping_type": "equivalent", "transform": None, "confidence": 0.7, "rationale": "llm pressure", "evidence": ["llm"]},
            ]
        }


def test_retrieval_keeps_candidates_per_source_node():
    src = CanonicalModel(
        standard="opcua",
        nodes=[CanonicalNode(id="opcua://a", type="signal", label="temperature"), CanonicalNode(id="opcua://b", type="signal", label="pressure")],
        edges=[],
    )
    tgt = CanonicalModel(
        standard="aas",
        nodes=[
            CanonicalNode(id="aas://asset/submodel/default/element/temperature/value", type="Property", label="temperature"),
            CanonicalNode(id="aas://asset/submodel/default/element/pressure/value", type="Property", label="pressure"),
        ],
        edges=[],
    )
    evidence = GraphRAGRetriever(top_k=1).retrieve(src, "aas", target_model=tgt)
    by_source = {item.payload.get("source_node"): item.payload.get("candidate_path") for item in evidence}
    assert by_source["opcua://a"] == "aas://asset/submodel/default/element/temperature/value"
    assert by_source["opcua://b"] == "aas://asset/submodel/default/element/pressure/value"


def test_rules_use_real_target_model():
    source = CanonicalModel(standard="opcua", nodes=[CanonicalNode(id="opcua://ns=2;i=1001", type="signal", label="temperature", attributes={"dtype": "FLOAT"})], edges=[])
    target = CanonicalModel(standard="aas", nodes=[CanonicalNode(id="aas://asset/submodel/default/element/temperature/value", type="Property", label="temperature", attributes={"dtype": "FLOAT"})], edges=[])
    mappings = RuleEngine().apply_rules(source, target_protocol="aas", target=target, allow_synthetic_shortcuts=False)
    assert mappings[0].mapping_type.value in {"label_match", "approximate"}
    assert mappings[0].target_path == "aas://asset/submodel/default/element/temperature/value"


def test_hybrid_runs_components_and_merges(monkeypatch):
    from isynkgr.pipeline import hybrid as hybrid_mod

    monkeypatch.setattr(hybrid_mod, "ADAPTERS", {"opcua": _Adapter("opcua"), "aas": _Adapter("aas")})
    pipeline = HybridPipeline(llm=_LLM(), retriever=_Retriever(), rules=_Rules())
    result = pipeline.run("opcua", "aas", source_raw="x", mode="hybrid", config=TranslatorConfig())
    execution = result.provenance.metadata["execution"]
    assert execution["rules_ran"] is True
    assert execution["retrieval_ran"] is True
    assert execution["llm_ran"] is True
    assert len(result.mappings) == 2


def test_ablation_flags_change_execution(monkeypatch):
    from isynkgr.pipeline import hybrid as hybrid_mod

    monkeypatch.setattr(hybrid_mod, "ADAPTERS", {"opcua": _Adapter("opcua"), "aas": _Adapter("aas")})
    pipeline = HybridPipeline(llm=_LLM(), retriever=_Retriever(), rules=_Rules())
    full = pipeline.run("opcua", "aas", source_raw="x", mode="hybrid", config=TranslatorConfig())
    no_rules = pipeline.run("opcua", "aas", source_raw="x", mode="hybrid", config=TranslatorConfig(component_flags={"rules": False}))
    assert full.provenance.metadata["execution"]["rules_ran"] is True
    assert no_rules.provenance.metadata["execution"]["rules_ran"] is False


def test_semantic_validator_catches_invalid_target_for_source_node():
    is_valid, violations, _ = _validate_mapping(
            {"source_path": "opcua://ns=2;i=1", "target_path": "aas://asset/submodel/default/element/x/value", "mapping_type": "equivalent", "transform": None, "confidence": 0.9, "rationale": "valid enough rationale", "evidence": []},
        source_protocol="opcua",
        target_protocol="aas",
        seen_keys=set(),
        semantic_context={"opcua://ns=2;i=1": {"candidate_paths": ["aas://asset/submodel/default/element/y/value"]}},
    )
    assert is_valid is False
    assert any(v["type"] == "semantic_target_not_in_schema_candidates" for v in violations)
