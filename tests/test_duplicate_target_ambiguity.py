from __future__ import annotations

from dataclasses import dataclass

from isynkgr.canonical.model import CanonicalModel, CanonicalNode
from isynkgr.pipeline import adaptive_candidate_ranker as ranker_mod
from isynkgr.pipeline.adaptive_candidate_ranker import AdaptiveCandidateRankerPipeline, TranslatorConfig
from isynkgr.retrieval.graphrag import GraphRAGRetriever
from isynkgr.rules.engine import RuleEngine


@dataclass
class _Adapter:
    standard: str
    node_id: str

    def parse(self, _raw):
        return CanonicalModel(
            standard=self.standard,
            nodes=[CanonicalNode(id=self.node_id, type="signal", label="pressure", attributes={"dtype": "FLOAT", "unit": "bar"})],
            edges=[],
        )

    def serialize(self, _model, mappings):
        return {"mappings": mappings}

    def validate(self, _artifact):
        return {"valid": True, "violations": []}


class _LLM:
    def complete_json(self, *_args, **_kwargs):
        return {"mappings": []}


def test_ambiguous_duplicate_targets_return_no_match(monkeypatch):
    monkeypatch.setattr(
        ranker_mod,
        "ADAPTERS",
        {
            "opcua": _Adapter("opcua", "opcua://ns=2;s=Pressure"),
            "aas": _Adapter("aas", "aas://asset-1/submodel/default/element/pressure/value"),
        },
    )
    pipeline = AdaptiveCandidateRankerPipeline(llm=_LLM(), retriever=GraphRAGRetriever(top_k=5), rules=RuleEngine())
    result = pipeline.run(
        source_standard="opcua",
        target_standard="aas",
        source_raw="x",
        mode="adaptive_candidate_ranker",
        config=TranslatorConfig(component_flags={"rules": False, "llm": False, "uncertainty_threshold": 0.0, "ambiguity_margin": 0.1}),
        target_candidates=[
            "aas://asset-1/submodel/default/element/pressure/value",
            "aas://asset-7/submodel/default/element/pressure/value",
            "aas://asset-13/submodel/default/element/pressure/value",
        ],
    )
    assert result.mappings[0].mapping_type.value == "no_match"
    assert "ranker:ambiguous_duplicate_candidates" in result.mappings[0].evidence


def test_asset_context_breaks_duplicate_tie(monkeypatch):
    monkeypatch.setattr(
        ranker_mod,
        "ADAPTERS",
        {
            "opcua": _Adapter("opcua", "opcua://ns=2;s=line-1.asset-7.pump.pressure"),
            "aas": _Adapter("aas", "aas://asset-1/submodel/default/element/pressure/value"),
        },
    )
    pipeline = AdaptiveCandidateRankerPipeline(llm=_LLM(), retriever=GraphRAGRetriever(top_k=5), rules=RuleEngine())
    result = pipeline.run(
        source_standard="opcua",
        target_standard="aas",
        source_raw="x",
        mode="adaptive_candidate_ranker",
        config=TranslatorConfig(component_flags={"rules": False, "llm": False, "uncertainty_threshold": 0.0, "ambiguity_margin": 0.1}),
        target_candidates=[
            "aas://asset-1/submodel/default/element/pressure/value",
            "aas://asset-7/submodel/default/element/pressure/value",
            "aas://asset-13/submodel/default/element/pressure/value",
        ],
    )
    assert result.mappings[0].mapping_type.value != "no_match"
    assert result.mappings[0].target_path == "aas://asset-7/submodel/default/element/pressure/value"
