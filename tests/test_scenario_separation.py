from __future__ import annotations

from dataclasses import dataclass

from isynkgr.canonical.model import CanonicalModel, CanonicalNode
from isynkgr.canonical.schemas import EvidenceItem, Mapping
from isynkgr.pipeline.hybrid import HybridPipeline, TranslatorConfig


@dataclass
class _FakeAdapter:
    standard: str

    def parse(self, _raw):
        return CanonicalModel(standard=self.standard, nodes=[CanonicalNode(id="opcua://ns=2;i=1", type="signal", label="n1")], edges=[])

    def serialize(self, _model, mappings):
        return {"mappings": mappings}

    def validate(self, _artifact):
        return {"valid": True, "violations": []}


class _FakeRetriever:
    def retrieve(self, *_args, **_kwargs):
        return [
            EvidenceItem(
                id="candidate:1",
                kind="target_candidate",
                text="aas://asset/submodel/default/element/temp/value",
                score=0.99,
                payload={"source_node": "opcua://ns=2;i=1", "candidate_path": "aas://asset/submodel/default/element/temp/value"},
            )
        ]


class _FakeRules:
    def apply_rules(self, *_args, **_kwargs):
        return [
            Mapping(
                source_path="opcua://ns=2;i=1",
                target_path="aas://asset/submodel/default/element/rules/value",
                mapping_type="equivalent",
                transform=None,
                confidence=0.8,
                rationale="rule mapping",
                evidence=["rule"],
            )
        ]


class _FakeLLM:
    def complete_json(self, *_args, **_kwargs):
        return {
            "mappings": [
                {
                    "source_path": "opcua://ns=2;i=1",
                    "target_path": "aas://asset/submodel/default/element/llm/value",
                    "mapping_type": "equivalent",
                    "transform": None,
                    "confidence": 0.9,
                    "rationale": "llm mapping",
                    "evidence": ["llm"],
                }
            ]
        }


def test_scenarios_change_execution_path(monkeypatch):
    from isynkgr.pipeline import hybrid as hybrid_mod

    monkeypatch.setattr(hybrid_mod, "ADAPTERS", {"opcua": _FakeAdapter("opcua"), "aas": _FakeAdapter("aas")})

    pipeline = HybridPipeline(llm=_FakeLLM(), retriever=_FakeRetriever(), rules=_FakeRules())

    full = pipeline.run("opcua", "aas", source_raw="x", mode="hybrid", config=TranslatorConfig(component_flags={}))
    no_retrieval = pipeline.run(
        "opcua",
        "aas",
        source_raw="x",
        mode="hybrid",
        config=TranslatorConfig(component_flags={"retrieval": False}),
    )

    assert full.provenance.metadata["execution"]["retrieval_ran"] is True
    assert full.provenance.metadata["execution"]["rules_ran"] is True
    assert full.provenance.metadata["execution"]["llm_ran"] is True
    assert no_retrieval.provenance.metadata["execution"]["retrieval_ran"] is False
    full_retrieval = full.provenance.metadata["component_outputs"]["retrieval"]
    no_retrieval_trace = no_retrieval.provenance.metadata["component_outputs"]["retrieval"]
    assert bool(full_retrieval) is True
    assert no_retrieval_trace == {}
