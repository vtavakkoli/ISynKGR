from __future__ import annotations

from dataclasses import dataclass

from isynkgr.canonical.model import CanonicalModel, CanonicalNode
from isynkgr.canonical.schemas import EvidenceItem
from isynkgr.pipeline.hybrid import HybridPipeline, TranslatorConfig
from isynkgr.pipeline.prompting import build_mapping_prompt
from isynkgr.rules.engine import RuleEngine


@dataclass
class _FakeAdapter:
    standard: str

    def parse(self, _raw):
        return CanonicalModel(
            standard=self.standard,
            nodes=[CanonicalNode(id="opcua://ns=2;i=1000", type="signal", label="Pump0")],
            edges=[],
        )

    def serialize(self, _model, mappings):
        return {"mappings": mappings}

    def validate(self, _artifact):
        return {"valid": True, "violations": []}


class _FakeRetriever:
    def retrieve(self, *_args, **_kwargs):
        return []


class _InventingLLM:
    def complete_json(self, *_args, **_kwargs):
        return {
            "mappings": [
                {
                    "source_path": "opcua://ns=2;i=1000",
                    "target_path": "aas://asset/submodel/default/element/pressure/value",
                    "mapping_type": "transform",
                    "transform": {"op": "identity", "args": {}},
                    "confidence": 0.8,
                    "rationale": "Invented path from model.",
                    "evidence": [],
                }
            ]
        }


def test_prompt_prefers_exact_target_candidates() -> None:
    source = CanonicalModel(standard="opcua", nodes=[CanonicalNode(id="opcua://ns=2;i=1000", type="signal", label="Pump0")], edges=[])
    evidence = [
        EvidenceItem(
            id="candidate:aas://aas-0/submodel/default/element/value",
            kind="target_candidate",
            text="aas://aas-0/submodel/default/element/value",
            score=1.0,
            payload={"candidate_path": "aas://aas-0/submodel/default/element/value"},
        )
    ]
    prompt = build_mapping_prompt(
        source_protocol="opcua",
        target_protocol="aas",
        source_schema_summary={"standard": "opcua"},
        target_schema_summary={"standard": "aas"},
        source_model=source,
        evidence=evidence,
    )
    assert "Choose target_path exactly from TARGET_VARIABLES" in prompt
    assert "Do not invent a new target_path" in prompt


def test_invented_paths_snap_to_exact_candidate(monkeypatch) -> None:
    from isynkgr.pipeline import hybrid as hybrid_mod

    monkeypatch.setattr(hybrid_mod, "ADAPTERS", {"opcua": _FakeAdapter("opcua"), "aas": _FakeAdapter("aas"), "iec61499": _FakeAdapter("iec61499")})
    pipeline = HybridPipeline(llm=_InventingLLM(), retriever=_FakeRetriever(), rules=RuleEngine())
    result = pipeline.run(
        "opcua",
        "aas",
        source_raw="x",
        mode="llm_only",
        config=TranslatorConfig(seed=7),
        target_candidates=["aas://aas-0/submodel/default/element/value"],
    )
    assert result.mappings[0].target_path == "aas://aas-0/submodel/default/element/value"

