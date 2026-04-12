from __future__ import annotations

import json
from pathlib import Path

from benchmark import orchestrate
from benchmark.full_workflow import _build_pair_dataset
from benchmark.scenarios import CANONICAL_SCENARIOS, SCENARIO_RUNTIME
from isynkgr.canonical.model import CanonicalModel, CanonicalNode
from isynkgr.pipeline import adaptive_candidate_ranker as ranker_mod
from isynkgr.pipeline.adaptive_candidate_ranker import AdaptiveCandidateRankerPipeline, TranslatorConfig
from isynkgr.retrieval.graphrag import GraphRAGRetriever
from isynkgr.rules.engine import RuleEngine


class _LLM:
    def complete_json(self, *_args, **_kwargs):
        return {"mappings": []}


class _Adapter:
    def __init__(self, standard: str, node_id: str, label: str):
        self.standard = standard
        self.node_id = node_id
        self.label = label

    def parse(self, _raw):
        return CanonicalModel(standard=self.standard, nodes=[CanonicalNode(id=self.node_id, type="signal", label=self.label)], edges=[])

    def serialize(self, _model, mappings):
        return {"mappings": mappings}

    def validate(self, _artifact):
        return {"valid": True, "violations": []}


def test_target_path_convention_consistency(tmp_path: Path):
    pair_dir = _build_pair_dataset(tmp_path, "OPCUA", "AAS", max_rows=12)
    gt_rows = [json.loads(line) for line in (pair_dir / "ground_truth.jsonl").read_text().splitlines() if line.strip()]
    for row in gt_rows:
        target = str(row.get("target_path") or "")
        if not target:
            continue
        assert target.startswith("aas://asset-")
        assert "/submodel/default/element/" in target
        assert target.endswith("/value")


def test_scenario_name_consistency_registry():
    assert tuple(SCENARIO_RUNTIME.keys()) == CANONICAL_SCENARIOS


def test_orchestrate_calls_full_workflow(monkeypatch):
    called = {"workflow": 0}

    def _fake_full_workflow():
        called["workflow"] += 1
        return 0

    monkeypatch.setattr(orchestrate, "run_full_workflow", _fake_full_workflow)
    monkeypatch.setattr(orchestrate, "generate_final_report", lambda *_args, **_kwargs: Path("results/final"))
    rc = orchestrate.main()
    assert rc == 0
    assert called["workflow"] == 1


def test_pump_like_false_positive_prevention(monkeypatch):
    monkeypatch.setattr(
        ranker_mod,
        "ADAPTERS",
        {
            "opcua": _Adapter("opcua", "opcua://ns=2;s=Pump0", "Pump0"),
            "aas": _Adapter("aas", "aas://asset-1/submodel/default/element/pressure/value", "pressure"),
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
        ],
    )
    assert result.mappings[0].mapping_type.value == "no_match"
