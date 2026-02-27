import argparse
import json
from pathlib import Path

from isynkgr.evaluation import benchmark_runner


class FakeRetriever:
    def __init__(self, k_hop: int) -> None:
        self.k_hop = k_hop

    def retrieve(self, graph: dict, terms: list[str], top_k: int) -> dict:
        return {"nodes": graph["nodes"], "stats": {"retrieved_edges": 3}}


class FakeLibrary:
    def __init__(self) -> None:
        self.saved = []

    def save_rule(self, *args, **kwargs):
        self.saved.append((args, kwargs))


class FakeClient:
    def complete_json(self, prompt: str, schema_name: str, seed: int) -> dict:
        return {"eval_count": 10, "prompt_eval_count": 2}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_run_pair_uses_fake_dependencies(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "prompts/v1").mkdir(parents=True)
    (tmp_path / "prompts/v1/reasoning_check.txt").write_text("{source_standard}:{target_standard}:{candidate_target}")

    sample = {
        "sample_id": "ieee1451_1",
        "standard": "ieee1451",
        "terms": ["temperature"],
        "entities": [{"id": "ieee1451.Sensor"}],
        "properties": [{"name": "reading"}],
    }
    _write_jsonl(tmp_path / "data/samples/ieee1451/samples_100.jsonl", [sample])
    _write_jsonl(tmp_path / "data/ground_truth/ieee1451__to__opcua62541/gt.jsonl", [{"sample_id": "ieee1451_1", "target_entity": "opcua62541.Sensor"}])

    fake_lib = FakeLibrary()
    monkeypatch.setattr(benchmark_runner, "GraphRAGRetriever", FakeRetriever)
    monkeypatch.setattr(benchmark_runner, "TranslationLogicLibrary", lambda: fake_lib)

    args = argparse.Namespace(model="mock", max_samples=1)
    rows = benchmark_runner.run_pair(
        source="ieee1451",
        target="opcua62541",
        args=args,
        client=FakeClient(),
        pair_index=1,
        pair_total=1,
        system_status={"ieee1451": 0, "opcua62541": 0},
    )

    assert len(rows) == 5
    assert any(r["method"] == "isynkgr" and r["token_count_avg"] == 12 for r in rows)
    assert len(fake_lib.saved) == 1
