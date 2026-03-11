import json
from pathlib import Path

from benchmark import sample_validate


def test_sample_validate_runs_all_scenarios_with_five_samples(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "benchmark").mkdir()
    (tmp_path / "benchmark" / "config.json").write_text("{}")

    (tmp_path / "datasets" / "v1" / "opcua" / "synthetic").mkdir(parents=True)
    (tmp_path / "datasets" / "v1" / "aas" / "synthetic").mkdir(parents=True)
    (tmp_path / "datasets" / "v1" / "opcua" / "synthetic" / "opcua_000.xml").write_text(
        """<UANodeSet><UAObjectType NodeId='ns=1;i=1' BrowseName='A'><DisplayName>A</DisplayName></UAObjectType></UANodeSet>"""
    )
    (tmp_path / "datasets" / "v1" / "aas" / "synthetic" / "aas_000.json").write_text(json.dumps({"assetAdministrationShells": [{"id": "aas-1", "submodels": [{"keys": [{"value": "sm-1"}]}]}], "submodels": [{"id": "sm-1", "submodelElements": []}]}))

    monkeypatch.setattr(sample_validate, "validate_or_generate", lambda *_args, **_kwargs: {})

    calls = []

    class _Proc:
        def __init__(self, code: int = 0):
            self.returncode = code

    def _fake_run(cmd):
        calls.append(cmd)
        out_dir = Path(cmd[cmd.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "gt_count": 5,
                    "pred_count": 5,
                    "dataset_count": 5,
                    "f1": 1.0,
                        "matched_count": 1,
                        "benchmark_target_shape_rate": 1.0,
                    "gt_path_used": str(out_dir / "ground_truth.jsonl"),
                    "pred_path_used": str(out_dir / "predictions" / "mappings.jsonl"),
                }
            )
        )
        return _Proc(0)

    monkeypatch.setattr(sample_validate.subprocess, "run", _fake_run)

    rc = sample_validate.main()

    assert rc == 0
    assert len(calls) == len(sample_validate.SCENARIO_MODE)
    for cmd in calls:
        assert cmd[0:3] == [sample_validate.sys.executable, "-u", "-m"]
        assert cmd[cmd.index("--max-items") + 1] == "5"


def test_sample_validate_fails_on_exact_match_regression(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "benchmark").mkdir()
    (tmp_path / "benchmark" / "config.json").write_text("{}")

    (tmp_path / "datasets" / "v1" / "opcua" / "synthetic").mkdir(parents=True)
    (tmp_path / "datasets" / "v1" / "aas" / "synthetic").mkdir(parents=True)
    (tmp_path / "datasets" / "v1" / "opcua" / "synthetic" / "opcua_000.xml").write_text(
        """<UANodeSet><UAObjectType NodeId='ns=1;i=1' BrowseName='A'><DisplayName>A</DisplayName></UAObjectType></UANodeSet>"""
    )
    (tmp_path / "datasets" / "v1" / "aas" / "synthetic" / "aas_000.json").write_text(json.dumps({"assetAdministrationShells": [{"id": "aas-1", "submodels": [{"keys": [{"value": "sm-1"}]}]}], "submodels": [{"id": "sm-1", "submodelElements": []}]}))

    monkeypatch.setattr(sample_validate, "validate_or_generate", lambda *_args, **_kwargs: {})

    class _Proc:
        def __init__(self, code: int = 0):
            self.returncode = code

    def _fake_run(cmd):
        out_dir = Path(cmd[cmd.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "gt_count": 5,
                    "pred_count": 5,
                    "dataset_count": 5,
                    "matched_count": 0,
                    "benchmark_target_shape_rate": 0.0,
                    "f1": 0.0,
                    "gt_path_used": str(out_dir / "ground_truth.jsonl"),
                    "pred_path_used": str(out_dir / "predictions" / "mappings.jsonl"),
                }
            )
        )
        return _Proc(0)

    monkeypatch.setattr(sample_validate.subprocess, "run", _fake_run)

    rc = sample_validate.main()
    assert rc == 1


def test_sample_validate_prefers_sample_match_rate_when_sample_artifacts_exist(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "benchmark").mkdir()
    (tmp_path / "benchmark" / "config.json").write_text("{}")

    (tmp_path / "datasets" / "v1" / "opcua" / "synthetic").mkdir(parents=True)
    (tmp_path / "datasets" / "v1" / "aas" / "synthetic").mkdir(parents=True)
    (tmp_path / "datasets" / "v1" / "opcua" / "synthetic" / "opcua_000.xml").write_text(
        """<UANodeSet><UAObjectType NodeId='ns=1;i=1' BrowseName='A'><DisplayName>A</DisplayName></UAObjectType></UANodeSet>"""
    )
    (tmp_path / "datasets" / "v1" / "aas" / "synthetic" / "aas_000.json").write_text(json.dumps({"assetAdministrationShells": [{"id": "aas-1", "submodels": [{"keys": [{"value": "sm-1"}]}]}], "submodels": [{"id": "sm-1", "submodelElements": []}]}))

    monkeypatch.setattr(sample_validate, "validate_or_generate", lambda *_args, **_kwargs: {})

    class _Proc:
        def __init__(self, code: int = 0):
            self.returncode = code

    def _fake_run(cmd):
        out_dir = Path(cmd[cmd.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "gt_count": 5,
                    "pred_count": 5,
                    "dataset_count": 5,
                    "f1": 0.0,
                    "matched_count": 1,
                    "benchmark_target_shape_rate": 1.0,
                    "gt_path_used": str(out_dir / "ground_truth.jsonl"),
                    "pred_path_used": str(out_dir / "predictions" / "mappings.jsonl"),
                }
            )
        )

        pred_dir = out_dir / "predictions" / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        (pred_dir / "sample_results.jsonl").write_text(json.dumps({"sample": "opcua_000.xml", "matched": True}) + "\n")
        (pred_dir / "llm_trace.jsonl").write_text(
            json.dumps(
                {
                    "sample": "opcua_000.xml",
                    "expected_target_path": "aas://aas-0/submodel/default/element/value",
                    "predicted_top": {"target_path": "aas://aas-0/submodel/default/element/value"},
                }
            )
            + "\n"
        )
        return _Proc(0)

    monkeypatch.setattr(sample_validate.subprocess, "run", _fake_run)

    rc = sample_validate.main()
    assert rc == 0
