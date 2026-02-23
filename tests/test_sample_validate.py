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
