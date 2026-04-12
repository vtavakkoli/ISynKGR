from benchmark import orchestrate


def test_full_run_returns_non_zero_on_failure(monkeypatch):
    monkeypatch.setattr(orchestrate, "run_full_workflow", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    rc = orchestrate.main()
    assert rc == 1
