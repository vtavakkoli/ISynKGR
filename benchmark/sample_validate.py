from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from benchmark.run import SCENARIO_MODE
from benchmark.validate_dataset import validate_or_generate
from isynkgr.adapters.aas import AASAdapter
from isynkgr.adapters.opcua import OPCUAAdapter


def _validate_fixture_parsing() -> None:
    print("[SAMPLE] fixture validation: start", flush=True)
    validate_or_generate(Path("datasets/v1"))

    opcua_fixture = Path("datasets/v1/opcua/synthetic/opcua_000.xml")
    aas_fixture = Path("datasets/v1/aas/synthetic/aas_000.json")

    opcua_raw = opcua_fixture.read_text()
    aas_raw = json.loads(aas_fixture.read_text())

    opcua_adapter = OPCUAAdapter()
    aas_adapter = AASAdapter()

    opcua_model = opcua_adapter.parse(opcua_raw)
    aas_model = aas_adapter.parse(aas_raw)
    opcua_validation = opcua_adapter.validate(opcua_raw)
    aas_validation = aas_adapter.validate(aas_raw)

    if not opcua_model.nodes:
        raise RuntimeError(f"OPCUA parse returned no nodes: {opcua_fixture}")
    if not aas_model.nodes:
        raise RuntimeError(f"AAS parse returned no nodes: {aas_fixture}")
    if not opcua_validation.valid:
        raise RuntimeError(f"OPCUA fixture validation failed: {opcua_fixture} violations={len(opcua_validation.violations)}")
    if not aas_validation.valid:
        raise RuntimeError(f"AAS fixture validation failed: {aas_fixture} violations={len(aas_validation.violations)}")

    print(
        "[SAMPLE] fixture validation: done "
        f"opcua_nodes={len(opcua_model.nodes)} aas_nodes={len(aas_model.nodes)} "
        f"opcua_fixture={opcua_fixture} aas_fixture={aas_fixture}",
        flush=True,
    )


def _run_scenario_samples(scenario: str, out_dir: Path) -> int:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "benchmark.run",
        "--scenario",
        scenario,
        "--config",
        "benchmark/config.json",
        "--out",
        str(out_dir),
        "--max-items",
        "5",
        "--model-name",
        "qwen3:0.6b",
        "--tier",
        "canonical",
    ]
    print(f"[SAMPLE] scenario={scenario} samples=5 out={out_dir}", flush=True)
    return subprocess.run(cmd).returncode


def main() -> int:
    try:
        _validate_fixture_parsing()
    except Exception as exc:  # noqa: BLE001
        print(f"[SAMPLE] fixture validation failed: {exc}", flush=True)
        return 1

    failures: list[str] = []
    for scenario in sorted(SCENARIO_MODE):
        out_dir = Path("results/sample_validation") / scenario
        out_dir.mkdir(parents=True, exist_ok=True)
        rc = _run_scenario_samples(scenario, out_dir)
        if rc != 0:
            failures.append(scenario)
            print(f"[SAMPLE] scenario failed: {scenario} rc={rc}", flush=True)
            continue

        metrics = json.loads((out_dir / "metrics.json").read_text())
        checks = {
            "gt_count == 5": metrics.get("gt_count") == 5,
            "pred_count == 5": metrics.get("pred_count") == 5,
            "dataset_count == 5": metrics.get("dataset_count") == 5,
            "f1 present": isinstance(metrics.get("f1"), (int, float)),
        }
        failed = [name for name, ok in checks.items() if not ok]
        print(
            f"[SAMPLE] scenario={scenario} gt_path={metrics.get('gt_path_used')} "
            f"pred_path={metrics.get('pred_path_used')} metrics_path={out_dir / 'metrics.json'}",
            flush=True,
        )
        if failed:
            failures.append(scenario)
            print(f"[SAMPLE] scenario validation failed: {scenario}: {', '.join(failed)}", flush=True)

    if failures:
        print(f"[SAMPLE] Validation failed scenarios={','.join(failures)}", flush=True)
        return 1

    print("[SAMPLE] Validation passed for all scenarios.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
