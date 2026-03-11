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
        "qwen3.5:0.8b",
        "--tier",
        "canonical",
    ]
    print(f"[SAMPLE] scenario={scenario} samples=5 out={out_dir}", flush=True)
    return subprocess.run(cmd).returncode


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _print_sample_validity(scenario: str, out_dir: Path) -> tuple[int, int]:
    predictions_dir = out_dir / "predictions" / "predictions"
    sample_rows = _load_jsonl(predictions_dir / "sample_results.jsonl")
    llm_rows = {row.get("sample"): row for row in _load_jsonl(predictions_dir / "llm_trace.jsonl") if row.get("sample")}

    valid_count = 0
    for row in sample_rows:
        sample = str(row.get("sample", "<unknown>"))
        matched = bool(row.get("matched"))
        trace = llm_rows.get(sample, {})
        expected_target = trace.get("expected_target_path") or "<none>"
        predicted_target = ((trace.get("predicted_top") or {}).get("target_path")) or "<none>"
        if matched:
            valid_count += 1
            reason = f"target matched expected target ({predicted_target})"
        else:
            reason = f"predicted target ({predicted_target}) != expected target ({expected_target})"
        print(f"[SAMPLE] scenario={scenario} sample={sample} valid={matched} reason={reason}", flush=True)

    return valid_count, len(sample_rows)


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
        f1 = float(metrics.get("f1", 0.0))
        matched_count = int(metrics.get("matched_count", 0))
        sample_valid_count, sample_total = _print_sample_validity(scenario, out_dir)
        sample_match_rate = (sample_valid_count / sample_total) if sample_total else 0.0
        checks = {
            "gt_count == 5": metrics.get("gt_count") == 5,
            "pred_count == 5": metrics.get("pred_count") == 5,
            "dataset_count == 5": metrics.get("dataset_count") == 5,
            "sample target match rate >= 0.20": sample_match_rate >= 0.20,
        }
        if scenario in {"full_framework", "ablation_no_graphrag", "ablation_no_parallel"}:
            checks["matched_count > 0"] = matched_count > 0
            checks["target paths benchmark-shaped"] = float(metrics.get("benchmark_target_shape_rate", 0.0)) >= 0.80
        failed = [name for name, ok in checks.items() if not ok]
        print(
            f"[SAMPLE] scenario={scenario} gt_path={metrics.get('gt_path_used')} "
            f"pred_path={metrics.get('pred_path_used')} metrics_path={out_dir / 'metrics.json'} "
            f"f1={f1:.3f} sample_target_match_rate={sample_match_rate:.3f}",
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
