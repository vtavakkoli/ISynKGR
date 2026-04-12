from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request
from urllib.error import HTTPError
from urllib.parse import urlparse

from benchmark.evaluate import evaluate_run
from benchmark.scenarios import CANONICAL_SCENARIOS, DEPRECATED_SCENARIO_ALIASES, SCENARIO_RUNTIME, resolve_scenario_name
from isynkgr.icr.mapping_schema import ingest_mapping_payload

# Backward-compatible mapping used by validation utilities/tests that import SCENARIO_MODE.
SCENARIO_MODE = {name: runtime.mode for name, runtime in SCENARIO_RUNTIME.items()}


def normalize_ollama_host(raw_host: str) -> str:
    value = (raw_host or "").strip() or "http://host.docker.internal:11434"
    if "://" not in value:
        value = f"http://{value}"
    parsed = urlparse(value)
    host = parsed.hostname or "host.docker.internal"
    port = parsed.port or 11434
    if host == "0.0.0.0":
        host = os.getenv("OLLAMA_HOST_IP", "host.docker.internal")
    return f"{parsed.scheme or 'http'}://{host}:{port}"


def _candidate_ollama_hosts(base_url: str) -> list[str]:
    primary = normalize_ollama_host(base_url)
    parsed = urlparse(primary)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "host.docker.internal"
    port = parsed.port or 11434

    candidates: list[str] = [f"{scheme}://{host}:{port}"]
    host_ip = (os.getenv("OLLAMA_HOST_IP") or "").strip()
    if host_ip:
        candidates.append(f"{scheme}://{host_ip}:{port}")
    if host in {"localhost", "127.0.0.1"}:
        candidates.append(f"{scheme}://host.docker.internal:{port}")
        if host_ip:
            candidates.append(f"{scheme}://{host_ip}:{port}")

    out: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def wait_for_ollama(base_url: str, timeout_s: int = 90) -> str:
    candidates = _candidate_ollama_hosts(base_url)
    print(f"Waiting for Ollama endpoints: {', '.join(candidates)} ...", flush=True)
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        for endpoint in candidates:
            url = endpoint.rstrip("/") + "/api/tags"
            try:
                with request.urlopen(url, timeout=3) as resp:
                    if resp.status == 200:
                        print(f"Ollama ready at {endpoint}.", flush=True)
                        return endpoint
            except HTTPError as exc:
                if exc.code in {401, 403, 405}:
                    print(f"Ollama reachable at {endpoint} (status={exc.code}); proceeding.", flush=True)
                    return endpoint
            except Exception:
                pass
        time.sleep(3)
    raise RuntimeError(f"Timed out waiting for Ollama. Tried: {', '.join(candidates)}")


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _cardinality_contract_for_sample(gt_row: dict) -> dict:
    grouped_1 = bool(gt_row.get("grouped_1") or gt_row.get("metadata", {}).get("grouped_1"))
    mode = "grouped_1" if grouped_1 else "one_to_one"
    expected_count = int(gt_row.get("expected_count", 1 if mode == "one_to_one" else 0))
    return {"mode": mode, "expected_count": expected_count, "grouped_1": grouped_1}


def run_scenario(args: argparse.Namespace) -> int:
    scenario, is_deprecated = resolve_scenario_name(args.scenario)
    if is_deprecated:
        print(f"[DEPRECATION] Scenario '{args.scenario}' is deprecated; using '{scenario}'.", flush=True)

    scenario_cfg = SCENARIO_RUNTIME[scenario]
    mode = scenario_cfg.mode
    component_flags = scenario_cfg.component_flags

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    dataset_items = int(args.max_items)
    dataset_path = out_dir / "dataset.jsonl"
    gt_path = out_dir / "ground_truth.jsonl"
    gt_source = Path("datasets/v1/crosswalk/gt_mappings.jsonl")
    gt_rows = [ingest_mapping_payload(json.loads(line), migrate_legacy=True).model_dump() for line in gt_source.read_text().splitlines() if line.strip()][:dataset_items]
    gt_path.write_text("\n".join(json.dumps(r) for r in gt_rows) + "\n")
    dataset_rows = [
        {
            "id": row["source_path"],
            "mapping_source_path": row["source_path"],
            "target_path": row["target_path"],
            "source_standard": "OPCUA",
            "target_standard": "AAS",
            "tier": args.tier,
            "source_path": str(Path("datasets/v1/opcua/synthetic") / f"opcua_{idx:03d}.xml"),
            "cardinality_contract": _cardinality_contract_for_sample(row),
        }
        for idx, row in enumerate(gt_rows)
    ]
    dataset_path.write_text("\n".join(json.dumps(r) for r in dataset_rows) + "\n")
    output_dir = Path(out_dir / "predictions")

    header = {
        "git_commit": _git_hash(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model_name,
        "seed": args.seed,
        "tier": args.tier,
        "item_count": dataset_items,
        "scenario": scenario,
        "mode": mode,
    }
    (logs_dir / "run.log").write_text(json.dumps(header) + "\n")
    ollama_host = normalize_ollama_host(args.ollama_host)

    if mode in {"adaptive_candidate_ranker", "llm_only", "rag_only"}:
        timeout_s = int(os.getenv("OLLAMA_READY_TIMEOUT_S", "120"))
        ollama_host = wait_for_ollama(ollama_host, timeout_s=timeout_s)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "DATASET_DIR": str(out_dir),
            "OUTPUT_DIR": str(output_dir),
            "CONFIG_PATH": args.config,
            "SUT_MODE": mode,
            "SEED": str(args.seed),
            "MODEL_NAME": args.model_name,
            "MAX_ITEMS": str(dataset_items),
            "TIER": args.tier,
            "OLLAMA_BASE_URL": ollama_host,
            "COMPONENT_FLAGS": json.dumps(component_flags),
            "ALLOW_TARGET_HINTS": "0",
        }
    )
    Path(env["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    with (logs_dir / "run.log").open("a") as fp:
        proc = subprocess.Popen([sys.executable, "-u", "-m", "benchmark.run_sut"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            fp.write(line)
            fp.flush()
        rc = proc.wait()
    if rc != 0:
        return rc

    metrics = evaluate_run(Path(env["OUTPUT_DIR"]))
    metrics.update({"scenario": scenario, "model": args.model_name, "seed": args.seed, "tier": args.tier, "items": dataset_items})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=sorted(set(CANONICAL_SCENARIOS) | set(DEPRECATED_SCENARIO_ALIASES)))
    parser.add_argument("--config", default="benchmark/config.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434"))
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "gemma4:e2b"))
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    parser.add_argument("--max-items", type=int, default=int(os.getenv("MAX_ITEMS", "100")))
    parser.add_argument("--tier", default=os.getenv("TIER", "canonical"))
    args = parser.parse_args()
    return run_scenario(args)


if __name__ == "__main__":
    raise SystemExit(main())
