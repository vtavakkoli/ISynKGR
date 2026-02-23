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
from urllib.parse import urlparse

from benchmark.evaluate import evaluate_run
from isynkgr.icr.mapping_schema import ingest_mapping_payload

SCENARIO_MODE = {
    "baseline": "rule_only",
    "full_framework": "isynkgr_hybrid",
    "ablation_no_graphrag": "llm_only",
    "ablation_no_parallel": "rag_only",
    "ablation_no_community": "graph_only",
    "ablation_no_reasoning": "rule_only",
}


def normalize_ollama_host(raw_host: str) -> str:
    value = (raw_host or "").strip()
    if not value:
        value = "http://host.docker.internal:11434"

    if "://" not in value:
        value = f"http://{value}"

    parsed = urlparse(value)
    host = parsed.hostname or "host.docker.internal"
    port = parsed.port or 11434

    # 0.0.0.0 is a bind address; clients must use an actual reachable host.
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

    # If local hostnames were provided, prefer host-routable Docker host aliases.
    if host in {"localhost", "127.0.0.1"}:
        candidates.append(f"{scheme}://host.docker.internal:{port}")
        if host_ip:
            candidates.append(f"{scheme}://{host_ip}:{port}")

    # De-duplicate while preserving order.
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
            except Exception as exc:  # noqa: BLE001
                print(f"... still waiting ({endpoint}: {exc})", flush=True)
        time.sleep(3)
    raise RuntimeError(f"Timed out waiting for Ollama. Tried: {', '.join(candidates)}")


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run_scenario(args: argparse.Namespace) -> int:
    scenario = args.scenario
    mode = SCENARIO_MODE[scenario]
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
        }
        for idx, row in enumerate(gt_rows)
    ]
    dataset_path.write_text("\n".join(json.dumps(r) for r in dataset_rows) + "\n")

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
    resolved_args = vars(args).copy()
    resolved_args["ollama_host"] = ollama_host
    (out_dir / "config_resolved.json").write_text(json.dumps(resolved_args, indent=2, sort_keys=True))

    if mode in {"isynkgr_hybrid", "llm_only", "rag_only"}:
        ollama_host = wait_for_ollama(ollama_host)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "DATASET_DIR": str(out_dir),
            "OUTPUT_DIR": str(out_dir / "predictions"),
            "CONFIG_PATH": args.config,
            "SUT_MODE": mode,
            "SEED": str(args.seed),
            "MODEL_NAME": args.model_name,
            "MAX_ITEMS": str(dataset_items),
            "TIER": args.tier,
            "OLLAMA_BASE_URL": ollama_host,
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
        print(f"Scenario failed: {scenario}. See {logs_dir / 'run.log'}", flush=True)
        return rc

    metrics = evaluate_run(Path(env["OUTPUT_DIR"]))
    metrics.update({"scenario": scenario, "model": args.model_name, "seed": args.seed, "tier": args.tier, "items": dataset_items})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIO_MODE))
    parser.add_argument("--config", default="benchmark/config.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434"))
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "qwen3:0.6b"))
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    parser.add_argument("--max-items", type=int, default=int(os.getenv("MAX_ITEMS", "100")))
    parser.add_argument("--tier", default=os.getenv("TIER", "canonical"))
    args = parser.parse_args()
    print(
        f"[BANNER] service=run scenario={args.scenario} ts_utc={datetime.now(timezone.utc).isoformat()} "
        f"model={args.model_name} seed={args.seed} tier={args.tier} item_count={args.max_items} config={args.config}",
        flush=True,
    )
    return run_scenario(args)


if __name__ == "__main__":
    raise SystemExit(main())
