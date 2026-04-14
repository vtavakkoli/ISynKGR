from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from benchmark.evaluate import evaluate_run
from benchmark.metrics import mean_std_ci
from benchmark.report import write_report
from benchmark.scenarios import CANONICAL_SCENARIOS, COMPONENT_FLAGS
from benchmark.validate_dataset import validate_or_generate
from isynkgr.icr.mapping_schema import ingest_mapping_payload
from isynkgr.pipeline.adaptive_candidate_ranker import ADAPTERS

DEFAULT_SEEDS = [11, 23, 37]
DEFAULT_RUNS_PER_PAIR = 20


def _now_run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def _resolve_seeds(cfg: dict) -> list[int]:
    env_seeds = os.getenv("BENCHMARK_SEEDS", "").strip()
    if env_seeds:
        seeds = [int(token.strip()) for token in env_seeds.split(",") if token.strip()]
        if not seeds:
            raise ValueError("BENCHMARK_SEEDS is set but no valid integer seed values were provided.")
        return seeds

    runs_per_pair = int(os.getenv("RUNS_PER_PAIR", str(cfg.get("runs_per_pair", DEFAULT_RUNS_PER_PAIR))))
    if runs_per_pair <= 0:
        raise ValueError("RUNS_PER_PAIR must be a positive integer.")

    seeds = list(DEFAULT_SEEDS)
    while len(seeds) < runs_per_pair:
        seeds.append(seeds[-1] + 17)
    return seeds[:runs_per_pair]


def _artifact_paths(run_id: str) -> tuple[Path, Path]:
    artifacts_dir = Path("artifacts") / run_id
    compat_dir = Path("results") / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    compat_dir.parent.mkdir(parents=True, exist_ok=True)
    if not compat_dir.exists():
        try:
            compat_dir.symlink_to(Path("..") / "artifacts" / run_id, target_is_directory=True)
        except OSError:
            compat_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir, compat_dir


def _pair_key(source: str, target: str) -> str:
    return f"{source.upper()}__TO__{target.upper()}"


def _pair_supported(source: str, target: str) -> tuple[bool, str]:
    src = source.lower()
    tgt = target.lower()
    if src not in ADAPTERS:
        return False, f"source adapter '{source}' not available"
    if tgt not in ADAPTERS:
        return False, f"target adapter '{target}' not available"
    return True, ""


def _source_fixture_path(source_standard: str, idx: int, source_dir: Path) -> Path:
    src = source_standard.upper()
    if src == "OPCUA":
        return Path("datasets/v1/opcua/synthetic") / f"opcua_{idx % 100:03d}.xml"
    if src == "AAS":
        return Path("datasets/v1/aas/synthetic") / f"aas_{idx % 100:03d}.json"
    source_file = source_dir / f"sample_{idx:04d}_{src.lower()}.json"
    payload = {"standard": src}
    if src == "IEC61499":
        payload |= {
            "devices": [
                {
                    "id": "Device0",
                    "name": "MainDevice",
                    "resources": [
                        {
                            "id": "Res1",
                            "name": "Resource1",
                            "function_blocks": [
                                {
                                    "id": "FB1",
                                    "name": "TelemetryFB",
                                    "type": "Telemetry",
                                    "inputs": [
                                        {"id": "SetPoint", "dtype": "FLOAT", "unit": "C", "range": {"min": -20, "max": 120}}
                                    ],
                                    "outputs": [
                                        {"id": "Pressure0", "name": "Pressure0", "dtype": "FLOAT", "unit": "bar", "range": {"min": 0, "max": 25}},
                                        {"id": "PumpState", "name": "PumpState", "dtype": "STRING"},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    elif src == "IEEE1451":
        payload |= {
            "teds": [
                {
                    "id": "teds0",
                    "name": "SensorTEDS",
                    "channels": [
                        {"id": "Channel0", "name": "Measurement", "dtype": "FLOAT", "unit": "bar", "range": {"min": 0, "max": 25}},
                        {"id": "Channel1", "name": "PumpState", "dtype": "STRING"},
                    ],
                }
            ]
        }
    else:
        payload |= {
            "classes": [{"id": "Class0", "label": "Pressure0"}],
            "relations": [{"source": "Class0", "target": "Class0Value", "type": "hasValue"}],
        }
    source_file.write_text(json.dumps(payload))
    return source_file


def _peek_opcua_variable(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        root = ET.fromstring(path.read_text())
    except Exception:
        return None
    for elem in root.iter():
        if elem.tag.split("}")[-1] != "UAVariable":
            continue
        browse_name = elem.attrib.get("BrowseName", "")
        if browse_name:
            _, _, name = browse_name.partition(":")
            return name or browse_name
    return None


def _peek_aas_element(path: Path) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, None
    try:
        doc = json.loads(path.read_text())
    except Exception:
        return None, None
    for sm in doc.get("submodels", []):
        sid = sm.get("id", "default")
        for elem in sm.get("submodelElements", []):
            eid = elem.get("idShort")
            if eid:
                return sid, eid
    return None, None


def _normalize_signal_hint(signal: str | None, idx: int) -> str:
    if not signal:
        return f"signal{idx}"
    lowered = signal.lower()
    for token in ("temperature", "pressure", "flow", "speed", "state", "vibration"):
        if token in lowered:
            return token
    return re.sub(r"\d+$", "", lowered) or lowered


def _synthetic_id_for_standard(standard: str, idx: int, default: str, source_path: Path | None = None, signal_hint: str | None = None) -> str:
    signal = _normalize_signal_hint(signal_hint, idx)
    s = standard.upper()
    if s == "OPCUA":
        raw_name = _peek_opcua_variable(source_path) if source_path else None
        if raw_name:
            return f"opcua://ns=2;s={raw_name}"
        return f"opcua://ns=2;s={signal.capitalize()}{idx}"
    if s == "AAS":
        _, element_id = _peek_aas_element(source_path) if source_path else (None, None)
        if element_id:
            element = _normalize_signal_hint(element_id, idx)
            return f"aas://asset-{idx}/submodel/default/element/{element}/value"
        return f"aas://asset-{idx}/submodel/default/element/{signal}/value"
    if s == "IEEE1451":
        channel_id = "Channel1" if signal == "state" else "Channel0"
        return f"ieee1451://teds0/{channel_id}/value"
    if s == "IEC61499":
        variable_id = "PumpState" if signal == "state" else "Pressure0"
        return f"iec61499://Device0/Res1/FB1/{variable_id}"
    if s == "ISO15926":
        return f"iso15926://class/Class{idx}"
    return default


def _build_pair_dataset(artifacts_dir: Path, source_standard: str, target_standard: str, max_rows: int) -> Path:
    pair_dir = artifacts_dir / "pairs" / _pair_key(source_standard, target_standard)
    pair_dir.mkdir(parents=True, exist_ok=True)
    source_dir = pair_dir / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)

    gt_src = Path("datasets/v1/crosswalk/gt_mappings.jsonl")
    rows: list[dict] = []
    gt_rows: list[dict] = []
    target_universe: list[str] = []
    seen_targets: set[str] = set()
    tiers = ["synthetic", "noisy", "realistic"]
    difficulties = ["easy", "medium", "hard"]

    for i, line in enumerate(gt_src.read_text().splitlines()):
        if not line.strip():
            continue
        rec = ingest_mapping_payload(json.loads(line), migrate_legacy=True).model_dump()
        source_fixture = _source_fixture_path(source_standard, i, source_dir)
        if source_standard.upper() == "OPCUA":
            source_signal_hint = _peek_opcua_variable(source_fixture)
        elif source_standard.upper() == "AAS":
            _, source_signal_hint = _peek_aas_element(source_fixture)
        else:
            source_signal_hint = None
        source_id = _synthetic_id_for_standard(source_standard, i, rec["source_path"], source_fixture, source_signal_hint)
        target_id = _synthetic_id_for_standard(target_standard, i, rec["target_path"], None, source_signal_hint)
        if target_id and target_id not in seen_targets:
            seen_targets.add(target_id)
            target_universe.append(target_id)
        if i >= max_rows:
            continue
        gt_rows.append(
            rec
            | {
                "source_path": source_id,
                "target_path": target_id,
                "mapping_type": rec.get("mapping_type", "equivalent"),
            }
        )
        signal_hint = source_signal_hint or (source_id.rsplit("/", 1)[-1] if "/" in source_id else source_id.split("=")[-1])
        context_id = f"asset-{i % 17}"
        rows.append(
            {
                "id": source_id,
                "mapping_source_path": source_id,
                "target_path": target_id,
                "source_standard": source_standard,
                "target_standard": target_standard,
                "pair": f"{source_standard}->{target_standard}",
                "tier": tiers[i % len(tiers)],
                "difficulty": difficulties[i % len(difficulties)],
                "source_path": str(source_fixture),
                "source_record": {
                    "variable_role": "measurement",
                    "datatype": "FLOAT" if signal_hint.lower() not in {"state"} else "STRING",
                    "unit": "bar" if "pressure" in signal_hint.lower() else ("C" if "temp" in signal_hint.lower() else ""),
                    "context_entity_id": context_id,
                    "description": f"{signal_hint} measurement for {context_id}",
                },
                "target_candidates": [
                    t
                    for t in target_universe
                    if (signal_hint.lower() in t.lower()) or (context_id in t)
                ][:5],
                "cardinality_contract": {"mode": "one_to_one", "grouped_1": False, "expected_count": 1},
            }
        )

    for row in rows:
        if row.get("target_path") and row["target_path"] not in row.get("target_candidates", []):
            row.setdefault("target_candidates", []).insert(0, row["target_path"])

    (pair_dir / "dataset.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (pair_dir / "ground_truth.jsonl").write_text("\n".join(json.dumps(r) for r in gt_rows) + "\n")
    (pair_dir / "target_candidates.jsonl").write_text(
        "\n".join(json.dumps({"target_path": target}) for target in target_universe) + ("\n" if target_universe else "")
    )
    return pair_dir


def _run_variant(variant_name: str, pair_dir: Path, cfg_path: Path, logs_dir: Path, seed: int, source_standard: str, target_standard: str) -> tuple[dict, float]:
    out_dir = pair_dir / "results" / variant_name / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_src = pair_dir / "ground_truth.jsonl"
    if gt_src.exists():
        (out_dir / "ground_truth.jsonl").write_text(gt_src.read_text())
    env = os.environ.copy()
    env.update(
        {
            "DATASET_DIR": str(pair_dir.resolve()),
            "OUTPUT_DIR": str(out_dir.resolve()),
            "CONFIG_PATH": str(cfg_path.resolve()),
            "SUT_MODE": "embedding_only" if variant_name == "embedding_similarity" else ("adaptive_candidate_ranker" if variant_name == "full_framework" or variant_name.startswith("ablation_") else variant_name),
            "SEED": str(seed),
            "MAX_ITEMS": str(int(os.getenv("MAX_ITEMS", "100"))),
            "COMPONENT_FLAGS": json.dumps(COMPONENT_FLAGS.get(variant_name, {})),
            "SOURCE_PROTOCOL": source_standard.lower(),
            "TARGET_PROTOCOL": target_standard.lower(),
            "ALLOW_SYNTHETIC_SHORTCUTS": "0",
        }
    )

    start = time.perf_counter()
    log_path = logs_dir / f"{_pair_key(source_standard, target_standard)}_{variant_name}_seed{seed}.log"
    with log_path.open("w") as fp:
        proc = subprocess.Popen(["python", "-u", "-m", "benchmark.run_sut"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            fp.write(line)
        proc.wait()
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"variant {variant_name} seed {seed} failed for {_pair_key(source_standard, target_standard)}")

    # Use sample-level Top-1 scoring for the canonical workflow headline metric.
    # Exact triple matching is still exported separately as exact_mapping_* for
    # diagnostics, but can be misleading when the source fixture contains multiple
    # variable-level nodes and the GT row is sample-oriented.
    metrics = evaluate_run(out_dir, evaluation_mode="sample_top1")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    metrics["baseline"] = variant_name
    metrics["seed"] = seed
    metrics["pair"] = f"{source_standard}->{target_standard}"
    metrics["time_s"] = elapsed

    results_link = Path("results") / _pair_key(source_standard, target_standard) / variant_name / f"seed{seed}"
    results_link.mkdir(parents=True, exist_ok=True)
    for f in ["metrics.json", "error_analysis.json", "error_summary.json", "retrieval_diagnostics.json", "strategy_usage.json"]:
        src = out_dir / f
        if src.exists():
            (results_link / f).write_text(src.read_text())
    return metrics, elapsed


def _measure_robustness(rows: list[dict]) -> dict:
    by_variant_pair: dict[str, list[dict]] = {}
    for row in rows:
        key = f"{row['pair']}::{row['baseline']}"
        by_variant_pair.setdefault(key, []).append(row)
    out: dict[str, dict] = {}
    for key, runs in by_variant_pair.items():
        f1s = [float(r.get("f1", 0.0)) for r in runs]
        rec1 = [float(r.get("retrieval_recall_at_1", 0.0)) for r in runs]
        out[key] = {"determinism": mean_std_ci(f1s), "prompt_sensitivity": max(f1s) - min(f1s) if f1s else 0.0, "retrieval_quality": mean_std_ci(rec1)}
    return out


def _write_error_tables(artifacts_dir: Path, rows: list[dict]) -> None:
    table_dir = artifacts_dir / "metrics"
    table_dir.mkdir(exist_ok=True)
    csv_path = table_dir / "error_summary.csv"
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["pair", "variant", "seed", "fp", "fn", "schema_invalid", "cardinality_issue", "retrieval_failure", "llm_hallucination"])
        writer.writeheader()
        for row in rows:
            source, target = row["pair"].split("->", 1)
            pred_dir = artifacts_dir / "pairs" / _pair_key(source, target) / "results" / row["baseline"] / f"seed{row['seed']}"
            analysis_path = pred_dir / "error_analysis.json"
            analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else {}
            reasons = analysis.get("validation_reasons", {})
            writer.writerow(
                {
                    "pair": row["pair"],
                    "variant": row["baseline"],
                    "seed": row["seed"],
                    "fp": len(analysis.get("false_positives", [])),
                    "fn": len(analysis.get("false_negatives", [])),
                    "schema_invalid": reasons.get("schema_invalid", 0),
                    "cardinality_issue": reasons.get("cardinality_issue", 0),
                    "retrieval_failure": len(analysis.get("retrieval_failures", [])),
                    "llm_hallucination": len(analysis.get("llm_hallucinations", [])),
                }
            )


def run_full_workflow() -> int:
    cfg = _load_config(Path(os.getenv("BENCHMARK_CONFIG", "benchmark/benchmark_full.json")))
    seeds = _resolve_seeds(cfg)
    run_id = os.getenv("RUN_ID", _now_run_id(cfg.get("run_id_prefix", "run")))
    artifacts_dir, compat_dir = _artifact_paths(run_id)
    logs_dir = artifacts_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    try:
        validate_or_generate(Path("datasets/v1"))
        pairs = [tuple(pair) for pair in cfg.get("pairs", [])]
        variants = [v["name"] for v in cfg["variants"] if v["name"] in CANONICAL_SCENARIOS]
        max_rows = int(os.getenv("MAX_ITEMS", str(cfg.get("items_per_standard", 120))))
        skipped_pairs: list[dict[str, str]] = []
        rows: list[dict] = []

        for source_standard, target_standard in pairs:
            supported, reason = _pair_supported(source_standard, target_standard)
            if not supported:
                skipped_pairs.append({"pair": f"{source_standard}->{target_standard}", "reason": reason})
                print(f"[SKIP] {source_standard}->{target_standard}: {reason}", flush=True)
                continue
            pair_dir = _build_pair_dataset(artifacts_dir, source_standard, target_standard, max_rows)
            for variant in variants:
                for seed in seeds:
                    metrics, _ = _run_variant(variant, pair_dir, Path("benchmark/config.json"), logs_dir, seed, source_standard, target_standard)
                    rows.append(metrics)

        (artifacts_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
        (artifacts_dir / "skipped_pairs.json").write_text(json.dumps(skipped_pairs, indent=2))
        (artifacts_dir / "metrics").mkdir(exist_ok=True)
        robustness = _measure_robustness(rows)
        (artifacts_dir / "metrics" / "advanced_analysis.json").write_text(
            json.dumps(
                {
                    "robustness": robustness,
                    "limitations": ["Unsupported pairs are skipped and recorded in skipped_pairs.json."],
                    "runtime_dependencies": {"model": os.getenv("MODEL_NAME", "gemma4:e2b")},
                },
                indent=2,
            )
        )
        _write_error_tables(artifacts_dir, rows)
        write_report(artifacts_dir, rows)

        if compat_dir.exists() and not compat_dir.is_symlink():
            for p in artifacts_dir.iterdir():
                target = compat_dir / p.name
                if p.is_file() and not target.exists():
                    target.write_text(p.read_text())
        print(f"RUN_ID={run_id}")
        return 0
    except Exception as exc:
        (logs_dir / "error.log").write_text(f"failed: {exc}\n")
        print(f"workflow_failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(run_full_workflow())
