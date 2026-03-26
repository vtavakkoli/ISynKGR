from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmark.evaluate import evaluate_run
from benchmark.metrics import mean_std_ci
from benchmark.report import write_report
from isynkgr.icr.mapping_schema import ingest_mapping_payload
from benchmark.validate_dataset import validate_or_generate

SEEDS = [11, 23, 37]

COMPONENT_FLAGS = {
    "full_framework": {},
    "rule_based_only": {"retrieval": False, "llm": False, "adaptive_selection": False},
    "llm_only": {"rules": False, "retrieval": False},
    "rag_only": {"rules": False, "llm": False},
    "embedding_similarity": {"rules": False, "llm": False},
    "ablation_no_rules": {"rules": False},
    "ablation_no_retrieval": {"retrieval": False},
    "ablation_no_graph_expansion": {"postprocess_snap": False},
    "ablation_no_llm": {"llm": False},
    "ablation_no_reasoning_prompt": {"reasoning_prompt": False},
    "ablation_no_community_filter": {"community_filter": False},
    "ablation_no_parallel_retrieval": {"parallel_retrieval": False},
}


def _now_run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text())


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


def _copy_gt_and_dataset(artifacts_dir: Path) -> None:
    gt_src = Path("datasets/v1/crosswalk/gt_mappings.jsonl")
    gt_dst = artifacts_dir / "ground_truth.jsonl"
    rows = []
    gt_rows = []
    tiers = ["synthetic", "noisy", "realistic"]
    difficulties = ["easy", "medium", "hard"]
    pair_cycle = [("OPCUA", "AAS"), ("IEEE1451", "IEC61499"), ("ISO15926", "AAS")]
    source_dir = artifacts_dir / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    max_rows = int(os.getenv("MAX_ITEMS", "180"))
    for i, line in enumerate(gt_src.read_text().splitlines()):
        if not line.strip():
            continue
        rec = ingest_mapping_payload(json.loads(line), migrate_legacy=True).model_dump()
        source_standard, target_standard = pair_cycle[i % len(pair_cycle)]
        source_file = source_dir / f"sample_{i:04d}_{source_standard.lower()}.json"
        if source_standard == "OPCUA":
            source_file = Path("datasets/v1/opcua/synthetic") / f"opcua_{i % 100:03d}.xml"
        else:
            payload = {
                "standard": source_standard,
                "classes": [{"id": f"tag_{i}", "label": "Temperature"}],
                "relations": [{"source": f"tag_{i}", "target": f"tag_{i}_value", "type": "hasValue"}],
                "teds": [{"id": f"teds_{i}", "name": "SensorTEDS", "channels": [{"id": "ch0", "dtype": "FLOAT", "unit": "C", "range": {"min": -50, "max": 250}}]}],
                "devices": [{"id": f"dev_{i}", "resources": [{"id": "res1", "function_blocks": [{"id": "fb1", "type": "SIFB", "inputs": [], "outputs": [{"id": "OUT_TEMP", "dtype": "FLOAT", "unit": "C", "range": {"min": -50, "max": 250}}]}]}]}],
            }
            source_file.write_text(json.dumps(payload))
        if i >= max_rows:
            break
        if source_standard == "OPCUA":
            source_id = rec["source_path"]
        elif source_standard == "IEEE1451":
            source_id = f"ieee1451://teds{i}/ch{i % 4}/temp"
        elif source_standard == "ISO15926":
            source_id = f"iso15926://class{i}"
        else:
            source_id = f"{source_standard.lower()}://device{i}/res1/fb1/var{i % 3}"

        if target_standard == "AAS":
            target_id = rec["target_path"]
        elif target_standard == "IEC61499":
            target_id = f"iec61499://Device{i}/Res1/FB1/OUT_TEMP"
        else:
            target_id = f"{target_standard.lower()}://ns=2;s=bench{i}"
        normalized = rec | {"source_path": source_id, "target_path": target_id}
        gt_rows.append(normalized)
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
                "transform_requirement": "unit_convert" if i % 4 == 0 else "none",
                "has_hard_negative": i % 7 == 0,
                "is_no_match": i % 11 == 0,
                "is_paraphrase": i % 5 == 0,
                "source_path": str(source_file),
            }
        )
    gt_dst.write_text("\n".join(json.dumps(r) for r in gt_rows) + "\n")
    (artifacts_dir / "dataset.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _run_variant(variant_name: str, artifacts_dir: Path, cfg_path: Path, logs_dir: Path, seed: int) -> tuple[dict, float]:
    out_dir = artifacts_dir / "predictions" / f"{variant_name}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "DATASET_DIR": str(artifacts_dir.resolve()),
            "OUTPUT_DIR": str(out_dir.resolve()),
            "CONFIG_PATH": str(cfg_path.resolve()),
            "SUT_MODE": "embedding_only" if variant_name == "embedding_similarity" else ("hybrid" if variant_name == "full_framework" or variant_name.startswith("ablation_") else variant_name),
            "SEED": str(seed),
            "MAX_ITEMS": str(int(os.getenv("MAX_ITEMS", "100"))),
            "COMPONENT_FLAGS": json.dumps(COMPONENT_FLAGS.get(variant_name, {})),
        }
    )

    start = time.perf_counter()
    log_path = logs_dir / f"{variant_name}_seed{seed}.log"
    with log_path.open("w") as fp:
        proc = subprocess.Popen(["python", "-u", "-m", "benchmark.run_sut"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            fp.write(line)
        proc.wait()
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"variant {variant_name} seed {seed} failed")

    metrics = evaluate_run(out_dir)
    metrics["baseline"] = variant_name
    metrics["seed"] = seed
    metrics["time_s"] = elapsed
    return metrics, elapsed


def _measure_robustness(rows: list[dict]) -> dict:
    by_variant: dict[str, list[dict]] = {}
    for row in rows:
        by_variant.setdefault(row["baseline"], []).append(row)
    out: dict[str, dict] = {}
    for variant, runs in by_variant.items():
        f1s = [float(r.get("f1", 0.0)) for r in runs]
        rec1 = [float(r.get("retrieval_recall_at_1", 0.0)) for r in runs]
        out[variant] = {
            "determinism": mean_std_ci(f1s),
            "prompt_sensitivity": max(f1s) - min(f1s) if f1s else 0.0,
            "noise_robustness": sum(float(r.get("per_tier", {}).get("noisy", {}).get("accuracy", 0.0)) for r in runs) / len(runs),
            "paraphrase_robustness": sum(float(r.get("per_tier", {}).get("realistic", {}).get("accuracy", 0.0)) for r in runs) / len(runs),
            "retrieval_quality": mean_std_ci(rec1),
        }
    return out


def _write_error_tables(artifacts_dir: Path, rows: list[dict]) -> None:
    table_dir = artifacts_dir / "metrics"
    table_dir.mkdir(exist_ok=True)
    csv_path = table_dir / "error_summary.csv"
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["variant", "seed", "fp", "fn", "invalid", "wrong_transform", "retrieval_failure", "llm_hallucination", "cardinality_issue"])
        writer.writeheader()
        for row in rows:
            pred_dir = artifacts_dir / "predictions" / f"{row['baseline']}_seed{row['seed']}"
            analysis = json.loads((pred_dir / "error_analysis.json").read_text())
            writer.writerow(
                {
                    "variant": row["baseline"],
                    "seed": row["seed"],
                    "fp": len(analysis.get("false_positives", [])),
                    "fn": len(analysis.get("false_negatives", [])),
                    "invalid": len(analysis.get("invalid_path", [])),
                    "wrong_transform": len(analysis.get("wrong_transform", [])),
                    "retrieval_failure": len(analysis.get("retrieval_failures", [])),
                    "llm_hallucination": len(analysis.get("llm_hallucinations", [])),
                    "cardinality_issue": len(analysis.get("cardinality_issues", [])),
                }
            )


def run_full_workflow() -> int:
    cfg = _load_config(Path(os.getenv("BENCHMARK_CONFIG", "benchmark/benchmark_full.json")))
    run_id = os.getenv("RUN_ID", _now_run_id(cfg.get("run_id_prefix", "run")))
    artifacts_dir, compat_dir = _artifact_paths(run_id)
    logs_dir = artifacts_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    try:
        validate_or_generate(Path("datasets/v1"))
        _copy_gt_and_dataset(artifacts_dir)
        variants = [v["name"] for v in cfg["variants"]]

        rows: list[dict] = []
        for variant in variants:
            for seed in SEEDS:
                metrics, _ = _run_variant(variant, artifacts_dir, Path("benchmark/config.json"), logs_dir, seed)
                rows.append(metrics)

        (artifacts_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
        (artifacts_dir / "metrics" ).mkdir(exist_ok=True)
        robustness = _measure_robustness(rows)
        (artifacts_dir / "metrics" / "advanced_analysis.json").write_text(json.dumps({"robustness": robustness, "limitations": ["Current execution supports OPCUA->AAS pair only."], "runtime_dependencies": {"model": os.getenv("MODEL_NAME", "qwen3.5:0.8b")}}, indent=2))
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
