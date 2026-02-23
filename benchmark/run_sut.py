from __future__ import annotations

import json
import os
import time
from pathlib import Path

from isynkgr.pipeline.hybrid import TranslatorConfig
from isynkgr.translator import Translator
from isynkgr.icr.mapping_schema import ingest_mapping_payload, normalize_mapping_path


def _fmt_s(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _read_dataset(dataset_dir: Path, max_samples: int) -> list[dict]:
    dataset_file = dataset_dir / "dataset.jsonl"
    if dataset_file.exists():
        rows = [json.loads(line) for line in dataset_file.read_text().splitlines() if line.strip()]
        return rows[:max_samples]
    opc_files = sorted((dataset_dir.parent / "opcua" / "synthetic").glob("*.xml"))[:max_samples]
    return [{"id": f.stem, "source_path": str(f)} for i, f in enumerate(opc_files)]


def _validate_mapping(mapping: dict, seen_keys: set[tuple[str, str, str]]) -> tuple[bool, list[dict], dict | None]:
    violations: list[dict] = []
    try:
        record = ingest_mapping_payload(mapping, migrate_legacy=True)
    except Exception as exc:
        return False, [{"type": "schema_invalid", "message": str(exc)}], None

    if float(record.confidence) < 0.5:
        violations.append({"type": "confidence_low", "message": f"confidence too low: {record.confidence}"})

    dedup_key = (normalize_mapping_path(record.source_path), normalize_mapping_path(record.target_path), str(record.mapping_type))
    if dedup_key in seen_keys:
        violations.append({"type": "duplicate_mapping", "message": f"Duplicate mapping key: {dedup_key}"})
    else:
        seen_keys.add(dedup_key)

    return (len(violations) == 0), violations, record.model_dump()


def main() -> None:
    dataset_dir = Path(os.getenv("DATASET_DIR", "/data"))
    output_dir = Path(os.getenv("OUTPUT_DIR", "/out"))
    config_path = Path(os.getenv("CONFIG_PATH", "/config/config.json"))
    mode = os.getenv("SUT_MODE", "hybrid")
    max_samples = int(os.getenv("MAX_ITEMS", os.getenv("MAX_SAMPLES", "100")))
    model_name = os.getenv("MODEL_NAME", "qwen3:0.6b")
    seed = int(os.getenv("SEED", "42"))
    tier = os.getenv("TIER", "canonical")
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_log = output_dir / "progress.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with progress_log.open("a") as fp:
            fp.write(msg + "\n")

    suite_start = time.perf_counter()
    log(
        f"[START] service=run_sut scenario={mode} ts_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
        f"model={model_name} seed={seed} tier={tier} items={max_samples} config={config_path}"
    )
    cfg_data = json.loads(config_path.read_text()) if config_path.exists() else {}
    cfg = TranslatorConfig(model_name=cfg_data.get("model_name", model_name), seed=cfg_data.get("seed", seed))
    translator = Translator(cfg)

    mapping_lines: list[str] = []
    validations: list[dict] = []
    llm_bugs: list[dict] = []
    dataset_rows = _read_dataset(dataset_dir, max_samples)
    total = len(dataset_rows)
    log(f"[SUITE] stage=translation total={total} completed=0 remaining={total}")
    seen_keys: set[tuple[str, str, str]] = set()

    for idx, row in enumerate(dataset_rows, start=1):
        source_path = row.get("source_path")
        if source_path:
            sample_path = Path(source_path)
        else:
            sample_path = dataset_dir.parent / "opcua" / "synthetic" / f"opcua_{idx - 1:03d}.xml"

        log(f"[SAMPLE] scenario={mode} sample {idx}/{total} source={sample_path}")
        result = translator.translate("opcua", "aas", str(sample_path), mode=mode if mode != "isynkgr_hybrid" else "hybrid")
        llm_error = (result.provenance.metadata or {}).get("llm_error") if result.provenance else None
        if llm_error:
            llm_bugs.append({"sample": sample_path.name, "mode": mode, "error": llm_error})

        item_violations: list[dict] = []
        if result.mappings:
            for m in result.mappings:
                record = m.model_dump()
                is_valid, violations, normalized = _validate_mapping(record, seen_keys)
                if not is_valid:
                    item_violations.extend(violations)
                if normalized is not None:
                    mapping_lines.append(json.dumps(normalized))
        else:
            fallback = {
                "source_path": normalize_mapping_path(row.get("mapping_source_path", f"opcua://ns=2;i={1000 + idx - 1}")),
                "target_path": f"aas://aas-{idx - 1}/submodel/default/element/value",
                "mapping_type": "fallback",
                "confidence": 0.2,
                "rationale": "Fallback mapping generated because model returned no mappings.",
                "evidence": [],
            }
            _, violations, normalized = _validate_mapping(fallback, seen_keys)
            item_violations.extend(violations)
            if normalized is not None:
                mapping_lines.append(json.dumps(normalized))

        valid = len(item_violations) == 0
        validations.append({"valid": valid, "violations": item_violations})

        if idx % 5 == 0 or idx == total:
            elapsed = time.perf_counter() - suite_start
            avg = elapsed / idx if idx else 0.0
            eta = avg * (total - idx)
            pct = (idx / total * 100.0) if total else 100.0
            log(f"Processed {idx}/{total} ({pct:.1f}%) | avg_time_per_item={avg:.3f}s | ETA={eta:.2f}s")

    (output_dir / "mappings.jsonl").write_text("\n".join(mapping_lines) + ("\n" if mapping_lines else ""))
    (output_dir / "validation.json").write_text(json.dumps(validations, indent=2))
    (output_dir / "provenance.json").write_text(json.dumps({"mode": mode, "dataset": str(dataset_dir), "llm_bug_count": len(llm_bugs)}, indent=2))
    (output_dir / "bugs.json").write_text(json.dumps(llm_bugs, indent=2))

    suite_elapsed = time.perf_counter() - suite_start
    throughput = (total / suite_elapsed) if suite_elapsed > 0 else 0.0
    log(f"[SUITE-END] total={total} elapsed={_fmt_s(suite_elapsed)} throughput={throughput:.2f}/s bugs={len(llm_bugs)}")


if __name__ == "__main__":
    main()
