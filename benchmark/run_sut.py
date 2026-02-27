from __future__ import annotations

import json
import os
import time
from pathlib import Path

from isynkgr.icr.mapping_output_contract import validate_mapping_item
from isynkgr.icr.mapping_schema import MappingType, normalize_mapping_path
from isynkgr.pipeline.hybrid import TranslatorConfig
from isynkgr.translator import Translator


def _fmt_s(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _mapping_key(mapping: dict) -> tuple[str, str, str]:
    return (
        normalize_mapping_path(mapping.get("source_path", "")),
        normalize_mapping_path(mapping.get("target_path", "")),
        str(mapping.get("mapping_type", "")),
    )


def _read_dataset(dataset_dir: Path, max_samples: int) -> list[dict]:
    dataset_file = dataset_dir / "dataset.jsonl"
    if dataset_file.exists():
        rows = [json.loads(line) for line in dataset_file.read_text().splitlines() if line.strip()]
        return rows[:max_samples]
    opc_files = sorted((dataset_dir.parent / "opcua" / "synthetic").glob("*.xml"))[:max_samples]
    return [{"id": f.stem, "source_path": str(f)} for i, f in enumerate(opc_files)]


def _validate_mapping(mapping: dict, source_protocol: str, target_protocol: str, seen_keys: set[tuple[str, str, str]]) -> tuple[bool, list[dict], dict | None]:
    violations: list[dict] = []
    is_ok, error = validate_mapping_item(mapping, source_protocol=source_protocol, target_protocol=target_protocol)
    if not is_ok:
        return False, [{"type": "schema_invalid", "message": error}], None

    if float(mapping.get("confidence", 0.0)) < 0.5 and mapping.get("mapping_type") != MappingType.NO_MATCH.value:
        violations.append({"type": "confidence_low", "message": f"confidence too low: {mapping.get('confidence')}"})

    dedup_key = _mapping_key(mapping)
    if dedup_key in seen_keys:
        violations.append({"type": "duplicate_mapping", "message": f"Duplicate mapping key: {dedup_key}"})
    else:
        seen_keys.add(dedup_key)

    return (len(violations) == 0), violations, mapping


def _deduplicate_and_sort_mappings(mappings: list[dict]) -> list[dict]:
    best_by_key: dict[tuple[str, str, str], dict] = {}
    for mapping in mappings:
        key = _mapping_key(mapping)
        current = best_by_key.get(key)
        if current is None or float(mapping.get("confidence", 0.0)) > float(current.get("confidence", 0.0)):
            best_by_key[key] = mapping
    return [best_by_key[key] for key in sorted(best_by_key)]




def _enforce_cardinality(sample_mappings: list[dict], contract: dict, item_violations: list[dict]) -> list[dict]:
    expected_count = contract["expected_count"]
    if contract["mode"] == "grouped_1":
        return sample_mappings
    if len(sample_mappings) <= expected_count:
        return sample_mappings

    def _rank(mapping: dict) -> tuple[float, int]:
        confidence = float(mapping.get("confidence", 0.0))
        no_match_penalty = 1 if mapping.get("mapping_type") == MappingType.NO_MATCH.value else 0
        return (confidence, -no_match_penalty)

    trimmed = sorted(sample_mappings, key=_rank, reverse=True)[:expected_count]
    item_violations.append(
        {
            "type": "cardinality_trimmed",
            "message": (
                f"Trimmed mappings from {len(sample_mappings)} to {expected_count} "
                f"for mode={contract['mode']}"
            ),
            "expected_count": expected_count,
            "actual_count": len(sample_mappings),
            "trimmed_count": len(trimmed),
            "contract": contract,
        }
    )
    return _deduplicate_and_sort_mappings(trimmed)
def _extract_cardinality_contract(row: dict) -> dict:
    contract = row.get("cardinality_contract") or {}
    mode = str(contract.get("mode") or "one_to_one")
    grouped_1 = bool(contract.get("grouped_1", mode == "grouped_1"))
    expected_count = contract.get("expected_count")
    if expected_count is None:
        expected_count = 1 if not grouped_1 else 0
    return {
        "mode": mode,
        "grouped_1": grouped_1,
        "expected_count": int(expected_count),
    }


def main() -> None:
    dataset_dir = Path(os.getenv("DATASET_DIR", "/data"))
    output_dir = Path(os.getenv("OUTPUT_DIR", "/out"))
    predictions_dir = output_dir / "predictions"
    config_path = Path(os.getenv("CONFIG_PATH", "/config/config.json"))
    mode = os.getenv("SUT_MODE", "hybrid")
    source_protocol = os.getenv("SOURCE_PROTOCOL", "opcua")
    target_protocol = os.getenv("TARGET_PROTOCOL", "aas")
    max_samples = int(os.getenv("MAX_ITEMS", os.getenv("MAX_SAMPLES", "100")))
    model_name = os.getenv("MODEL_NAME", "qwen3:0.6b")
    seed = int(os.getenv("SEED", "42"))
    tier = os.getenv("TIER", "canonical")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

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

    mapping_records: list[dict] = []
    validations: list[dict] = []
    llm_bugs: list[dict] = []
    rejected_mappings: list[dict] = []
    llm_raw_output: list[dict] = []
    dataset_rows = _read_dataset(dataset_dir, max_samples)
    total = len(dataset_rows)
    log(f"[SUITE] stage=translation total={total} completed=0 remaining={total}")
    seen_keys: set[tuple[str, str, str]] = set()

    for idx, row in enumerate(dataset_rows, start=1):
        source_path = row.get("source_path")
        if source_path:
            sample_path = Path(source_path)
        else:
            sample_path = dataset_dir.parent / source_protocol / "synthetic" / f"{source_protocol}_{idx - 1:03d}.xml"

        contract = _extract_cardinality_contract(row)

        log(f"[SAMPLE] scenario={mode} sample {idx}/{total} source={sample_path}")
        result = translator.translate(source_protocol, target_protocol, str(sample_path), mode=mode if mode != "isynkgr_hybrid" else "hybrid")
        metadata = (result.provenance.metadata or {}) if result.provenance else {}
        llm_error = metadata.get("llm_error")
        if llm_error:
            llm_bugs.append({"sample": sample_path.name, "mode": mode, "error": llm_error})

        llm_raw_output.extend(metadata.get("llm_raw_output", []))
        rejected_mappings.extend(metadata.get("rejected_mappings", []))

        item_violations: list[dict] = []
        sample_mappings: list[dict] = []
        for m in result.mappings:
            record = m.model_dump()
            is_valid, violations, normalized = _validate_mapping(record, source_protocol=source_protocol, target_protocol=target_protocol, seen_keys=seen_keys)
            if not is_valid:
                item_violations.extend(violations)
            if normalized is not None:
                sample_mappings.append(normalized)

        if not sample_mappings:
            source_id = row.get("mapping_source_path", f"{source_protocol}://unknown-{idx - 1}")
            sample_mappings.append(
                {
                    "source_path": normalize_mapping_path(source_id),
                    "target_path": "",
                    "mapping_type": "no_match",
                    "transform": None,
                    "confidence": 0.0,
                    "rationale": "No valid mappings were produced by the pipeline.",
                    "evidence": [],
                }
            )

        sample_mappings = _deduplicate_and_sort_mappings(sample_mappings)

        sample_mappings = _enforce_cardinality(sample_mappings, contract, item_violations)

        expected_count = contract["expected_count"]
        if contract["mode"] != "grouped_1" and len(sample_mappings) != expected_count:
            item_violations.append(
                {
                    "type": "cardinality_mismatch",
                    "message": f"Expected {expected_count} mappings for sample in mode={contract['mode']}, got {len(sample_mappings)}",
                    "expected_count": expected_count,
                    "actual_count": len(sample_mappings),
                    "contract": contract,
                }
            )

        mapping_records.extend(sample_mappings)

        valid = len(item_violations) == 0
        validations.append({"valid": valid, "violations": item_violations, "cardinality_contract": contract})

        if idx % 5 == 0 or idx == total:
            elapsed = time.perf_counter() - suite_start
            avg = elapsed / idx if idx else 0.0
            eta = avg * (total - idx)
            pct = (idx / total * 100.0) if total else 100.0
            log(f"Processed {idx}/{total} ({pct:.1f}%) | avg_time_per_item={avg:.3f}s | ETA={eta:.2f}s")

    mapping_records = _deduplicate_and_sort_mappings(mapping_records)
    mapping_lines = [json.dumps(record) for record in mapping_records]

    errors_summary = {
        "mode": mode,
        "source_protocol": source_protocol,
        "target_protocol": target_protocol,
        "rejected_count": len(rejected_mappings),
        "llm_error_count": len(llm_bugs),
        "validation_invalid_count": sum(1 for v in validations if not v.get("valid")),
    }

    (output_dir / "mappings.jsonl").write_text("\n".join(mapping_lines) + ("\n" if mapping_lines else ""))
    (output_dir / "validation.json").write_text(json.dumps(validations, indent=2))
    (output_dir / "provenance.json").write_text(json.dumps({"mode": mode, "dataset": str(dataset_dir), "llm_bug_count": len(llm_bugs)}, indent=2))
    (output_dir / "bugs.json").write_text(json.dumps(llm_bugs, indent=2))
    (predictions_dir / "rejected_mappings.jsonl").write_text("\n".join(json.dumps(row) for row in rejected_mappings) + ("\n" if rejected_mappings else ""))
    (predictions_dir / "llm_raw_output.jsonl").write_text("\n".join(json.dumps(row) for row in llm_raw_output) + ("\n" if llm_raw_output else ""))
    (predictions_dir / "errors_summary.json").write_text(json.dumps(errors_summary, indent=2))

    suite_elapsed = time.perf_counter() - suite_start
    throughput = (total / suite_elapsed) if suite_elapsed > 0 else 0.0
    log(f"[SUITE-END] total={total} elapsed={_fmt_s(suite_elapsed)} throughput={throughput:.2f}/s bugs={len(llm_bugs)}")


if __name__ == "__main__":
    main()
