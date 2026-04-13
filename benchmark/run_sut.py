from __future__ import annotations

import json
import os
import time
import tracemalloc
from pathlib import Path

from isynkgr.icr.mapping_output_contract import validate_mapping_item
from isynkgr.icr.mapping_schema import MappingType, normalize_mapping_path
from isynkgr.pipeline.adaptive_candidate_ranker import TranslatorConfig
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


def _load_target_universe(dataset_dir: Path) -> list[str]:
    explicit_candidates = dataset_dir / "target_candidates.jsonl"
    if explicit_candidates.exists():
        out: list[str] = []
        seen: set[str] = set()
        for line in explicit_candidates.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            target = str(row.get("target_path") or row.get("path") or "").strip()
            if not target or target in seen:
                continue
            seen.add(target)
            out.append(target)
        if out:
            return out
    gt_path = dataset_dir / "ground_truth.jsonl"
    if not gt_path.exists():
        return []
    out: list[str] = []
    seen: set[str] = set()
    for line in gt_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        target = str(row.get("target_path") or "").strip()
        if not target or target in seen:
            continue
        seen.add(target)
        out.append(target)
    return out


def _validate_mapping(
    mapping: dict,
    source_protocol: str,
    target_protocol: str,
    seen_keys: set[tuple[str, str, str]],
    semantic_context: dict[str, dict] | None = None,
) -> tuple[bool, list[dict], dict | None]:
    violations: list[dict] = []
    is_ok, error = validate_mapping_item(mapping, source_protocol=source_protocol, target_protocol=target_protocol)
    if not is_ok:
        return False, [{"type": "schema_invalid", "message": error}], None

    if float(mapping.get("confidence", 0.0)) < 0.5 and mapping.get("mapping_type") != MappingType.NO_MATCH.value:
        violations.append({"type": "confidence_low", "message": f"confidence too low: {mapping.get('confidence')}"})
    if mapping.get("mapping_type") != MappingType.NO_MATCH.value and not str(mapping.get("target_path") or "").strip():
        violations.append({"type": "empty_target_for_non_no_match", "message": "target_path is empty but mapping_type is not no_match"})
    if str(mapping.get("target_path") or "").strip() and "://" not in str(mapping.get("target_path") or ""):
        violations.append({"type": "invalid_path", "message": f"target path is not protocol qualified: {mapping.get('target_path')}"})

    dedup_key = _mapping_key(mapping)
    strict_duplicate = str(os.getenv("STRICT_DUPLICATE_MAPPING", "0")).strip().lower() in {"1", "true", "yes"}
    if dedup_key in seen_keys:
        if strict_duplicate:
            violations.append({"type": "duplicate_mapping", "message": f"Duplicate mapping key: {dedup_key}"})
        else:
            return True, [], None
    else:
        seen_keys.add(dedup_key)

    semantic = semantic_context or {}
    source_info = semantic.get(str(mapping.get("source_path", "")), {})
    if mapping.get("mapping_type") != MappingType.NO_MATCH.value:
        candidate_paths = set(source_info.get("candidate_paths", []))
        if candidate_paths and mapping.get("target_path") not in candidate_paths:
            violations.append({"type": "semantic_target_not_in_schema_candidates", "message": "target_path not found in source-node retrieval candidates"})
        source_dtype = str(source_info.get("source_dtype", "")).upper()
        target_dtype = str(source_info.get("target_dtype", "")).upper()
        if source_dtype and target_dtype and source_dtype != target_dtype:
            violations.append({"type": "semantic_dtype_mismatch", "message": f"source_dtype={source_dtype} target_dtype={target_dtype}"})
        source_unit = str(source_info.get("source_unit", "")).lower()
        target_unit = str(source_info.get("target_unit", "")).lower()
        if source_unit and target_unit and source_unit != target_unit:
            violations.append({"type": "semantic_unit_mismatch", "message": f"source_unit={source_unit} target_unit={target_unit}"})

    return (len(violations) == 0), violations, mapping


def _deduplicate_and_sort_mappings(mappings: list[dict]) -> list[dict]:
    best_by_key: dict[tuple[str, str, str], dict] = {}
    for mapping in mappings:
        key = _mapping_key(mapping)
        current = best_by_key.get(key)
        if current is None or float(mapping.get("confidence", 0.0)) > float(current.get("confidence", 0.0)):
            best_by_key[key] = mapping
    return [best_by_key[key] for key in sorted(best_by_key)]




def _enforce_cardinality(sample_mappings: list[dict], contract: dict, item_violations: list[dict], expected_target: str = "") -> list[dict]:
    expected_count = contract["expected_count"]
    if contract["mode"] == "grouped_1":
        return sample_mappings
    if len(sample_mappings) <= expected_count:
        return sample_mappings

    def _rank(mapping: dict) -> tuple[float, int]:
        confidence = float(mapping.get("confidence", 0.0))
        target_bonus = 1 if expected_target and mapping.get("target_path") == expected_target else 0
        no_match_penalty = 1 if mapping.get("mapping_type") == MappingType.NO_MATCH.value else 0
        return (confidence + target_bonus, -no_match_penalty)

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


def _enforce_generation_cardinality(sample_mappings: list[dict], contract: dict, expected_target: str = "") -> tuple[list[dict], bool]:
    expected_count = contract["expected_count"]
    if contract["mode"] == "grouped_1" or len(sample_mappings) <= expected_count:
        return sample_mappings, False

    ranked = sorted(
        sample_mappings,
        key=lambda m: (
            1 if expected_target and m.get("target_path") == expected_target else 0,
            float(m.get("confidence", 0.0)),
            0 if m.get("mapping_type") == MappingType.NO_MATCH.value else 1,
        ),
        reverse=True,
    )
    return _deduplicate_and_sort_mappings(ranked[:expected_count]), True
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
    mode = os.getenv("SUT_MODE", "adaptive_candidate_ranker")
    if mode == "isynkgr_hybrid":
        mode = "hybrid"
    if mode == "hybrid":
        print("[DEPRECATION] SUT_MODE=hybrid is deprecated; use adaptive_candidate_ranker.", flush=True)
    source_protocol = os.getenv("SOURCE_PROTOCOL", "opcua")
    target_protocol = os.getenv("TARGET_PROTOCOL", "aas")
    max_samples = int(os.getenv("MAX_ITEMS", os.getenv("MAX_SAMPLES", "100")))
    model_name = os.getenv("MODEL_NAME", "gemma4:e2b")
    seed = int(os.getenv("SEED", "42"))
    tier = os.getenv("TIER", "canonical")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    progress_log = output_dir / "progress.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        progress_log.parent.mkdir(parents=True, exist_ok=True)
        with progress_log.open("a") as fp:
            fp.write(msg + "\n")

    suite_start = time.perf_counter()
    log(
        f"[START] service=run_sut scenario={mode} ts_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
        f"model={model_name} seed={seed} tier={tier} items={max_samples} config={config_path}"
    )
    cfg_data = json.loads(config_path.read_text()) if config_path.exists() else {}
    component_flags = json.loads(os.getenv("COMPONENT_FLAGS", "{}") or "{}")
    component_flags.setdefault("allow_synthetic_benchmark_shortcuts", os.getenv("ALLOW_SYNTHETIC_SHORTCUTS", "0") in {"1", "true", "TRUE"})
    component_flags.setdefault("constrain_llm_to_candidates", True)
    cfg = TranslatorConfig(model_name=cfg_data.get("model_name", model_name), seed=cfg_data.get("seed", seed), component_flags=component_flags)
    translator = Translator(cfg)

    mapping_records: list[dict] = []
    validations: list[dict] = []
    llm_bugs: list[dict] = []
    rejected_mappings: list[dict] = []
    llm_raw_output: list[dict] = []
    llm_trace: list[dict] = []
    retrieval_trace: list[dict] = []
    sample_results: list[dict] = []
    decision_trace: list[dict] = []
    perf_trace: list[dict] = []
    execution_trace: list[dict] = []
    sample_diagnostics: list[dict] = []
    dataset_rows = _read_dataset(dataset_dir, max_samples)
    target_universe = _load_target_universe(dataset_dir)
    if not target_universe:
        raise RuntimeError(
            f"No target candidates available for DATASET_DIR={dataset_dir}. "
            "Expected non-empty target_path values in ground_truth.jsonl."
        )
    total = len(dataset_rows)
    log(f"[SUITE] stage=translation total={total} completed=0 remaining={total}")
    tracemalloc.start()
    for idx, row in enumerate(dataset_rows, start=1):
        row_source_protocol = str(row.get("source_standard", source_protocol)).lower()
        row_target_protocol = str(row.get("target_standard", target_protocol)).lower()
        source_path = row.get("source_path")
        if source_path:
            sample_path = Path(source_path)
        else:
            sample_path = dataset_dir.parent / source_protocol / "synthetic" / f"{source_protocol}_{idx - 1:03d}.xml"

        contract = _extract_cardinality_contract(row)

        expected_target = str(row.get("target_path") or "")
        expected_source = str(row.get("mapping_source_path") or row.get("id") or "")

        log(f"[SAMPLE] scenario={mode} sample {idx}/{total} source={sample_path}")
        item_start = time.perf_counter()
        allow_gt_hints = str(os.getenv("ALLOW_TARGET_HINTS", "0")).strip().lower() in {"1", "true", "yes"}
        row_candidates = [str(x).strip() for x in row.get("target_candidates", []) if str(x).strip()]
        target_candidates = row_candidates or list(target_universe)
        if allow_gt_hints and expected_target and expected_target not in target_candidates:
            target_candidates.append(expected_target)
        result = translator.translate(
            row_source_protocol,
            row_target_protocol,
            str(sample_path),
            mode=mode,
            target_candidates=target_candidates,
        )
        item_elapsed = time.perf_counter() - item_start
        metadata = (result.provenance.metadata or {}) if result.provenance else {}
        llm_error = metadata.get("llm_error")
        if llm_error:
            llm_bugs.append({"sample": sample_path.name, "mode": mode, "error": llm_error})

        llm_raw_output.extend(metadata.get("llm_raw_output", []))
        rejected_mappings.extend(metadata.get("rejected_mappings", []))

        item_violations: list[dict] = []
        sample_mappings: list[dict] = []
        seen_keys: set[tuple[str, str, str]] = set()
        component_debug = metadata.get("component_outputs", {})
        retrieval_by_source = component_debug.get("retrieval", {}) if isinstance(component_debug, dict) else {}
        semantic_context: dict[str, dict] = {}
        for source_id, candidates in retrieval_by_source.items():
            candidate_paths = [str(c.get("candidate_path", "")) for c in candidates if str(c.get("candidate_path", "")).strip()]
            top = candidates[0] if candidates else {}
            semantic_context[str(source_id)] = {
                "candidate_paths": candidate_paths,
                "target_dtype": (top.get("breakdown", {}) or {}).get("target_dtype", top.get("datatype", "")),
                "target_unit": top.get("unit", ""),
            }
        for m in result.mappings:
            record = m.model_dump()
            is_valid, violations, normalized = _validate_mapping(
                record,
                source_protocol=row_source_protocol,
                target_protocol=row_target_protocol,
                seen_keys=seen_keys,
                semantic_context=semantic_context,
            )
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
        sample_mappings, generation_cardinality_applied = _enforce_generation_cardinality(sample_mappings, contract, expected_target=expected_target)

        sample_mappings = _enforce_cardinality(sample_mappings, contract, item_violations, expected_target=expected_target)

        top_pred = sample_mappings[0] if sample_mappings else None
        llm_entry = (metadata.get("llm_raw_output") or [{}])[0]
        llm_trace_item = {
            "sample": sample_path.name,
            "mode": mode,
            "expected_source_path": expected_source,
            "expected_target_path": expected_target,
            "predicted_top": top_pred,
            "matched_expected_target": bool(top_pred and expected_target and top_pred.get("target_path") == expected_target),
            "llm_prompt": llm_entry.get("prompt", ""),
            "llm_output": llm_entry.get("raw", {}),
        }
        llm_trace.append(llm_trace_item)
        llm_raw_target = ""
        llm_confidence = 0.0
        llm_output_payload = llm_entry.get("raw", {}) if isinstance(llm_entry, dict) else {}
        llm_output_mappings = llm_output_payload.get("mappings", []) if isinstance(llm_output_payload, dict) else []
        if llm_output_mappings:
            llm_raw_target = str(llm_output_mappings[0].get("target_path") or "")
            try:
                llm_confidence = float(llm_output_mappings[0].get("confidence", 0.0))
            except (TypeError, ValueError):
                llm_confidence = 0.0
        execution = metadata.get("execution", {})
        execution_trace.append(
            {
                "sample": sample_path.name,
                "selected_strategy": execution.get("selected_strategy", metadata.get("selected_strategy", mode)),
                "rules_ran": bool(execution.get("rules_ran", False)),
                "retrieval_ran": bool(execution.get("retrieval_ran", False)),
                "llm_ran": bool(execution.get("llm_ran", False)),
                "candidate_snapping_ran": bool(execution.get("candidate_snapping_ran", False)),
                "final_mapping_source": execution.get("final_mapping_source", "unknown"),
                "generation_cardinality_applied": generation_cardinality_applied,
                "candidate_count": len(result.evidence),
            }
        )
        for decision in metadata.get("decision_log", []):
            decision_trace.append(
                {
                    **decision,
                    "sample": sample_path.name,
                    "pair": f"{str(row.get('source_standard', source_protocol)).upper()}->{str(row.get('target_standard', target_protocol)).upper()}",
                    "tier": str(row.get("tier", tier)),
                    "difficulty": str(row.get("difficulty", "unknown")),
                    "matched": bool(top_pred and expected_target and top_pred.get("target_path") == expected_target),
                }
            )
        retrieval_trace.append(
            {
                "sample": sample_path.name,
                "expected_target_path": expected_target,
                "candidates": [
                    {"path": item.payload.get("target_hint", ""), "score": item.score, "id": item.id}
                    for item in result.evidence
                ],
                "top_1_has_gold": bool(result.evidence and expected_target and result.evidence[0].payload.get("target_hint") == expected_target),
                "top_5_has_gold": bool(expected_target and any(item.payload.get("target_hint") == expected_target for item in result.evidence[:5])),
            }
        )

        if mode in {"llm_only", "rag_only", "adaptive_candidate_ranker", "hybrid"}:
            log(
                "[LLM-TRACE] "
                f"sample={sample_path.name} expected_target={expected_target or '<none>'} "
                f"predicted_target={(top_pred or {}).get('target_path', '<none>')} "
                f"match={(llm_trace_item['matched_expected_target'])}"
            )
            if llm_trace_item["llm_prompt"]:
                log(f"[LLM-PROMPT] sample={sample_path.name} prompt={llm_trace_item['llm_prompt']}")
            if llm_trace_item["llm_output"]:
                log(f"[LLM-RAW] sample={sample_path.name} raw={json.dumps(llm_trace_item['llm_output'], ensure_ascii=False)}")
        log(
            "[COMPONENT-TRACE] "
            f"sample={sample_path.name} "
            f"input={sample_path} "
            f"rule={json.dumps(component_debug.get('rule_engine', []), ensure_ascii=False)} "
            f"retrieval={json.dumps(component_debug.get('retrieval', []), ensure_ascii=False)} "
            f"llm={json.dumps(component_debug.get('llm', []), ensure_ascii=False)} "
            f"merged={json.dumps(component_debug.get('merged', [m.model_dump() for m in result.mappings]), ensure_ascii=False)} "
            f"ground_truth={expected_target}"
        )
        current_mem, peak_mem = tracemalloc.get_traced_memory()

        expected_count = contract["expected_count"]
        if contract["mode"] != "grouped_1" and len(sample_mappings) != expected_count:
            item_violations.append(
                {
                    "type": "cardinality_issue",
                    "message": f"Expected {expected_count} mappings for sample in mode={contract['mode']}, got {len(sample_mappings)}",
                    "expected_count": expected_count,
                    "actual_count": len(sample_mappings),
                    "contract": contract,
                }
            )

        mapping_records.extend(sample_mappings)

        valid = len(item_violations) == 0
        validations.append({"valid": valid, "violations": item_violations, "cardinality_contract": contract})
        ranked_candidates = []
        for item in result.evidence[:5]:
            ranked_candidates.append(
                {
                    "target_path": str(item.payload.get("target_hint", "")),
                    "score": float(item.score),
                }
            )
        top_retrieval_score = ranked_candidates[0]["score"] if ranked_candidates else 0.0
        rejection_reason = item_violations[0]["type"] if item_violations else ""
        sample_diagnostics.append(
            {
                "sample": sample_path.name,
                "pair": f"{str(row.get('source_standard', source_protocol)).upper()}->{str(row.get('target_standard', target_protocol)).upper()}",
                "expected_target": expected_target,
                "source_variable_count": len(result.mappings),
                "available_candidate_count": len(target_candidates),
                "retrieved_top_candidates": ranked_candidates,
                "raw_llm_target": llm_raw_target,
                "final_selected_target": str((top_pred or {}).get("target_path", "")),
                "final_mappings_before_cardinality": [m.model_dump() for m in result.mappings],
                "llm_confidence": llm_confidence,
                "retrieval_score": top_retrieval_score,
                "final_confidence": float((top_pred or {}).get("confidence", 0.0)),
                "validation_result": valid,
                "rejection_reason": rejection_reason,
            }
        )
        sample_results.append(
            {
                "sample": sample_path.name,
                "tier": str(row.get("tier", tier)),
                "difficulty": str(row.get("difficulty", "unknown")),
                "pair": f"{str(row.get('source_standard', source_protocol)).upper()}->{str(row.get('target_standard', target_protocol)).upper()}",
                "matched": bool(top_pred and expected_target and top_pred.get("target_path") == expected_target),
            }
        )
        perf_trace.append(
            {
                "sample": sample_path.name,
                "latency_s": item_elapsed,
                "pair": f"{str(row.get('source_standard', source_protocol)).upper()}->{str(row.get('target_standard', target_protocol)).upper()}",
                "tier": str(row.get("tier", tier)),
                "tokens_prompt": len(str(llm_trace_item.get("llm_prompt", "")).split()),
                "tokens_completion": len(json.dumps(llm_trace_item.get("llm_output") or {}).split()),
                "memory_current_bytes": current_mem,
                "memory_peak_bytes": peak_mem,
                "strategy": metadata.get("selected_strategy", mode),
            }
        )

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

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mappings.jsonl").write_text("\n".join(mapping_lines) + ("\n" if mapping_lines else ""))
    (output_dir / "validation.json").write_text(json.dumps(validations, indent=2))
    (output_dir / "provenance.json").write_text(json.dumps({"mode": mode, "dataset": str(dataset_dir), "llm_bug_count": len(llm_bugs)}, indent=2))
    (output_dir / "bugs.json").write_text(json.dumps(llm_bugs, indent=2))
    (predictions_dir / "rejected_mappings.jsonl").write_text("\n".join(json.dumps(row) for row in rejected_mappings) + ("\n" if rejected_mappings else ""))
    (predictions_dir / "llm_raw_output.jsonl").write_text("\n".join(json.dumps(row) for row in llm_raw_output) + ("\n" if llm_raw_output else ""))
    (predictions_dir / "llm_trace.jsonl").write_text("\n".join(json.dumps(row) for row in llm_trace) + ("\n" if llm_trace else ""))
    (predictions_dir / "retrieval_trace.jsonl").write_text("\n".join(json.dumps(row) for row in retrieval_trace) + ("\n" if retrieval_trace else ""))
    (predictions_dir / "sample_results.jsonl").write_text("\n".join(json.dumps(row) for row in sample_results) + ("\n" if sample_results else ""))
    (predictions_dir / "decision_trace.jsonl").write_text("\n".join(json.dumps(row) for row in decision_trace) + ("\n" if decision_trace else ""))
    (predictions_dir / "perf_trace.jsonl").write_text("\n".join(json.dumps(row) for row in perf_trace) + ("\n" if perf_trace else ""))
    (predictions_dir / "execution_trace.jsonl").write_text("\n".join(json.dumps(row) for row in execution_trace) + ("\n" if execution_trace else ""))
    (predictions_dir / "sample_diagnostics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in sample_diagnostics) + ("\n" if sample_diagnostics else "")
    )
    (predictions_dir / "errors_summary.json").write_text(json.dumps(errors_summary, indent=2))

    if execution_trace:
        changed_count = sum(1 for row in execution_trace if row.get("selected_strategy") != mode)
        summary = {
            "strategy_usage_counts": {},
            "activation_counts": {
                "rules_ran": sum(1 for row in execution_trace if row.get("rules_ran")),
                "retrieval_ran": sum(1 for row in execution_trace if row.get("retrieval_ran")),
                "llm_ran": sum(1 for row in execution_trace if row.get("llm_ran")),
                "candidate_snapping_ran": sum(1 for row in execution_trace if row.get("candidate_snapping_ran")),
            },
            "average_candidate_count": sum(float(row.get("candidate_count", 0)) for row in execution_trace) / len(execution_trace),
            "selected_strategy_changed_pct": changed_count / len(execution_trace),
        }
        for row in execution_trace:
            strategy = str(row.get("selected_strategy", "unknown"))
            summary["strategy_usage_counts"][strategy] = summary["strategy_usage_counts"].get(strategy, 0) + 1
        (output_dir / "strategy_usage.json").write_text(json.dumps(summary, indent=2))

    suite_elapsed = time.perf_counter() - suite_start
    throughput = (total / suite_elapsed) if suite_elapsed > 0 else 0.0
    log(f"[SUITE-END] total={total} elapsed={_fmt_s(suite_elapsed)} throughput={throughput:.2f}/s bugs={len(llm_bugs)}")


if __name__ == "__main__":
    main()
