from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from isynkgr.icr.mapping_schema import ingest_mapping_payload

SCENARIOS = (
    "opcua->aas",
    "aas->opcua",
    "ieee1451->aas",
    "iec61499->aas",
    "ieee1451->opcua",
    "iec61499->opcua",
)


@dataclass(frozen=True)
class ScenarioSpec:
    source_standard: str
    target_standard: str
    transform_kind: str


SCENARIO_SPECS: dict[str, ScenarioSpec] = {
    "opcua->aas": ScenarioSpec("OPCUA", "AAS", "unit_convert"),
    "aas->opcua": ScenarioSpec("AAS", "OPCUA", "cast"),
    "ieee1451->aas": ScenarioSpec("IEEE1451", "AAS", "id_normalize"),
    "iec61499->aas": ScenarioSpec("IEC61499", "AAS", "structural_map"),
    "ieee1451->opcua": ScenarioSpec("IEEE1451", "OPCUA", "cast"),
    "iec61499->opcua": ScenarioSpec("IEC61499", "OPCUA", "unit_convert"),
}


def _source_path(standard: str, idx: int, scenario_tag: str) -> str:
    if standard == "OPCUA":
        return f"opcua://ns=2;s={scenario_tag}.node.{1000 + idx}"
    if standard == "AAS":
        return f"aas://line-{scenario_tag}-{idx}/submodel/Process/element/Temperature_{idx}"
    if standard == "IEEE1451":
        return f"ieee1451://teds-{scenario_tag}-{idx}/ch-{idx % 4}/temp_c"
    if standard == "IEC61499":
        return f"iec61499://Device{scenario_tag}/Res1/FB{idx % 7}/OUT_TEMP"
    raise ValueError(f"Unsupported source standard: {standard}")


def _target_path(standard: str, idx: int, scenario_tag: str, variant: int = 0) -> str:
    if standard == "OPCUA":
        return f"opcua://ns=3;s=Plant.{scenario_tag}.Line{idx}.Temp{variant}"
    if standard == "AAS":
        suffix = "Value" if variant == 0 else f"Value{variant}"
        return f"aas://asset-{scenario_tag}-{idx}/submodel/Telemetry/element/Temperature/{suffix}"
    raise ValueError(f"Unsupported target standard: {standard}")


def _transform(transform_kind: str, idx: int) -> dict | None:
    if transform_kind == "unit_convert":
        return {"op": "format", "args": {"kind": "unit_convert", "from": "C", "to": "K", "offset": 273.15}}
    if transform_kind == "cast":
        return {"op": "cast", "args": {"kind": "cast", "to": "float", "round": idx % 3}}
    if transform_kind == "id_normalize":
        return {"op": "regex_extract", "args": {"kind": "id_normalize", "pattern": r"[A-Za-z0-9_-]+", "group": 0}}
    if transform_kind == "structural_map":
        return {"op": "concat", "args": {"kind": "structural_map", "fields": ["equipment", "channel"], "sep": "/"}}
    return None


def _artifact_payload(path: str, standard: str, idx: int, rng: random.Random) -> dict:
    synonyms = ["temperature", "temp", "process_temp"]
    rng.shuffle(synonyms)
    return {
        "id": path,
        "standard": standard,
        "signal": {
            "name": f"temperature_{idx}",
            "value": round(20.0 + rng.random() * 5.0, 3),
            "unit": "C",
        },
        "aliases": synonyms,
        "distractors": [
            {"name": f"pressure_{idx}", "unit": "bar"},
            {"name": f"humidity_{idx}", "unit": "%"},
        ],
    }


def _make_mapping_item(scenario: str, spec: ScenarioSpec, idx: int, scenario_tag: str, variant: int = 0) -> dict:
    source = _source_path(spec.source_standard, idx, scenario_tag)
    target = _target_path(spec.target_standard, idx, scenario_tag, variant=variant)
    transform = _transform(spec.transform_kind, idx)
    return {
        "source_path": source,
        "target_path": target,
        "mapping_type": "transform" if transform else "equivalent",
        "transform": transform,
        "confidence": 1.0,
        "rationale": f"Deterministic GT for {scenario} sample {idx}.",
        "evidence": [f"scenario:{scenario}", f"sample:{idx}"],
    }


def generate_pipeline(out_dir: Path, seed: int = 42, sample_size: int = 100) -> dict:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_rows: list[dict] = []
    dataset_rows: list[dict] = []

    for scenario_idx, scenario in enumerate(SCENARIOS):
        spec = SCENARIO_SPECS[scenario]
        scenario_tag = f"s{scenario_idx}"
        scenario_slug = scenario.replace("->", "_to_")
        src_dir = out_dir / "scenarios" / scenario_slug / "source"
        tgt_dir = out_dir / "scenarios" / scenario_slug / "target"
        src_dir.mkdir(parents=True, exist_ok=True)
        tgt_dir.mkdir(parents=True, exist_ok=True)

        per_source_count: dict[str, int] = defaultdict(int)
        for idx in range(sample_size):
            mapping = _make_mapping_item(scenario, spec, idx, scenario_tag, variant=0)
            gt_rows.append(mapping)
            per_source_count[mapping["source_path"]] += 1

            # structural mapping: emit one-to-many by adding an extra target per source.
            if spec.transform_kind == "structural_map":
                extra = _make_mapping_item(scenario, spec, idx, scenario_tag, variant=1)
                gt_rows.append(extra)
                per_source_count[mapping["source_path"]] += 1

            source_payload = _artifact_payload(mapping["source_path"], spec.source_standard, idx, rng)
            target_payload = _artifact_payload(mapping["target_path"], spec.target_standard, idx, rng)
            (src_dir / f"sample_{idx:04d}.json").write_text(json.dumps(source_payload, indent=2))
            (tgt_dir / f"sample_{idx:04d}.json").write_text(json.dumps(target_payload, indent=2))

        for idx in range(sample_size):
            source_path = _source_path(spec.source_standard, idx, scenario_tag)
            expected = per_source_count[source_path]
            dataset_rows.append(
                {
                    "id": f"{scenario_slug}:{idx:04d}",
                    "scenario": scenario,
                    "source_standard": spec.source_standard,
                    "target_standard": spec.target_standard,
                    "mapping_source_path": source_path,
                    "source_path": str(src_dir / f"sample_{idx:04d}.json"),
                    "target_path": str(tgt_dir / f"sample_{idx:04d}.json"),
                    "cardinality_contract": {
                        "mode": "one_to_one",
                        "grouped_1": False,
                        "expected_count": expected,
                    },
                }
            )

    normalized_gt = [ingest_mapping_payload(row, migrate_legacy=False).model_dump() for row in gt_rows]

    (out_dir / "ground_truth.jsonl").write_text("\n".join(json.dumps(r) for r in normalized_gt) + "\n")
    (out_dir / "dataset.jsonl").write_text("\n".join(json.dumps(r) for r in dataset_rows) + "\n")

    summary = {
        "seed": seed,
        "sample_size": sample_size,
        "scenario_count": len(SCENARIOS),
        "dataset_rows": len(dataset_rows),
        "gt_rows": len(normalized_gt),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark/data_gen synthetic cross-standard artifacts and GT.")
    parser.add_argument("--out-dir", default=os.getenv("BENCHMARK_DATA_GEN_OUT", "benchmark/data_gen/out"))
    parser.add_argument("--seed", type=int, default=int(os.getenv("BENCHMARK_DATA_GEN_SEED", "42")))
    parser.add_argument("--sample-size", type=int, default=int(os.getenv("BENCHMARK_DATA_GEN_SAMPLE_SIZE", "100")))
    args = parser.parse_args()

    summary = generate_pipeline(Path(args.out_dir), seed=args.seed, sample_size=args.sample_size)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
