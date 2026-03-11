from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from isynkgr.icr.mapping_output_contract import normalize_mapping_items
from isynkgr.llm.ollama import OllamaClient


@dataclass
class Sample:
    name: str
    source_protocol: str
    target_protocol: str
    source_path: str
    expected_target_path: str
    expected_mapping_type: str
    rationale_hint: str


SAMPLES: list[Sample] = [
    Sample("speed-equivalent", "opcua", "aas", "opcua://ns=2;s=Machine.Speed", "aas://machineA/submodel/default/element/speed/value", "equivalent", "same physical quantity"),
    Sample("temp-transform", "opcua", "aas", "opcua://ns=2;s=Machine.TempC", "aas://machineA/submodel/default/element/temperature/value", "transform", "temperature may need unit conversion"),
    Sample("pressure-equivalent", "opcua", "aas", "opcua://ns=2;s=Machine.Pressure", "aas://machineA/submodel/default/element/pressure/value", "equivalent", "same physical quantity"),
    Sample("current-equivalent", "opcua", "aas", "opcua://ns=2;s=Machine.Current", "aas://machineA/submodel/default/element/current/value", "equivalent", "same physical quantity"),
    Sample("voltage-equivalent", "opcua", "aas", "opcua://ns=2;s=Machine.Voltage", "aas://machineA/submodel/default/element/voltage/value", "equivalent", "same physical quantity"),
    Sample("status-no-match", "opcua", "aas", "opcua://ns=2;s=Machine.InternalDebugFlag", "", "no_match", "internal debug flag is not interoperable"),
    Sample("iec-to-aas-temp", "iec61499", "aas", "iec61499://deviceA/resource1/fbTemp/out", "aas://machineA/submodel/default/element/temperature/value", "equivalent", "same signal meaning"),
    Sample("ieee-to-opcua-humidity", "ieee1451", "opcua", "ieee1451://ted1/ch2/humidity", "opcua://ns=2;s=Environment.Humidity", "equivalent", "same physical quantity"),
    Sample("aas-to-opcua-power", "aas", "opcua", "aas://machineA/submodel/default/element/power/value", "opcua://ns=2;s=Machine.Power", "equivalent", "same physical quantity"),
    Sample("flow-transform", "opcua", "aas", "opcua://ns=2;s=Machine.FlowLpm", "aas://machineA/submodel/default/element/flow/value", "transform", "may require unit/scale normalization"),
]


def _prompt(sample: Sample) -> str:
    return (
        "You are generating industrial mapping output. "
        "Return ONLY valid JSON with this exact top-level shape: {\"mappings\": [ ... ]}. "
        "Do not include markdown.\n"
        f"Source protocol: {sample.source_protocol}\n"
        f"Target protocol: {sample.target_protocol}\n"
        f"Required source_path: {sample.source_path}\n"
        f"Expected target_path: {sample.expected_target_path or '<empty for no_match>'}\n"
        f"Expected mapping_type: {sample.expected_mapping_type}\n"
        f"Context: {sample.rationale_hint}\n"
        "Return one mapping with fields: source_path,target_path,mapping_type,confidence,rationale,evidence."
    )


def run(model: str, base_url: str | None, dry_run: bool = False) -> int:
    client = OllamaClient(model=model, base_url=base_url)
    passed = 0

    print(f"[simple-llm-test] model={model} base_url={client.base_url} samples={len(SAMPLES)}", flush=True)
    for idx, sample in enumerate(SAMPLES, start=1):
        expected = {
            "source_path": sample.source_path,
            "target_path": sample.expected_target_path,
            "mapping_type": sample.expected_mapping_type,
        }
        prompt = _prompt(sample)

        if dry_run:
            raw = {"mappings": [expected | {"confidence": 1.0, "rationale": "dry run", "evidence": []}]}
        else:
            raw = client.complete_json(prompt=prompt, schema_name="simple_llm_test", seed=4242 + idx)

        report = normalize_mapping_items(
            raw.get("mappings", []),
            source_protocol=sample.source_protocol,
            target_protocol=sample.target_protocol,
            method="llm",
        )

        observed = report.accepted[0].model_dump() if report.accepted else None
        error = report.rejected[0].model_dump() if report.rejected else None
        sample_pass = bool(observed) and observed["mapping_type"] == sample.expected_mapping_type
        if sample.expected_target_path:
            sample_pass = sample_pass and observed["target_path"] == sample.expected_target_path if observed else False
        else:
            sample_pass = sample_pass and observed["target_path"] == "" if observed else False

        if sample_pass:
            passed += 1

        print(f"\n=== sample {idx}/{len(SAMPLES)}: {sample.name} ===", flush=True)
        print("input:", json.dumps({"source_protocol": sample.source_protocol, "target_protocol": sample.target_protocol, "prompt": prompt}, indent=2), flush=True)
        print("expected_output:", json.dumps(expected, indent=2), flush=True)
        print("model_output:", json.dumps(raw, indent=2), flush=True)
        print("normalized_output:", json.dumps(observed, indent=2) if observed else "null", flush=True)
        if error:
            print("normalization_error:", json.dumps(error, indent=2), flush=True)
        print(f"result: {'PASS' if sample_pass else 'FAIL'}", flush=True)

    print(f"\n[simple-llm-test] passed={passed}/{len(SAMPLES)}", flush=True)
    capable = passed >= 8
    print(f"[simple-llm-test] capability verdict: {'CAPABLE' if capable else 'NOT_CAPABLE'}", flush=True)
    return 0 if capable else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5:0.8b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run(model=args.model, base_url=args.base_url, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
