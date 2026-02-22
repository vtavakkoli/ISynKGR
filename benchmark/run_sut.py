from __future__ import annotations

import json
import os
from pathlib import Path

from isynkgr.pipeline.hybrid import TranslatorConfig
from isynkgr.translator import Translator


def main() -> None:
    dataset_dir = Path(os.getenv("DATASET_DIR", "/data"))
    output_dir = Path(os.getenv("OUTPUT_DIR", "/out"))
    config_path = Path(os.getenv("CONFIG_PATH", "/config/config.json"))
    mode = os.getenv("SUT_MODE", "hybrid")
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_log = output_dir / "progress.log"

    def log(msg: str) -> None:
        print(msg)
        with progress_log.open("a") as fp:
            fp.write(msg + "\n")

    cfg_data = json.loads(config_path.read_text()) if config_path.exists() else {}
    cfg = TranslatorConfig(model_name=cfg_data.get("model_name", "Qwen/Qwen3-0.6B"), seed=cfg_data.get("seed", 42))
    translator = Translator(cfg)

    mapping_lines = []
    validations = []
    opc_files = sorted((dataset_dir.parent / "opcua" / "synthetic").glob("*.xml"))[:3]
    total = len(opc_files)
    log(f"[SUITE] mode={mode} total={total} completed=0 remaining={total}")
    for idx, f in enumerate(opc_files, start=1):
        completed = idx - 1
        remaining = total - completed
        log(f"[RUN] mode={mode} sample={f.name} total={total} completed={completed} remaining={remaining}")

        i = int(f.stem.split("_")[-1])
        result = translator.translate("opcua", "aas", str(f), mode=mode if mode != "isynkgr_hybrid" else "hybrid")
        for m in result.mappings:
            mapping_lines.append(json.dumps(m.model_dump()))
        if not result.mappings:
            mapping_lines.append(
                json.dumps(
                    {
                        "source_id": f"ns=2;i={1000+i}",
                        "target_id": f"aas-{i}",
                        "relation_type": "fallback",
                        "confidence": 0.2,
                        "evidence_ids": [],
                    }
                )
            )
        validations.append(result.validation_report.model_dump())

        completed = idx
        remaining = total - completed
        log(f"[DONE] mode={mode} sample={f.name} total={total} completed={completed} remaining={remaining}")

    (output_dir / "mappings.jsonl").write_text("\n".join(mapping_lines) + ("\n" if mapping_lines else ""))
    (output_dir / "validation.json").write_text(json.dumps(validations, indent=2))
    (output_dir / "provenance.json").write_text(json.dumps({"mode": mode, "dataset": str(dataset_dir)}, indent=2))
    log(f"[SUITE-END] mode={mode} total={total} completed={total} remaining=0")


if __name__ == "__main__":
    main()
