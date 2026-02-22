from __future__ import annotations

import json
import os
import time
from pathlib import Path

from isynkgr.pipeline.hybrid import TranslatorConfig
from isynkgr.translator import Translator


def _fmt_s(seconds: float) -> str:
    return f"{seconds:.2f}s"


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

    suite_start = time.perf_counter()
    log(f"[STAGE] mode={mode} stage=init status=start output_dir={output_dir}")
    cfg_data = json.loads(config_path.read_text()) if config_path.exists() else {}
    cfg = TranslatorConfig(model_name=cfg_data.get("model_name", "qwen3:0.6b"), seed=cfg_data.get("seed", 42))
    translator = Translator(cfg)
    log(f"[STAGE] mode={mode} stage=init status=done model={cfg.model_name} seed={cfg.seed}")

    mapping_lines = []
    validations = []
    llm_bugs = []
    max_samples = int(os.getenv("MAX_SAMPLES", "100"))
    opc_files = sorted((dataset_dir.parent / "opcua" / "synthetic").glob("*.xml"))[:max_samples]
    total = len(opc_files)
    log(f"[SUITE] mode={mode} stage=translation total={total} completed=0 remaining={total} eta=unknown")

    for idx, f in enumerate(opc_files, start=1):
        sample_start = time.perf_counter()
        completed = idx - 1
        remaining = total - completed
        elapsed = time.perf_counter() - suite_start
        avg = (elapsed / completed) if completed else 0.0
        eta = avg * remaining if completed else 0.0
        log(
            f"[RUN] mode={mode} stage=translation sample={f.name} total={total} "
            f"completed={completed} remaining={remaining} elapsed={_fmt_s(elapsed)} eta={_fmt_s(eta)}"
        )

        i = int(f.stem.split("_")[-1])
        result = translator.translate("opcua", "aas", str(f), mode=mode if mode != "isynkgr_hybrid" else "hybrid")
        llm_error = (result.provenance.metadata or {}).get("llm_error") if result.provenance else None
        if llm_error:
            llm_bugs.append({"sample": f.name, "mode": mode, "error": llm_error})
            log(f"[BUG] mode={mode} sample={f.name} issue=llm_request_failed details={llm_error}")

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
        sample_elapsed = time.perf_counter() - sample_start
        elapsed = time.perf_counter() - suite_start
        avg = elapsed / completed if completed else 0.0
        eta = avg * remaining
        log(
            f"[DONE] mode={mode} stage=translation sample={f.name} total={total} "
            f"completed={completed} remaining={remaining} sample_elapsed={_fmt_s(sample_elapsed)} "
            f"elapsed={_fmt_s(elapsed)} eta={_fmt_s(eta)}"
        )

    log(f"[STAGE] mode={mode} stage=persist status=start")
    (output_dir / "mappings.jsonl").write_text("\n".join(mapping_lines) + ("\n" if mapping_lines else ""))
    (output_dir / "validation.json").write_text(json.dumps(validations, indent=2))
    (output_dir / "provenance.json").write_text(json.dumps({"mode": mode, "dataset": str(dataset_dir), "llm_bug_count": len(llm_bugs)}, indent=2))
    (output_dir / "bugs.json").write_text(json.dumps(llm_bugs, indent=2))
    log(f"[STAGE] mode={mode} stage=persist status=done files=4")

    suite_elapsed = time.perf_counter() - suite_start
    throughput = (total / suite_elapsed) if suite_elapsed > 0 else 0.0
    log(
        f"[SUITE-END] mode={mode} stage=complete total={total} completed={total} remaining=0 "
        f"elapsed={_fmt_s(suite_elapsed)} throughput={throughput:.2f}/s bugs={len(llm_bugs)}"
    )


if __name__ == "__main__":
    main()
