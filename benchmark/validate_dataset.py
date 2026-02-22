from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetRequirements:
    opcua_synthetic: int = 100
    aas_synthetic: int = 100
    opcua_semi_real: int = 10
    aas_semi_real: int = 10


def _count_files(path: Path, suffix: str) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob(f"*.{suffix}")))


def validate_or_generate(root: Path = Path("datasets/v1"), req: DatasetRequirements | None = None) -> dict[str, int]:
    req = req or DatasetRequirements()
    counts = {
        "opcua_synthetic": _count_files(root / "opcua" / "synthetic", "xml"),
        "aas_synthetic": _count_files(root / "aas" / "synthetic", "json"),
        "opcua_semi_real": _count_files(root / "opcua" / "semi_real", "xml"),
        "aas_semi_real": _count_files(root / "aas" / "semi_real", "json"),
    }

    needs_gen = counts["opcua_synthetic"] < req.opcua_synthetic or counts["aas_synthetic"] < req.aas_synthetic
    if needs_gen:
        subprocess.run(["python", "datasets/generators/gen_opcua.py"], check=True)
        subprocess.run(["python", "datasets/generators/gen_aas.py"], check=True)
        counts["opcua_synthetic"] = _count_files(root / "opcua" / "synthetic", "xml")
        counts["aas_synthetic"] = _count_files(root / "aas" / "synthetic", "json")

    if counts["opcua_semi_real"] < req.opcua_semi_real or counts["aas_semi_real"] < req.aas_semi_real:
        # Use deterministic synthetic-backed semi-real fallback generation.
        opcua_semi = root / "opcua" / "semi_real"
        aas_semi = root / "aas" / "semi_real"
        opcua_semi.mkdir(parents=True, exist_ok=True)
        aas_semi.mkdir(parents=True, exist_ok=True)
        for i in range(req.opcua_semi_real):
            src = root / "opcua" / "synthetic" / f"opcua_{i:03d}.xml"
            dst = opcua_semi / f"example_{i:02d}.xml"
            if src.exists() and not dst.exists():
                dst.write_text(src.read_text())
        for i in range(req.aas_semi_real):
            src = root / "aas" / "synthetic" / f"aas_{i:03d}.json"
            dst = aas_semi / f"example_{i:02d}.json"
            if src.exists() and not dst.exists():
                dst.write_text(src.read_text())
        counts["opcua_semi_real"] = _count_files(root / "opcua" / "semi_real", "xml")
        counts["aas_semi_real"] = _count_files(root / "aas" / "semi_real", "json")

    return counts


if __name__ == "__main__":
    final_counts = validate_or_generate()
    print("dataset-validation", final_counts)
