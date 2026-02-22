from __future__ import annotations

import json
from pathlib import Path


def make_aas(i: int) -> dict:
    return {
        "assetAdministrationShells": [{"id": f"aas-{i}", "idShort": f"PumpAAS{i}", "submodels": [{"keys": [{"value": f"sm-{i}"}]}]}],
        "submodels": [{"id": f"sm-{i}", "idShort": f"Telemetry{i}", "submodelElements": [{"idShort": "pressure", "modelType": "Property", "valueType": "double", "value": str(10+i)}]}],
    }


def main() -> None:
    root = Path("datasets/v1/aas/synthetic")
    root.mkdir(parents=True, exist_ok=True)
    for i in range(100):
        (root / f"aas_{i:03d}.json").write_text(json.dumps(make_aas(i), indent=2))


if __name__ == "__main__":
    main()
