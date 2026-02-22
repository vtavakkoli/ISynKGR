from pathlib import Path

from isynkgr.translator import Translator

r = Translator().translate("aas", "opcua", "datasets/v1/aas/synthetic/aas_000.json", mode="hybrid")
Path("output_example_aas_to_opcua.json").write_text(r.model_dump_json(indent=2))
print("written output_example_aas_to_opcua.json")
