from pathlib import Path

from isynkgr.translator import Translator

r = Translator().translate("opcua", "aas", "datasets/v1/opcua/synthetic/opcua_000.xml", mode="hybrid")
Path("output_example_opcua_to_aas.json").write_text(r.model_dump_json(indent=2))
print("written output_example_opcua_to_aas.json")
