import json
from pathlib import Path

from isynkgr.adapters.aas import AASAdapter
from isynkgr.adapters.iec61499 import IEC61499Adapter
from isynkgr.adapters.ieee1451 import IEEE1451Adapter
from isynkgr.adapters.opcua import OPCUAAdapter


def test_opcua_parse_and_validate():
    xml = """<UANodeSet><UAObjectType NodeId='ns=1;i=1' BrowseName='A'><DisplayName>A</DisplayName><References><Reference>ns=1;i=2</Reference></References></UAObjectType><UAVariable NodeId='ns=1;i=2' BrowseName='B'><DisplayName>B</DisplayName></UAVariable></UANodeSet>"""
    ad = OPCUAAdapter()
    model = ad.parse(xml)
    assert len(model.nodes) == 2
    assert ad.validate(xml).valid


def test_aas_parse_and_validate():
    doc = {"assetAdministrationShells": [{"id": "aas-1", "submodels": [{"keys": [{"value": "sm-1"}]}]}], "submodels": [{"id": "sm-1", "submodelElements": []}]}
    ad = AASAdapter()
    model = ad.parse(doc)
    assert model.nodes
    assert ad.validate(doc).valid


def test_iec61499_happy_path_parse_serialize_and_validate():
    fixture = json.loads(Path("datasets/v2/iec61499/fixtures/happy.json").read_text())
    ad = IEC61499Adapter()

    model = ad.parse(fixture)
    assert model.standard == "iec61499"
    assert any(n.type == "FunctionBlock" for n in model.nodes)

    report = ad.validate(fixture)
    assert report.valid

    serialized = ad.serialize(model)
    assert serialized["standard"] == "iec61499"
    assert serialized["devices"][0]["resources"][0]["function_blocks"][0]["id"] == "fb-temp"


def test_iec61499_constraint_violations():
    fixture = json.loads(Path("datasets/v2/iec61499/fixtures/invalid_constraints.json").read_text())
    report = IEC61499Adapter().validate(fixture)

    assert not report.valid
    messages = [v.message for v in report.violations]
    assert any("dtype invalid" in m for m in messages)
    assert any("unit must be a non-empty string" in m for m in messages)
    assert any("min cannot be greater than max" in m for m in messages)


def test_ieee1451_happy_path_parse_serialize_and_validate():
    fixture = json.loads(Path("datasets/v2/ieee1451/fixtures/happy.json").read_text())
    ad = IEEE1451Adapter()

    model = ad.parse(fixture)
    assert model.standard == "ieee1451"
    assert any(n.type == "Channel" for n in model.nodes)

    report = ad.validate(fixture)
    assert report.valid

    serialized = ad.serialize(model)
    assert serialized["standard"] == "ieee1451"
    assert serialized["teds"][0]["channels"][0]["id"] == "ch-1"


def test_ieee1451_constraint_violations():
    fixture = json.loads(Path("datasets/v2/ieee1451/fixtures/invalid_constraints.json").read_text())
    report = IEEE1451Adapter().validate(fixture)

    assert not report.valid
    messages = [v.message for v in report.violations]
    assert any("range required for numeric dtype" in m for m in messages)
    assert any("duplicate channel id" in m for m in messages)
    assert any("range allowed only for numeric dtype" in m for m in messages)
