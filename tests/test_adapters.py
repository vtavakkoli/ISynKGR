from isynkgr.adapters.aas import AASAdapter
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
