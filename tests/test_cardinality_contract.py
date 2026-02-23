from benchmark.run import _cardinality_contract_for_sample


def test_cardinality_contract_defaults_to_one_to_one():
    contract = _cardinality_contract_for_sample({})
    assert contract == {"mode": "one_to_one", "expected_count": 1, "grouped_1": False}


def test_cardinality_contract_grouped_from_metadata():
    contract = _cardinality_contract_for_sample({"metadata": {"grouped_1": True}})
    assert contract["mode"] == "grouped_1"
    assert contract["grouped_1"] is True
