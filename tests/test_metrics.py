from benchmark.metrics import prf1


def test_prf1():
    scores = prf1({("a", "b"), ("c", "d")}, {("a", "b")})
    assert scores["precision"] == 0.5
    assert scores["recall"] == 1.0
