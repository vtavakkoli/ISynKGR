from benchmark.metrics import prf1


def test_metric_non_identity():
    out = prf1({("s1", "t1")}, {("s1", "t2")})
    assert out["f1"] == 0.0
