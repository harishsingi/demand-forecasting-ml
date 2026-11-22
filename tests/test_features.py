from src.features import add_features
def test_no_leakage():
    f=add_features([10,20,30]); assert f[1]["lag1"]==10 and f[2]["roll28"]==15
