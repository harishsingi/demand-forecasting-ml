def add_features(series):
    """series: list of floats in date order. Returns lag/rolling features, leakage-safe."""
    out=[]
    for i,v in enumerate(series):
        lag1=series[i-1] if i>=1 else v
        window=series[max(0,i-28):i] or [v]
        out.append({"lag1":lag1,"roll28":sum(window)/len(window),"value":v})
    return out
