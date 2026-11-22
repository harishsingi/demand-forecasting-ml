import csv, json, argparse, pathlib
from src.features import add_features
ap=argparse.ArgumentParser(); ap.add_argument("--data",default="data/demand.csv"); ap.add_argument("--out",default="models/model.json"); a=ap.parse_args()
rows=list(csv.DictReader(open(a.data)))
by_part={}
for r in rows: by_part.setdefault(r["part"],[]).append(float(r["demand"]))
model={}
for part,series in by_part.items():
    feats=add_features(series); train,test=feats[:-90],feats[-90:]
    mae=sum(abs(f["value"]-f["roll28"]) for f in test)/len(test)
    model[part]={"baseline_roll28":train[-1]["roll28"],"backtest_mae":round(mae,2)}
pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True); pathlib.Path(a.out).write_text(json.dumps(model,indent=2)); print(model)
