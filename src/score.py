import json, argparse
ap=argparse.ArgumentParser(); ap.add_argument("--model",default="models/model.json"); ap.add_argument("--part",required=True); a=ap.parse_args()
m=json.load(open(a.model)); print({"part":a.part,"forecast_next_30d_per_day":m[a.part]["baseline_roll28"]})
