# Demand Forecasting with ML

Portfolio implementation based on production work experience. All data is synthetic, all code is original and clean-room, and no proprietary code, data, schema names, or internal system details from any employer are included.

**Original context:** An ML product for internal teams to predict parts and service demand, improving forecast accuracy and cutting costs, built on top of curated lakehouse data.

**This portfolio version:** Synthetic demand history, a transparent baseline + seasonal model in pure Python (no heavy ML dependencies to install), feature engineering, backtesting, and a scoring entry point. In production this pattern was implemented with PySpark feature pipelines at petabyte scale data volumes across sources.

## Run
```bash
python -m src.generate_data
python -m src.train --data data/demand.csv --out models/model.json
python -m src.score --model models/model.json --part "brake pad"
python -m pytest tests/ -q
```
## Approach
- Features: lag demand, rolling means, seasonality by month, dealer region mix
- Backtest: last 90 days held out, MAE reported, no leakage (features use only past data)
- Output: per part, per region forecast for the next 30 days, with a plain-English driver summary

## Architecture

![ML demand forecasting architecture](docs/ml-demand-forecasting.svg)

### Before / After

Hover any component to see what it is and why it is needed.

![Before / After architecture](docs/before-after-elaborated.svg)
