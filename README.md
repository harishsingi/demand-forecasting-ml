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
<details>
<summary><b>Feature engineering + backtest — click to expand the 4 steps</b></summary>

Collapsed, this is just: Feature engineering -> Train -> Backtest 90 days -> Score next 30 days

1. Lag demand, rolling means and seasonality are built only from past data, so there is no leakage.
2. A transparent baseline and seasonal model is trained, so the method can be explained.
3. Backtest: the last 90 days are held out and scored with MAE, so accuracy is measured honestly.
4. Scores forecast the next 30 days by part and region, with a plain-English driver summary.
</details>

### Earlier issues, risk, and impact after

| Aspect | Detail |
|---|---|
| Earlier issues | Forecasts kept in spreadsheets and gut feel, hard to repeat or check; historical data dumped by hand, stale before it is used; features built once by hand, so they drift and cannot be reused; no held-out backtest; a model built once with no scoring entry point, so it never reached the team; no driver summary, so teams could not tell why a forecast moved. |
| Risk | A model could look good by accidentally using future data; a forecast nobody could check or repeat could drive a real planning decision; features could drift silently. |
| Impact after | Forecasts start from curated lakehouse data; features use only past data, so no leakage; the last 90 days are held out and scored with MAE; every forecast ships with a plain-English driver summary and reaches dashboards and planning teams. |

