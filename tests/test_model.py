"""Tests for the baseline + seasonal model and its backtest."""

from datetime import date, timedelta

import pytest

from demand_forecast.config import ForecastConfig
from demand_forecast.exceptions import ModelError
from demand_forecast.model import BaselineSeasonalModel


def _flat_series(days: int = 200, value: float = 100.0):
    start = date(2023, 1, 1)
    dates = [start + timedelta(days=i) for i in range(days)]
    return dates, [value] * days


def test_fit_flat_series_baseline_and_zero_mae():
    dates, values = _flat_series()
    model = BaselineSeasonalModel.fit(
        "filter", "North", dates, values, ForecastConfig()
    )
    assert model.baseline == pytest.approx(100.0)
    assert model.backtest_mae == pytest.approx(0.0)
    assert model.history_days == 200


def test_forecast_length_and_start_day():
    dates, values = _flat_series()
    model = BaselineSeasonalModel.fit(
        "filter", "North", dates, values, ForecastConfig()
    )
    days = model.forecast(30)
    assert len(days) == 30
    assert days[0].date == date(2023, 7, 20)  # day after 200 days from Jan 1
    assert all(d.forecast == pytest.approx(100.0) for d in days)


def test_forecast_uses_monthly_seasonality():
    # January values high, July values low: the seasonal index must differ.
    dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(200)]
    values = [200.0 if d.month == 1 else 100.0 for d in dates]
    model = BaselineSeasonalModel.fit(
        "filter", "North", dates, values, ForecastConfig()
    )
    assert model.seasonal_index[1] > model.seasonal_index.get(7, 1.0)
    january = model.forecast(31)
    assert january[0].date.month == 7  # forecast starts in July
    # July has no training history, so its seasonal factor defaults to 1.0.
    assert january[0].seasonal_factor == pytest.approx(1.0)
    assert january[0].forecast == pytest.approx(model.baseline)


def test_backtest_mae_is_honest_on_trending_series():
    dates, values = _flat_series()
    values = [100.0 + i * 0.5 for i in range(200)]  # steady upward trend
    model = BaselineSeasonalModel.fit(
        "filter", "North", dates, values, ForecastConfig()
    )
    # A rolling-mean baseline lags a trend, so MAE must be clearly > 0.
    assert model.backtest_mae > 1.0


def test_short_series_rejected():
    dates, values = _flat_series(days=50)
    with pytest.raises(ModelError, match="need at least"):
        BaselineSeasonalModel.fit(
            "filter", "North", dates, values, ForecastConfig()
        )


def test_series_not_longer_than_backtest_rejected():
    dates, values = _flat_series(days=90)
    config = ForecastConfig(min_history_days=30, backtest_days=90)
    with pytest.raises(ModelError, match="does not exceed"):
        BaselineSeasonalModel.fit("filter", "North", dates, values, config)


def test_mismatched_dates_and_values_rejected():
    dates, values = _flat_series(days=200)
    with pytest.raises(ModelError, match="disagree in length"):
        BaselineSeasonalModel.fit(
            "filter", "North", dates, values[:-1], ForecastConfig()
        )


def test_forecast_horizon_must_be_positive():
    dates, values = _flat_series()
    model = BaselineSeasonalModel.fit(
        "filter", "North", dates, values, ForecastConfig()
    )
    with pytest.raises(ModelError, match="horizon must be >= 1"):
        model.forecast(0)


def test_to_dict_from_dict_round_trip():
    dates, values = _flat_series()
    model = BaselineSeasonalModel.fit(
        "filter", "North", dates, values, ForecastConfig()
    )
    restored = BaselineSeasonalModel.from_dict(model.to_dict())
    assert restored == model


def test_from_dict_missing_field_rejected():
    with pytest.raises(ModelError, match="missing fields"):
        BaselineSeasonalModel.from_dict({"part": "filter"})


def test_from_dict_malformed_rejected():
    dates, values = _flat_series()
    model = BaselineSeasonalModel.fit(
        "filter", "North", dates, values, ForecastConfig()
    )
    raw = model.to_dict()
    raw["history_start"] = "not-a-date"
    with pytest.raises(ModelError, match="malformed"):
        BaselineSeasonalModel.from_dict(raw)


def test_driver_summary_mentions_key_facts():
    dates, values = _flat_series()
    model = BaselineSeasonalModel.fit(
        "filter", "North", dates, values, ForecastConfig()
    )
    summary = model.driver_summary(30)
    assert "filter" in summary
    assert "North" in summary
    assert "30 days" in summary
    assert "backtest MAE" in summary
