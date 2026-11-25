"""Tests for leakage-free feature engineering."""

import pytest

from demand_forecast.data import DemandRecord
from demand_forecast.features import FeatureRow, add_features, series_by_part_region
from datetime import date


def test_no_leakage_lag_uses_only_past():
    features = add_features([10, 20, 30])
    assert features[0].lag_1 == 10  # first day falls back to its own value
    assert features[1].lag_1 == 10
    assert features[2].lag_1 == 20


def test_rolling_mean_uses_only_past():
    features = add_features([10, 20, 30], window=28)
    assert features[0].rolling_mean == 10
    assert features[1].rolling_mean == 10
    assert features[2].rolling_mean == 15  # mean of [10, 20], not [10, 20, 30]


def test_rolling_window_bounds_history():
    features = add_features([1, 2, 3, 4, 5], window=2)
    assert features[4].rolling_mean == 3.5  # mean of [3, 4]
    assert features[4].lag_1 == 4


def test_empty_series_returns_empty():
    assert add_features([]) == []


def test_single_value_series():
    features = add_features([42])
    assert features == [FeatureRow(lag_1=42, rolling_mean=42, value=42)]


def test_window_must_be_positive():
    with pytest.raises(ValueError, match="window must be >= 1"):
        add_features([1, 2, 3], window=0)


def test_non_finite_value_rejected():
    with pytest.raises(ValueError, match="not a finite number"):
        add_features([1, float("nan"), 3])
    with pytest.raises(ValueError, match="not a finite number"):
        add_features([1, float("inf"), 3])


def test_non_numeric_value_rejected():
    with pytest.raises(ValueError, match="not a finite number"):
        add_features([1, "two", 3])  # type: ignore[list-item]


def test_series_grouped_and_sorted_by_date():
    records = [
        DemandRecord(date(2023, 1, 2), "filter", "North", 5.0),
        DemandRecord(date(2023, 1, 1), "filter", "North", 4.0),
        DemandRecord(date(2023, 1, 1), "filter", "South", 9.0),
    ]
    grouped = series_by_part_region(records)
    assert set(grouped) == {("filter", "North"), ("filter", "South")}
    dates, values = grouped[("filter", "North")]
    assert dates == [date(2023, 1, 1), date(2023, 1, 2)]
    assert values == [4.0, 5.0]
