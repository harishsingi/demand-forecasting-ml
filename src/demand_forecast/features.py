"""Leakage-free feature engineering for daily demand series.

Every feature for a given day is built only from observations strictly
before that day, so training can never peek at the future. The one
exception is the first day of a series, where there is no history yet:
its lag and rolling mean fall back to its own value, and the model
training step requires enough history that this fallback never drives a
forecast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .data import DemandRecord


@dataclass(frozen=True)
class FeatureRow:
    """Features for one day of one demand series."""

    lag_1: float
    rolling_mean: float
    value: float


def add_features(series: list[float], window: int = 28) -> list[FeatureRow]:
    """Build lag / rolling-mean features for a date-ordered series.

    Args:
        series: Demand values in date order.
        window: Number of past days in the rolling mean (``>= 1``).

    Returns:
        One :class:`FeatureRow` per input value. ``lag_1`` is the
        previous day's value and ``rolling_mean`` is the mean of the
        previous ``window`` days; both use only past data.

    Raises:
        ValueError: If ``window`` is < 1, or a value is not a finite
            number.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    features: list[FeatureRow] = []
    for i, value in enumerate(series):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"series[{i}] is not a finite number: {value!r}")
        value = float(value)
        lag_1 = float(series[i - 1]) if i >= 1 else value
        history = series[max(0, i - window) : i] or [value]
        rolling_mean = sum(float(v) for v in history) / len(history)
        features.append(
            FeatureRow(lag_1=lag_1, rolling_mean=rolling_mean, value=value)
        )
    return features


def series_by_part_region(
    records: list[DemandRecord],
) -> dict[tuple[str, str], tuple[list, list[float]]]:
    """Split validated records into date-ordered series per (part, region).

    Returns:
        Mapping of ``(part, region)`` to ``(dates, values)`` where both
        lists are sorted by date.
    """
    grouped: dict[tuple[str, str], list[DemandRecord]] = {}
    for record in records:
        grouped.setdefault((record.part, record.region), []).append(record)
    result: dict[tuple[str, str], tuple[list, list[float]]] = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda r: r.date)
        result[key] = ([r.date for r in rows], [r.demand for r in rows])
    return result
