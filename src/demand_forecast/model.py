"""Transparent baseline + seasonal forecast model.

The model is deliberately explainable rather than powerful:

* **Baseline**: the rolling mean of the most recent ``rolling_window``
  training days.
* **Seasonality**: a per-month multiplier, computed from the training
  period only as ``mean demand in that month / overall training mean``.
  Months with no training history default to 1.0 (no seasonal effect).
* **Forecast**: ``baseline * seasonal factor of the forecast month``,
  one value per day for the horizon.

Backtesting is walk-forward on the last ``backtest_days`` days: each
held-out day is forecast from history strictly before it (rolling mean
over past days, seasonal index from training only), and MAE is reported.
No future data is used anywhere, so the backtest MAE is an honest
measure of this simple model on the held-out period. It is not a claim
about accuracy on real data; on the synthetic demo data it is whatever
the run computes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

from .config import ForecastConfig
from .exceptions import ModelError

logger = logging.getLogger(__name__)

MODEL_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ForecastDay:
    """One day of forecast output."""

    date: date
    forecast: float
    seasonal_factor: float


@dataclass
class BaselineSeasonalModel:
    """A fitted baseline + monthly-seasonal model for one part/region."""

    part: str
    region: str
    baseline: float
    seasonal_index: dict[int, float]
    backtest_mae: float
    history_start: date
    history_end: date
    history_days: int
    rolling_window: int
    backtest_days: int

    @classmethod
    def fit(
        cls,
        part: str,
        region: str,
        dates: list[date],
        values: list[float],
        config: ForecastConfig,
    ) -> "BaselineSeasonalModel":
        """Fit the model on one date-ordered series.

        The last ``config.backtest_days`` days are held out for the
        walk-forward backtest; the rest is the training period.

        Raises:
            ModelError: If the series is shorter than
                ``min_history_days`` or not longer than the backtest
                window, or if dates and values disagree in length.
        """
        if len(dates) != len(values):
            raise ModelError(
                f"{part!r} / {region!r}: dates ({len(dates)}) and values "
                f"({len(values)}) disagree in length"
            )
        if len(values) < config.min_history_days:
            raise ModelError(
                f"{part!r} / {region!r}: only {len(values)} day(s) of history, "
                f"need at least {config.min_history_days}"
            )
        if len(values) <= config.backtest_days:
            raise ModelError(
                f"{part!r} / {region!r}: {len(values)} days of history does "
                f"not exceed the {config.backtest_days}-day backtest window"
            )
        train_values = values[: -config.backtest_days]
        train_dates = dates[: -config.backtest_days]
        baseline = sum(train_values[-config.rolling_window :]) / min(
            config.rolling_window, len(train_values)
        )
        seasonal_index = _monthly_seasonal_index(train_dates, train_values)
        backtest_mae = _walk_forward_mae(
            dates, values, config, seasonal_index
        )
        model = cls(
            part=part,
            region=region,
            baseline=baseline,
            seasonal_index=seasonal_index,
            backtest_mae=backtest_mae,
            history_start=dates[0],
            history_end=dates[-1],
            history_days=len(values),
            rolling_window=config.rolling_window,
            backtest_days=config.backtest_days,
        )
        logger.info(
            "Fitted %r / %r: baseline=%.1f backtest_mae=%.2f history=%d days",
            part,
            region,
            baseline,
            backtest_mae,
            len(values),
        )
        return model

    def forecast(self, horizon: int) -> list[ForecastDay]:
        """Forecast ``horizon`` days starting the day after history ends.

        Raises:
            ModelError: If ``horizon`` is < 1.
        """
        if horizon < 1:
            raise ModelError(f"horizon must be >= 1, got {horizon}")
        days: list[ForecastDay] = []
        for offset in range(1, horizon + 1):
            day = self.history_end + timedelta(days=offset)
            factor = self.seasonal_index.get(day.month, 1.0)
            days.append(
                ForecastDay(
                    date=day,
                    forecast=round(self.baseline * factor, 2),
                    seasonal_factor=round(factor, 3),
                )
            )
        return days

    def driver_summary(self, horizon: int) -> str:
        """Plain-English summary of what drives this model's forecast."""
        days = self.forecast(horizon)
        average = sum(d.forecast for d in days) / len(days)
        first_month = days[0].date.strftime("%B")
        factor = days[0].seasonal_factor
        return (
            f"Forecast for {self.part!r} in {self.region!r} averages about "
            f"{average:.0f} units/day over the next {horizon} days. "
            f"The recent {self.rolling_window}-day average is "
            f"{self.baseline:.0f} units/day, and {first_month} seasonality "
            f"multiplies that by {factor:.2f}. Held-out backtest MAE over "
            f"the last {self.backtest_days} days was {self.backtest_mae:.1f} "
            f"units."
        )

    def to_dict(self) -> dict:
        """Serialise the fitted model to a JSON-safe dict."""
        return {
            "part": self.part,
            "region": self.region,
            "baseline": self.baseline,
            "seasonal_index": {str(m): f for m, f in self.seasonal_index.items()},
            "backtest_mae": self.backtest_mae,
            "history_start": self.history_start.isoformat(),
            "history_end": self.history_end.isoformat(),
            "history_days": self.history_days,
            "rolling_window": self.rolling_window,
            "backtest_days": self.backtest_days,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "BaselineSeasonalModel":
        """Rebuild a model from :meth:`to_dict` output.

        Raises:
            ModelError: If required fields are missing or malformed.
        """
        required = {
            "part",
            "region",
            "baseline",
            "seasonal_index",
            "backtest_mae",
            "history_start",
            "history_end",
            "history_days",
            "rolling_window",
            "backtest_days",
        }
        missing = required - set(raw)
        if missing:
            raise ModelError(f"Model dict is missing fields: {sorted(missing)}")
        try:
            seasonal_index = {
                int(month): float(factor)
                for month, factor in raw["seasonal_index"].items()
            }
            return cls(
                part=str(raw["part"]),
                region=str(raw["region"]),
                baseline=float(raw["baseline"]),
                seasonal_index=seasonal_index,
                backtest_mae=float(raw["backtest_mae"]),
                history_start=date.fromisoformat(raw["history_start"]),
                history_end=date.fromisoformat(raw["history_end"]),
                history_days=int(raw["history_days"]),
                rolling_window=int(raw["rolling_window"]),
                backtest_days=int(raw["backtest_days"]),
            )
        except (ValueError, TypeError, AttributeError) as exc:
            raise ModelError(f"Model dict is malformed: {exc}") from exc


def _monthly_seasonal_index(
    dates: list[date], values: list[float]
) -> dict[int, float]:
    """Per-month multiplier from training data only.

    ``mean(month) / overall mean`` for each month present in training;
    months with no training data are absent (callers default them to 1.0).
    """
    overall = sum(values) / len(values)
    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    for day, value in zip(dates, values):
        totals[day.month] = totals.get(day.month, 0.0) + value
        counts[day.month] = counts.get(day.month, 0) + 1
    if overall == 0:
        return {}
    return {
        month: (totals[month] / counts[month]) / overall for month in totals
    }


def _walk_forward_mae(
    dates: list[date],
    values: list[float],
    config: ForecastConfig,
    seasonal_index: dict[int, float],
) -> float:
    """MAE over the held-out window, forecasting each day from past only."""
    test_start = len(values) - config.backtest_days
    errors: list[float] = []
    for i in range(test_start, len(values)):
        history = values[max(0, i - config.rolling_window) : i]
        rolling = sum(history) / len(history)
        factor = seasonal_index.get(dates[i].month, 1.0)
        errors.append(abs(values[i] - rolling * factor))
    return sum(errors) / len(errors)
