"""Configuration for the demand-forecasting pipeline.

All tunables live here. Pipeline modules take a ``ForecastConfig`` instead
of reading module-level constants, so behaviour is explicit, testable, and
overridable from a JSON file (``--config``) or from CLI flags.

There is intentionally no network configuration in this package: the
pipeline reads local CSV files and writes local files, so there are no
API calls to retry and no credentials anywhere in the code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .exceptions import ConfigError

# Synthetic catalogue. These are invented product/region labels for the
# generated demo data only; they are not real products, customers, or
# locations from any employer.
DEFAULT_PART_BASES: dict[str, float] = {
    "brake pad": 100.0,
    "filter": 160.0,
    "battery": 80.0,
}
DEFAULT_REGION_MULTIPLIERS: dict[str, float] = {
    "North": 1.0,
    "Central": 0.85,
    "South": 1.15,
}
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_DAYS = 540
DEFAULT_SEED = 7
DEFAULT_ROLLING_WINDOW = 28
DEFAULT_BACKTEST_DAYS = 90
DEFAULT_FORECAST_HORIZON = 30
DEFAULT_MIN_HISTORY_DAYS = 120
DEFAULT_SEASONAL_AMPLITUDE = 0.2
DEFAULT_NOISE_STDDEV = 10.0

# Keys accepted in a JSON config file. Anything else is rejected so a
# typo fails loudly instead of being silently ignored.
_CONFIG_KEYS = {
    "part_bases",
    "region_multipliers",
    "start_date",
    "days",
    "seed",
    "rolling_window",
    "backtest_days",
    "forecast_horizon",
    "min_history_days",
    "seasonal_amplitude",
    "noise_stddev",
}


@dataclass(frozen=True)
class ForecastConfig:
    """All settings for data generation, training, and scoring.

    Attributes:
        part_bases: Base daily demand per synthetic part label.
        region_multipliers: Demand multiplier per synthetic region label.
        start_date: First date of generated history (ISO ``YYYY-MM-DD``).
        days: Number of days of history to generate.
        seed: Random seed. Same seed + same config = identical output,
            so data generation is reproducible (idempotent).
        rolling_window: Days in the rolling-mean baseline.
        backtest_days: Most recent days held out for backtesting.
        forecast_horizon: Days ahead that ``score`` forecasts.
        min_history_days: Minimum history a series needs to train.
        seasonal_amplitude: Size of the synthetic annual seasonal wave
            (0 = flat, 1 = demand swings to zero at the trough).
        noise_stddev: Standard deviation of the synthetic daily noise.
    """

    part_bases: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PART_BASES)
    )
    region_multipliers: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_REGION_MULTIPLIERS)
    )
    start_date: str = DEFAULT_START_DATE
    days: int = DEFAULT_DAYS
    seed: int = DEFAULT_SEED
    rolling_window: int = DEFAULT_ROLLING_WINDOW
    backtest_days: int = DEFAULT_BACKTEST_DAYS
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS
    seasonal_amplitude: float = DEFAULT_SEASONAL_AMPLITUDE
    noise_stddev: float = DEFAULT_NOISE_STDDEV

    def __post_init__(self) -> None:
        """Validate every value; fail fast on a bad configuration."""
        if not self.part_bases:
            raise ConfigError("part_bases must not be empty")
        for part, base in self.part_bases.items():
            if not isinstance(base, (int, float)) or base <= 0:
                raise ConfigError(f"part_bases[{part!r}] must be a positive number")
        if not self.region_multipliers:
            raise ConfigError("region_multipliers must not be empty")
        for region, mult in self.region_multipliers.items():
            if not isinstance(mult, (int, float)) or mult <= 0:
                raise ConfigError(
                    f"region_multipliers[{region!r}] must be a positive number"
                )
        try:
            from datetime import date as _date

            _date.fromisoformat(self.start_date)
        except ValueError as exc:
            raise ConfigError(
                f"start_date must be ISO YYYY-MM-DD, got {self.start_date!r}"
            ) from exc
        if self.days <= 0:
            raise ConfigError("days must be positive")
        if self.seed < 0:
            raise ConfigError("seed must be >= 0")
        if self.rolling_window < 1:
            raise ConfigError("rolling_window must be >= 1")
        if self.backtest_days < 1:
            raise ConfigError("backtest_days must be >= 1")
        if self.forecast_horizon < 1:
            raise ConfigError("forecast_horizon must be >= 1")
        if self.min_history_days <= self.rolling_window:
            raise ConfigError(
                "min_history_days must be larger than rolling_window"
            )
        if not 0.0 <= self.seasonal_amplitude <= 1.0:
            raise ConfigError("seasonal_amplitude must be between 0 and 1")
        if self.noise_stddev < 0:
            raise ConfigError("noise_stddev must be >= 0")

    @classmethod
    def from_dict(cls, values: dict) -> "ForecastConfig":
        """Build a config from a plain dict, rejecting unknown keys."""
        unknown = set(values) - _CONFIG_KEYS
        if unknown:
            raise ConfigError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict:
        """Serialise the config so a trained model can record it."""
        return {
            "part_bases": dict(self.part_bases),
            "region_multipliers": dict(self.region_multipliers),
            "start_date": self.start_date,
            "days": self.days,
            "seed": self.seed,
            "rolling_window": self.rolling_window,
            "backtest_days": self.backtest_days,
            "forecast_horizon": self.forecast_horizon,
            "min_history_days": self.min_history_days,
            "seasonal_amplitude": self.seasonal_amplitude,
            "noise_stddev": self.noise_stddev,
        }

    @classmethod
    def load_json(cls, path: str | Path) -> "ForecastConfig":
        """Load a config from a JSON file.

        Raises:
            ConfigError: If the file is missing, is not valid JSON, or
                contains invalid values.
        """
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"Config file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Config file is not valid JSON: {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"Config file must contain a JSON object: {path}")
        return cls.from_dict(raw)
