"""Tests for configuration defaults, overrides, and validation."""

import json

import pytest

from demand_forecast.config import ForecastConfig
from demand_forecast.exceptions import ConfigError


def test_defaults_are_sane():
    config = ForecastConfig()
    assert config.days == 540
    assert config.seed == 7
    assert config.rolling_window == 28
    assert config.backtest_days == 90
    assert config.forecast_horizon == 30
    assert set(config.part_bases) == {"brake pad", "filter", "battery"}
    assert set(config.region_multipliers) == {"North", "Central", "South"}


def test_from_dict_overrides_only_named_keys():
    config = ForecastConfig.from_dict({"days": 100, "seed": 1})
    assert config.days == 100
    assert config.seed == 1
    assert config.backtest_days == 90  # untouched default


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="Unknown config keys"):
        ForecastConfig.from_dict({"dayz": 100})


def test_empty_parts_rejected():
    with pytest.raises(ConfigError, match="part_bases must not be empty"):
        ForecastConfig(part_bases={})


def test_non_positive_base_rejected():
    with pytest.raises(ConfigError, match="must be a positive number"):
        ForecastConfig(part_bases={"filter": 0})


def test_bad_start_date_rejected():
    with pytest.raises(ConfigError, match="start_date must be ISO"):
        ForecastConfig(start_date="01/01/2023")


def test_min_history_must_exceed_window():
    with pytest.raises(ConfigError, match="min_history_days"):
        ForecastConfig(min_history_days=10, rolling_window=28)


def test_amplitude_bounds():
    with pytest.raises(ConfigError, match="seasonal_amplitude"):
        ForecastConfig(seasonal_amplitude=1.5)
    assert ForecastConfig(seasonal_amplitude=0.0).seasonal_amplitude == 0.0


def test_load_json_round_trip(tmp_path):
    config = ForecastConfig(days=365, seed=42)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    loaded = ForecastConfig.load_json(path)
    assert loaded == config


def test_load_json_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        ForecastConfig.load_json(tmp_path / "missing.json")


def test_load_json_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        ForecastConfig.load_json(path)


def test_load_json_non_object(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ConfigError, match="must contain a JSON object"):
        ForecastConfig.load_json(path)
