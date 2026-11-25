"""Tests for synthetic data generation: determinism and shape."""

from demand_forecast.config import ForecastConfig
from demand_forecast.generate_data import generate


def test_row_count_matches_config(tmp_path):
    config = ForecastConfig(days=10)
    report = generate(config, tmp_path / "demand.csv")
    expected = 10 * len(config.part_bases) * len(config.region_multipliers)
    assert report.rows_written == expected
    lines = (tmp_path / "demand.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "date,part,region,demand"
    assert len(lines) == expected + 1


def test_same_seed_same_output(tmp_path):
    config = ForecastConfig(days=30, seed=7)
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    generate(config, first)
    generate(config, second)
    assert first.read_bytes() == second.read_bytes()


def test_different_seed_different_output(tmp_path):
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    generate(ForecastConfig(days=30, seed=7), first)
    generate(ForecastConfig(days=30, seed=8), second)
    assert first.read_bytes() != second.read_bytes()


def test_demand_never_negative_and_dates_cover_range(tmp_path):
    config = ForecastConfig(days=400, seed=3)
    report = generate(config, tmp_path / "demand.csv")
    assert report.start_date.isoformat() == "2023-01-01"
    assert report.end_date.isoformat() == "2024-02-04"
    rows = (tmp_path / "demand.csv").read_text(encoding="utf-8").splitlines()[1:]
    for row in rows:
        demand = int(row.rsplit(",", 1)[1])
        assert demand >= 0


def test_report_describes_output(tmp_path):
    config = ForecastConfig(days=5)
    report = generate(config, tmp_path / "sub" / "demand.csv")
    assert report.parts == tuple(config.part_bases)
    assert report.regions == tuple(config.region_multipliers)
    assert report.out_path.exists()
