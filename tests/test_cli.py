"""End-to-end CLI tests: generate -> train -> score, plus failure paths."""

import json

import pytest

from demand_forecast import generate_data, score, train
from demand_forecast.config import ForecastConfig


@pytest.fixture()
def trained_model(tmp_path):
    """Run generate + train in a temp dir; return the model file path."""
    data_path = tmp_path / "data" / "demand.csv"
    model_path = tmp_path / "models" / "model.json"
    assert generate_data.main(["--out", str(data_path)]) == 0
    assert train.main(["--data", str(data_path), "--out", str(model_path)]) == 0
    return model_path


def test_full_pipeline_produces_models(trained_model):
    payload = json.loads(trained_model.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["data_quality"]["rows_rejected"] == 0
    assert payload["data_quality"]["rows_valid"] == 540 * 3 * 3
    assert len(payload["models"]) == 3 * 3  # 3 parts x 3 regions
    assert all(m["backtest_mae"] >= 0 for m in payload["models"])


def test_score_single_region(trained_model, capsys):
    assert score.main(
        ["--model", str(trained_model), "--part", "filter", "--region", "North"]
    ) == 0
    results = json.loads(capsys.readouterr().out)
    assert len(results) == 1
    assert results[0]["part"] == "filter"
    assert results[0]["region"] == "North"
    assert len(results[0]["forecast"]) == 30
    assert "driver_summary" in results[0]
    assert all(
        day["forecast_units_per_day"] >= 0 for day in results[0]["forecast"]
    )


def test_score_all_regions_when_region_omitted(trained_model, capsys):
    assert score.main(["--model", str(trained_model), "--part", "battery"]) == 0
    results = json.loads(capsys.readouterr().out)
    assert {r["region"] for r in results} == {"North", "Central", "South"}


def test_score_horizon_override(trained_model, capsys):
    assert score.main(
        ["--model", str(trained_model), "--part", "filter", "--horizon", "7"]
    ) == 0
    results = json.loads(capsys.readouterr().out)
    assert all(len(r["forecast"]) == 7 for r in results)


def test_score_unknown_part_fails(trained_model):
    assert score.main(
        ["--model", str(trained_model), "--part", "turbocharger"]
    ) == 2


def test_score_unknown_region_fails(trained_model):
    assert score.main(
        ["--model", str(trained_model), "--part", "filter", "--region", "Mars"]
    ) == 2


def test_score_missing_model_fails():
    assert score.main(["--model", "nope.json", "--part", "filter"]) == 2


def test_score_bad_horizon_fails(trained_model):
    assert score.main(
        ["--model", str(trained_model), "--part", "filter", "--horizon", "0"]
    ) == 2


def test_train_with_quarantine(tmp_path):
    data_path = tmp_path / "demand.csv"
    data_path.write_text(
        "date,part,region,demand\n"
        "2023-01-01,filter,North,10\n"
        "2023-01-02,filter,North,xx\n",
        encoding="utf-8",
    )
    # Too little history to train: expect a clean failure, not a crash.
    assert train.main(
        [
            "--data",
            str(data_path),
            "--out",
            str(tmp_path / "m.json"),
            "--quarantine-out",
            str(tmp_path / "quarantine.csv"),
        ]
    ) == 2


def test_train_quarantine_file_written(tmp_path):
    data_path = tmp_path / "demand.csv"
    config = ForecastConfig(days=200)
    assert generate_data.main(["--out", str(data_path), "--days", "200"]) == 0
    with data_path.open("a", encoding="utf-8") as handle:
        handle.write("bad-date,filter,North,10\n")
    quarantine = tmp_path / "quarantine.csv"
    assert train.main(
        [
            "--data",
            str(data_path),
            "--out",
            str(tmp_path / "model.json"),
            "--quarantine-out",
            str(quarantine),
        ]
    ) == 0
    assert quarantine.exists()
    assert "unparseable date" in quarantine.read_text(encoding="utf-8")


def test_generate_cli_seed_override(tmp_path):
    out = tmp_path / "demand.csv"
    assert generate_data.main(["--out", str(out), "--seed", "99", "--days", "5"]) == 0
    assert out.exists()
