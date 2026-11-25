"""Tests for CSV loading, schema validation, and quarantine."""

from pathlib import Path

import pytest

from demand_forecast.data import (
    group_by_part_region,
    load_demand_csv,
    write_quarantine_csv,
)
from demand_forecast.exceptions import DataValidationError

HEADER = "date,part,region,demand\n"


def write_csv(tmp_path: Path, body: str, name: str = "demand.csv") -> Path:
    path = tmp_path / name
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def test_valid_rows_load_and_sort(tmp_path):
    path = write_csv(
        tmp_path,
        "2023-01-02,filter,North,5\n2023-01-01,filter,North,4\n",
    )
    result = load_demand_csv(path)
    assert result.rows_read == 2
    assert result.rows_valid == 2
    assert result.rows_rejected == 0
    assert [r.demand for r in result.records] == [4.0, 5.0]


def test_missing_file_raises(tmp_path):
    with pytest.raises(DataValidationError, match="not found"):
        load_demand_csv(tmp_path / "nope.csv")


def test_missing_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("date,part,demand\n2023-01-01,filter,5\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="missing required columns"):
        load_demand_csv(path)


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DataValidationError, match="empty"):
        load_demand_csv(path)


def test_bad_rows_quarantined_with_reasons(tmp_path):
    path = write_csv(
        tmp_path,
        "2023-01-01,filter,North,10\n"
        "not-a-date,filter,North,10\n"
        "2023-01-02,filter,North,abc\n"
        "2023-01-03,filter,North,-5\n"
        "2023-01-04,,North,10\n"
        "2023-01-05,filter,North,\n"
        "2023-01-01,filter,North,10\n",
    )
    result = load_demand_csv(path)
    assert result.rows_read == 7
    assert result.rows_valid == 1
    assert result.rows_rejected == 6
    assert result.duplicate_count == 1
    reasons = [r.reason for r in result.rejected]
    assert any("unparseable date" in r for r in reasons)
    assert any("not numeric" in r for r in reasons)
    assert any("negative demand" in r for r in reasons)
    assert any("missing or empty value: part" in r for r in reasons)
    assert any("missing or empty value: demand" in r for r in reasons)
    assert any(r.startswith("duplicate") for r in reasons)


def test_quarantine_csv_round_trip(tmp_path):
    path = write_csv(
        tmp_path,
        "2023-01-01,filter,North,10\n2023-01-02,filter,North,xx\n",
    )
    result = load_demand_csv(path)
    out = tmp_path / "quarantine.csv"
    write_quarantine_csv(result.rejected, out)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "line_number,reason,date,part,region,demand"
    assert len(lines) == 2
    assert "not numeric" in lines[1]
    assert "2023-01-02" in lines[1]


def test_group_by_part_region_requires_records():
    with pytest.raises(DataValidationError, match="No valid demand records"):
        group_by_part_region([])


def test_group_by_part_region_groups(tmp_path):
    path = write_csv(
        tmp_path,
        "2023-01-01,filter,North,4\n2023-01-01,filter,South,9\n",
    )
    result = load_demand_csv(path)
    grouped = group_by_part_region(result.records)
    assert set(grouped) == {("filter", "North"), ("filter", "South")}
    assert grouped[("filter", "South")][0][1] == 9.0
