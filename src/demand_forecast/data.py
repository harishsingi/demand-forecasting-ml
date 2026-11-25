"""Demand-history loading and schema validation.

This is the data-quality gate for the pipeline. Every row of the input
CSV is checked before training:

* required columns ``date``, ``part``, ``region``, ``demand`` present;
* ``date`` parses as ISO ``YYYY-MM-DD``;
* ``part`` / ``region`` non-empty;
* ``demand`` numeric, finite, and >= 0;
* no duplicate ``(date, part, region)`` rows (first occurrence wins).

Bad rows are never silently dropped: they are collected as
:class:`RejectedRecord` entries with a reason, counted, logged, and
optionally written to a quarantine CSV so they can be inspected and
fixed upstream. Row counts in / out / rejected are logged on every load.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .exceptions import DataValidationError

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("date", "part", "region", "demand")
QUARANTINE_COLUMNS = ("line_number", "reason", "date", "part", "region", "demand")


@dataclass(frozen=True)
class DemandRecord:
    """One validated daily demand observation."""

    date: date
    part: str
    region: str
    demand: float


@dataclass(frozen=True)
class RejectedRecord:
    """One input row that failed validation, kept for quarantine."""

    line_number: int
    reason: str
    raw: dict[str, str] = field(default_factory=dict)


@dataclass
class LoadResult:
    """Outcome of loading a demand CSV."""

    records: list[DemandRecord]
    rejected: list[RejectedRecord]
    rows_read: int

    @property
    def rows_valid(self) -> int:
        """Number of rows that passed validation."""
        return len(self.records)

    @property
    def rows_rejected(self) -> int:
        """Number of rows quarantined."""
        return len(self.rejected)

    @property
    def duplicate_count(self) -> int:
        """Rejected rows that were duplicates of an earlier row."""
        return sum(1 for r in self.rejected if r.reason.startswith("duplicate"))


def _parse_row(
    raw: dict[str, str | None], line_number: int, seen: set[tuple[date, str, str]]
) -> DemandRecord | RejectedRecord:
    """Validate one CSV row. Returns a record or a rejection, never raises."""
    cleaned = {k: (v.strip() if isinstance(v, str) else "") for k, v in raw.items()}
    for column in REQUIRED_COLUMNS:
        if not cleaned.get(column):
            return RejectedRecord(
                line_number, f"missing or empty value: {column}", cleaned
            )
    try:
        parsed_date = date.fromisoformat(cleaned["date"])
    except ValueError:
        return RejectedRecord(
            line_number, f"unparseable date: {cleaned['date']!r}", cleaned
        )
    try:
        demand = float(cleaned["demand"])
    except ValueError:
        return RejectedRecord(
            line_number, f"demand is not numeric: {cleaned['demand']!r}", cleaned
        )
    if not math.isfinite(demand):
        return RejectedRecord(line_number, "demand is not finite", cleaned)
    if demand < 0:
        return RejectedRecord(
            line_number, f"negative demand: {demand}", cleaned
        )
    key = (parsed_date, cleaned["part"], cleaned["region"])
    if key in seen:
        return RejectedRecord(
            line_number,
            f"duplicate of an earlier row for {key[1]!r} / {key[2]!r} on {key[0]}",
            cleaned,
        )
    seen.add(key)
    return DemandRecord(
        date=parsed_date,
        part=cleaned["part"],
        region=cleaned["region"],
        demand=demand,
    )


def load_demand_csv(path: str | Path) -> LoadResult:
    """Load and validate a demand CSV, quarantining bad rows.

    Args:
        path: Path to a CSV with columns ``date,part,region,demand``.

    Returns:
        A :class:`LoadResult` with validated records (sorted by part,
        region, date), rejected rows, and counts.

    Raises:
        DataValidationError: If the file is missing or lacks the
            required columns.
    """
    path = Path(path)
    if not path.is_file():
        raise DataValidationError(f"Input file not found: {path}")
    records: list[DemandRecord] = []
    rejected: list[RejectedRecord] = []
    seen: set[tuple[date, str, str]] = set()
    rows_read = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DataValidationError(f"Input file is empty: {path}")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise DataValidationError(
                f"Input file {path} is missing required columns: {missing}"
            )
        for line_number, raw in enumerate(reader, start=2):
            rows_read += 1
            result = _parse_row(raw, line_number, seen)
            if isinstance(result, RejectedRecord):
                rejected.append(result)
            else:
                records.append(result)
    records.sort(key=lambda r: (r.part, r.region, r.date))
    outcome = LoadResult(records=records, rejected=rejected, rows_read=rows_read)
    logger.info(
        "Loaded %s: rows_read=%d rows_valid=%d rows_rejected=%d duplicates=%d",
        path,
        outcome.rows_read,
        outcome.rows_valid,
        outcome.rows_rejected,
        outcome.duplicate_count,
    )
    for bad in rejected:
        logger.warning("Quarantined line %d: %s", bad.line_number, bad.reason)
    return outcome


def write_quarantine_csv(rejected: list[RejectedRecord], path: str | Path) -> Path:
    """Write quarantined rows to a CSV for inspection. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(QUARANTINE_COLUMNS))
        writer.writeheader()
        for bad in rejected:
            writer.writerow(
                {
                    "line_number": bad.line_number,
                    "reason": bad.reason,
                    "date": bad.raw.get("date", ""),
                    "part": bad.raw.get("part", ""),
                    "region": bad.raw.get("region", ""),
                    "demand": bad.raw.get("demand", ""),
                }
            )
    logger.info("Wrote %d quarantined rows to %s", len(rejected), path)
    return path


def group_by_part_region(
    records: list[DemandRecord],
) -> dict[tuple[str, str], list[tuple[date, float]]]:
    """Group validated records into date-sorted series per (part, region).

    Raises:
        DataValidationError: If ``records`` is empty.
    """
    if not records:
        raise DataValidationError("No valid demand records to group")
    grouped: dict[tuple[str, str], list[tuple[date, float]]] = {}
    for record in records:
        grouped.setdefault((record.part, record.region), []).append(
            (record.date, record.demand)
        )
    for series in grouped.values():
        series.sort(key=lambda item: item[0])
    return grouped
