"""Synthetic demand-history generator.

Writes a CSV of daily demand per part and region. The data is invented
for this portfolio: a base level per part, a region multiplier, an
annual seasonal wave, and Gaussian noise, all driven by a seeded local
random generator. Same seed + same config = byte-identical output, so
generation is reproducible (idempotent).

This module never touches real data, real products, or any external
system. There are no network calls here, so there is nothing to retry.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from .config import ForecastConfig
from .exceptions import DemandForecastError
from .logging_setup import configure_logging, get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class GenerationReport:
    """Outcome of one data-generation run."""

    out_path: Path
    rows_written: int
    parts: tuple[str, ...]
    regions: tuple[str, ...]
    start_date: date
    end_date: date


def generate(config: ForecastConfig, out_path: str | Path) -> GenerationReport:
    """Generate synthetic demand history and write it as CSV.

    Args:
        config: Generation settings (parts, regions, dates, seed).
        out_path: Destination CSV path; parent directories are created.

    Returns:
        A :class:`GenerationReport` describing what was written.
    """
    out_path = Path(out_path)
    start = date.fromisoformat(config.start_date)
    rng = random.Random(config.seed)  # local RNG: no global state touched
    rows: list[tuple[date, str, str, int]] = []
    for part, base in config.part_bases.items():
        for region, multiplier in config.region_multipliers.items():
            for offset in range(config.days):
                day = start + timedelta(days=offset)
                # Annual seasonal wave: peaks mid-year, troughs at year end.
                seasonal = 1.0 + config.seasonal_amplitude * math.sin(
                    2 * math.pi * offset / 365.0
                )
                noisy = base * multiplier * seasonal + rng.gauss(
                    0.0, config.noise_stddev
                )
                rows.append((day, part, region, max(0, round(noisy))))
    rows.sort(key=lambda row: (row[1], row[2], row[0]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "part", "region", "demand"])
        writer.writerows(
            (day.isoformat(), part, region, demand)
            for day, part, region, demand in rows
        )
    report = GenerationReport(
        out_path=out_path,
        rows_written=len(rows),
        parts=tuple(config.part_bases),
        regions=tuple(config.region_multipliers),
        start_date=start,
        end_date=start + timedelta(days=config.days - 1),
    )
    logger.info(
        "Generated %d rows (%d parts x %d regions x %d days) -> %s",
        report.rows_written,
        len(report.parts),
        len(report.regions),
        config.days,
        out_path,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    """CLI argument parser for data generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic daily demand history as CSV."
    )
    parser.add_argument("--out", default="data/demand.csv")
    parser.add_argument("--config", default=None, help="JSON config file")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        config = (
            ForecastConfig.load_json(args.config)
            if args.config
            else ForecastConfig()
        )
        overrides: dict = {}
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.days is not None:
            overrides["days"] = args.days
        if overrides:
            config = ForecastConfig.from_dict({**config.to_dict(), **overrides})
        generate(config, args.out)
    except DemandForecastError as exc:
        logger.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
