"""Scoring CLI: forecast the next N days for a part, with explanation.

Reads a model file written by :mod:`demand_forecast.train`, forecasts
``--horizon`` days starting the day after the model's history ends, and
prints one JSON object per (part, region) to stdout, including the
per-day forecast and a plain-English driver summary.

Scoring is read-only and deterministic: the same model file and horizon
always produce the same output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ForecastConfig
from .exceptions import DemandForecastError, ModelError, UnknownPartError
from .logging_setup import configure_logging, get_logger
from .model import MODEL_FORMAT_VERSION, BaselineSeasonalModel, ForecastDay

logger = get_logger(__name__)


def load_models(path: str | Path) -> list[BaselineSeasonalModel]:
    """Load models from a training output file.

    Raises:
        ModelError: If the file is missing, is not valid JSON, has an
            unsupported format version, or contains no models.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelError(f"Model file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelError(f"Model file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("version") != MODEL_FORMAT_VERSION:
        raise ModelError(
            f"Model file {path} has an unsupported format "
            f"(expected version {MODEL_FORMAT_VERSION})"
        )
    entries = raw.get("models")
    if not isinstance(entries, list) or not entries:
        raise ModelError(f"Model file {path} contains no models")
    return [BaselineSeasonalModel.from_dict(entry) for entry in entries]


def select_models(
    models: list[BaselineSeasonalModel], part: str, region: str | None
) -> list[BaselineSeasonalModel]:
    """Pick the model(s) for a part, optionally narrowed to one region.

    Raises:
        UnknownPartError: If the part, or the part/region pair, has no
            trained model. The error lists what is available.
    """
    available_parts = sorted({m.part for m in models})
    matches = [m for m in models if m.part == part]
    if not matches:
        raise UnknownPartError(
            f"No trained model for part {part!r}. "
            f"Available parts: {available_parts}"
        )
    if region is not None:
        matches = [m for m in matches if m.region == region]
        if not matches:
            available_regions = sorted(
                {m.region for m in models if m.part == part}
            )
            raise UnknownPartError(
                f"No trained model for part {part!r} in region "
                f"{region!r}. Available regions for {part!r}: "
                f"{available_regions}"
            )
    return sorted(matches, key=lambda m: m.region)


def forecast_payload(
    model: BaselineSeasonalModel, horizon: int
) -> dict:
    """Build the JSON-serialisable forecast for one model."""
    days: list[ForecastDay] = model.forecast(horizon)
    return {
        "part": model.part,
        "region": model.region,
        "forecast_start": days[0].date.isoformat(),
        "forecast_end": days[-1].date.isoformat(),
        "horizon_days": horizon,
        "backtest_mae": model.backtest_mae,
        "forecast": [
            {
                "date": day.date.isoformat(),
                "forecast_units_per_day": day.forecast,
                "seasonal_factor": day.seasonal_factor,
            }
            for day in days
        ],
        "driver_summary": model.driver_summary(horizon),
    }


def build_parser() -> argparse.ArgumentParser:
    """CLI argument parser for scoring."""
    parser = argparse.ArgumentParser(
        description="Forecast the next N days of demand for a part."
    )
    parser.add_argument("--model", default="models/model.json")
    parser.add_argument("--part", required=True)
    parser.add_argument(
        "--region",
        default=None,
        help="Forecast one region only. Omit to forecast all regions "
        "for the part.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=ForecastConfig().forecast_horizon,
        help="Days ahead to forecast (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional file to write the JSON result to (in addition "
        "to stdout).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        if args.horizon < 1:
            raise ModelError(f"--horizon must be >= 1, got {args.horizon}")
        models = load_models(args.model)
        selected = select_models(models, args.part, args.region)
        results = [forecast_payload(model, args.horizon) for model in selected]
        text = json.dumps(results, indent=2)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(text + "\n", encoding="utf-8")
            logger.info("Wrote forecast to %s", args.output)
        sys.stdout.write(text + "\n")
        for model in selected:
            logger.info("%s", model.driver_summary(args.horizon))
    except DemandForecastError as exc:
        logger.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
