"""Training pipeline: validate data, fit one model per part/region, save.

Steps, in order:

1. Load and validate the demand CSV (:mod:`demand_forecast.data`).
   Bad rows are quarantined, never silently dropped.
2. Fit a :class:`~demand_forecast.model.BaselineSeasonalModel` per
   (part, region) series, each with a walk-forward backtest MAE.
3. Write the models to JSON atomically (temp file + rename), so a
   failed run never leaves a half-written model file behind.
   Re-running training on the same data and config reproduces the same
   model file apart from the ``trained_at`` timestamp, so training is
   idempotent in content.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import ForecastConfig
from .data import LoadResult, load_demand_csv, write_quarantine_csv
from .exceptions import DemandForecastError, ModelError
from .features import series_by_part_region
from .logging_setup import configure_logging, get_logger
from .model import MODEL_FORMAT_VERSION, BaselineSeasonalModel

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingReport:
    """Outcome of one training run."""

    model_path: Path
    models_trained: int
    parts: tuple[str, ...]
    regions: tuple[str, ...]
    rows_read: int
    rows_valid: int
    rows_rejected: int
    backtest_mae_by_key: dict[tuple[str, str], float]


def run_training(
    data_path: str | Path,
    model_path: str | Path,
    config: ForecastConfig,
    quarantine_path: str | Path | None = None,
) -> TrainingReport:
    """Run the full training pipeline and write the model file.

    Raises:
        DemandForecastError: On invalid input data, a series too short
            to train, or an unwritable output path.
    """
    model_path = Path(model_path)
    loaded: LoadResult = load_demand_csv(data_path)
    if loaded.rows_valid == 0:
        raise ModelError(
            f"No valid demand records in {data_path} "
            f"({loaded.rows_rejected} rows rejected)"
        )
    if quarantine_path is not None and loaded.rejected:
        write_quarantine_csv(loaded.rejected, quarantine_path)
    series = series_by_part_region(loaded.records)
    models: list[BaselineSeasonalModel] = []
    for (part, region), (dates, values) in sorted(series.items()):
        models.append(
            BaselineSeasonalModel.fit(part, region, dates, values, config)
        )
    payload = {
        "version": MODEL_FORMAT_VERSION,
        # Honest timestamp of this run; never backdated.
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "data_quality": {
            "rows_read": loaded.rows_read,
            "rows_valid": loaded.rows_valid,
            "rows_rejected": loaded.rows_rejected,
            "duplicates": loaded.duplicate_count,
            "input_file": str(data_path),
        },
        "models": [model.to_dict() for model in models],
    }
    _atomic_write_json(model_path, payload)
    logger.info(
        "Trained %d models (%d parts, %d regions) -> %s",
        len(models),
        len({m.part for m in models}),
        len({m.region for m in models}),
        model_path,
    )
    return TrainingReport(
        model_path=model_path,
        models_trained=len(models),
        parts=tuple(sorted({m.part for m in models})),
        regions=tuple(sorted({m.region for m in models})),
        rows_read=loaded.rows_read,
        rows_valid=loaded.rows_valid,
        rows_rejected=loaded.rows_rejected,
        backtest_mae_by_key={
            (m.part, m.region): m.backtest_mae for m in models
        },
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON via a temp file in the same directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    """CLI argument parser for training."""
    parser = argparse.ArgumentParser(
        description="Train baseline + seasonal demand models."
    )
    parser.add_argument("--data", default="data/demand.csv")
    parser.add_argument("--out", default="models/model.json")
    parser.add_argument("--config", default=None, help="JSON config file")
    parser.add_argument(
        "--quarantine-out",
        default="data/quarantine.csv",
        help="Where to write quarantined rows (CSV). Written only when "
        "any rows are rejected.",
    )
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
        quarantine = args.quarantine_out
        report = run_training(args.data, args.out, config, quarantine)
        if report.rows_rejected:
            logger.warning(
                "%d of %d input rows were quarantined; "
                "see the quarantine file or the warnings above",
                report.rows_rejected,
                report.rows_read,
            )
    except DemandForecastError as exc:
        logger.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
