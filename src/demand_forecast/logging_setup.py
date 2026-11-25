"""Logging setup for the demand-forecasting pipeline.

One place owns the log format so every module logs the same way.
Nothing in this package writes to stdout with ``print``; the CLIs log
progress and results, and ``score`` writes its JSON result to stdout.
"""

from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once, at CLI entry points.

    Args:
        level: A standard logging level name such as ``"INFO"`` or
            ``"DEBUG"``. Unknown names raise ``ValueError``.
    """
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Unknown log level: {level!r}")
    logging.basicConfig(level=numeric, format=LOG_FORMAT, stream=sys.stderr)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)
