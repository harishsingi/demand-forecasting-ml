"""Custom exceptions for the demand-forecasting pipeline.

Every failure that a caller or the CLI should report clearly gets its own
type, so a bad input file is never confused with a bad model file or a
bad configuration value.
"""


class DemandForecastError(Exception):
    """Base class for all errors raised by this package."""


class ConfigError(DemandForecastError):
    """A configuration value is missing, unknown, or out of range."""


class DataValidationError(DemandForecastError):
    """Input data is missing, unreadable, or fails schema validation."""


class ModelError(DemandForecastError):
    """A model is invalid, too small to train, or fails to load."""


class UnknownPartError(DemandForecastError):
    """A requested part (or part/region pair) has no trained model."""
