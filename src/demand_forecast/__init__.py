"""Demand forecasting with a transparent baseline + seasonal model.

Pure Python standard library. All data is synthetic; the package is a
clean-room portfolio recreation of a forecasting pattern, not production
code and not connected to any employer's systems or data.
"""

from .config import ForecastConfig
from .model import BaselineSeasonalModel, ForecastDay

__all__ = ["BaselineSeasonalModel", "ForecastConfig", "ForecastDay"]
__version__ = "0.2.0"
