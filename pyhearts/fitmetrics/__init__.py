"""Fit quality metrics for Gaussian model evaluation."""

from .rmse import calc_rmse
from .rsquared import calc_r_squared

__all__ = [
    "calc_r_squared",
    "calc_rmse",
]
