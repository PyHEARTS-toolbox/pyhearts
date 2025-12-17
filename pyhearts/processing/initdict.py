from typing import List, Optional
import numpy as np


def initialize_output_dict(
    cycle_inds,
    components: List[str],
    peak_features: List[str],
    intervals: List[str],
    pairwise_differences: Optional[List[str]] = None,
):
    """
    Initialize an output dictionary for ECG metrics.

    Parameters
    ----------
    cycle_inds : array-like
        Cycle numbers for each heartbeat.

    components : list of str
        ECG component labels (e.g., 'P', 'Q', 'R', 'S', 'T').

    peak_featuers : list of str
        Per-component metrics to store (e.g., 'duration_ms', 'sharpness').

    intervals : list of str
        Interval metrics between components (e.g., 'RR_interval_ms').

    pairwise_differences : list of str, optional
        Names of per-cycle metrics derived from differences between components
        (e.g., 'R_minus_S_voltage_diff_signed').
    """
    n = len(cycle_inds)

    output_dict = {
        "cycle_trend": [np.nan] * n,
        "r_squared":   [np.nan] * n,
        "rmse":        [np.nan] * n,
    }

    # Per-component metrics
    for comp in components:
        for metric in peak_features:
            output_dict[f"{comp}_{metric}"] = [np.nan] * n

    # Interval metrics
    for interval in intervals:
        output_dict[interval] = [np.nan] * n

    # Pairwise differences (per-cycle)
    if pairwise_differences:
        for key in pairwise_differences:
            output_dict[key] = [np.nan] * n

    return output_dict
