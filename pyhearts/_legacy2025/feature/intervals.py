import numpy as np
from typing import Dict, Any, Optional


def interpolate_peaks(peak_series, cycle_idx, window_size=5):
    """
    Interpolate missing or invalid peak values using surrounding cycles.

    Parameters:
    - peak_series (list or np.array): List of peak values (e.g. Q_le_idx_ind) for all cycles.
    - cycle_idx (int): Index of the current cycle.
    - window_size (int): Number of surrounding cycles to use for interpolation.

    Returns:
    - interpolated_value (float or np.nan): Interpolated peak index or np.nan if none found.
    """
    valid_values = []

    for ri_idx in range(-window_size, window_size + 1):
        idx = cycle_idx + ri_idx
        if 0 <= idx < len(peak_series) and not np.isnan(peak_series[idx]):
            valid_values.append(peak_series[idx])

    return np.mean(valid_values) if valid_values else np.nan



def calc_intervals(
    all_peak_series: Dict[str, Any],
    cycle_idx: int,
    sampling_rate: int,
    window_size: int = 3
) -> Dict[str, float]:
    """
    Calculate intervals (in milliseconds) for a given cycle, with fallback interpolation.

    Parameters
    ----------
    all_peak_series : dict
        Dict of component edge indices across all cycles (e.g., 'P_le_idx': [...], 'Q_le_idx': [...]).
    cycle_idx : int
        Index of the current cycle to analyze.
    sampling_rate : int
        Sampling rate in Hz.
    window_size : int
        Number of cycles before and after to use for interpolation if needed.

    Returns
    -------
    Dict[str, float]
        Mapping of interval name (e.g., 'PR_interval_ms') to duration in milliseconds.
        If invalid or uninterpolatable, the value is np.nan.
    """

    intervals_to_calculate = {
        'PR_interval_ms': ('P_le_idx', 'Q_le_idx'),
        'PR_segment_ms': ('P_ri_idx', 'Q_le_idx'),
        'QRS_interval_ms': ('Q_le_idx', 'S_ri_idx'),
        'ST_segment_ms': ('S_ri_idx', 'T_le_idx'),
        'ST_interval_ms': ('S_ri_idx', 'T_ri_idx'),
        'QT_interval_ms': ('Q_le_idx', 'T_ri_idx')
    }

    PHYSIOLOGICAL_LIMITS_MS = {
        'PR_interval': (50, 500),
        'PR_segment': (10, 400),
        'QRS_interval': (20, 300),
        'ST_segment': (20, 500),
        'ST_interval': (5, 700),
        'QT_interval': (200, 750),
    }

    result = {}

    for interval_ms, (on_key, off_key) in intervals_to_calculate.items():
        on_series = all_peak_series.get(on_key, [])
        off_series = all_peak_series.get(off_key, [])

        on = on_series[cycle_idx] if cycle_idx < len(on_series) else np.nan
        off = off_series[cycle_idx] if cycle_idx < len(off_series) else np.nan

        if np.isnan(on):
            on = interpolate_peaks(on_series, cycle_idx, window_size)
        if np.isnan(off):
            off = interpolate_peaks(off_series, cycle_idx, window_size)

        if np.isnan(on) or np.isnan(off):
            result[interval_ms] = np.nan
            continue

        sample_diff = off - on
        if sample_diff < 0:
            result[interval_ms] = np.nan
            continue

        duration_ms = (sample_diff / sampling_rate) * 1000
        interval_name = interval_ms.replace('_ms', '')
        min_ms, max_ms = PHYSIOLOGICAL_LIMITS_MS[interval_name]

        result[interval_ms] = duration_ms if min_ms <= duration_ms <= max_ms else np.nan

    return result


def interval_ms(
    curr_idx: Optional[int],
    prev_idx: Optional[int],
    lo_ms: float,
    hi_ms: float,
    ms_per_sample: float,
) -> float:
    """
    Convert a sample index difference to milliseconds and gate it to a
    physiologic interval range.

    Parameters
    ----------
    curr_idx : Optional[int]
        Current event index in absolute samples. If ``None`` the result is NaN.
    prev_idx : Optional[int]
        Previous event index in absolute samples. If ``None`` the result is NaN.
    lo_ms : float
        Lower bound for a valid interval in milliseconds, inclusive.
    hi_ms : float
        Upper bound for a valid interval in milliseconds, inclusive.
    ms_per_sample : float
        Milliseconds per sample, i.e., ``1000.0 / sampling_rate``.

    Returns
    -------
    float
        Interval in milliseconds if valid and within bounds. ``np.nan`` otherwise.

    Notes
    -----
    - Non-positive or non-finite differences return NaN.
    - Bounds are inclusive to keep on-threshold values.
    """
    if curr_idx is None or prev_idx is None:
        return float("nan")
    diff = curr_idx - prev_idx
    if diff <= 0:
        return float("nan")
    val_ms = diff * ms_per_sample
    return float(val_ms) if (lo_ms <= val_ms <= hi_ms) else float("nan")
