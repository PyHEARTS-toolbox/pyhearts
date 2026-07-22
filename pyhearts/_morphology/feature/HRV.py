from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def rr_intervals_ms_from_r_peaks(
    r_peaks: Sequence[float] | np.ndarray,
    sampling_rate: float,
    rr_bounds_ms: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Build physiology-gated RR intervals (ms) from detected R-peak sample indices.

    This path does not depend on epoch retention or morphology fitting: it uses
    successive R peaks from ``r_peak_detection``.

    Parameters
    ----------
    r_peaks :
        Absolute sample indices of R peaks.
    sampling_rate :
        Sampling rate in Hz (must be > 0).
    rr_bounds_ms :
        Optional inclusive ``(lo, hi)`` gate in milliseconds. Intervals outside
        the gate are dropped.

    Returns
    -------
    np.ndarray
        1-D array of RR intervals in milliseconds (may be empty).
    """
    if sampling_rate <= 0:
        return np.asarray([], dtype=float)

    peaks = np.asarray(r_peaks, dtype=float)
    peaks = peaks[np.isfinite(peaks)]
    if peaks.size < 2:
        return np.asarray([], dtype=float)

    peaks = np.unique(np.round(peaks).astype(int))
    if peaks.size < 2:
        return np.asarray([], dtype=float)

    rr_ms = np.diff(peaks.astype(float)) * (1000.0 / float(sampling_rate))
    if rr_bounds_ms is not None:
        lo, hi = float(rr_bounds_ms[0]), float(rr_bounds_ms[1])
        if not (lo < hi):
            raise ValueError("rr_bounds_ms requires lo < hi")
        rr_ms = rr_ms[(rr_ms >= lo) & (rr_ms <= hi)]
    return np.asarray(rr_ms, dtype=float)


def calc_hrv_metrics(rr_intervals: np.ndarray):
    """
    Calculate basic heart rate variability (HRV) metrics from R-R intervals.

    Parameters
    ----------
    rr_intervals : np.ndarray
        Array of R-R intervals in milliseconds (ms). May contain NaN values.

    Returns
    -------
    average_heart_rate : int
        Mean heart rate in beats per minute (bpm), rounded to nearest int.
    sdnn : int
        Standard deviation of NN intervals (SDNN), rounded to nearest int.
    rmssd : int
        Root mean square of successive differences (RMSSD), rounded to nearest int.
    nn50 : int
        Number of successive RR interval differences greater than 50 ms (NN50).
    """
    clean_rr_intervals = rr_intervals[~np.isnan(rr_intervals)]

    # Calculate instantaneous heart rate in bpm
    heart_rate = 60 / (clean_rr_intervals / 1000)
    average_heart_rate = np.nanmean(heart_rate) if len(heart_rate) > 0 else np.nan

    if len(clean_rr_intervals) > 1:
        sdnn = np.std(clean_rr_intervals, ddof=1)
        rmssd = np.sqrt(np.mean(np.diff(clean_rr_intervals) ** 2))
        nn50 = int(np.sum(np.abs(np.diff(clean_rr_intervals)) > 50))
    else:
        sdnn, rmssd, nn50 = np.nan, np.nan, np.nan

    # Round and convert to int, handling NaNs safely
    def safe_int(val):
        return int(round(val)) if not np.isnan(val) else None

    return (
        safe_int(average_heart_rate),
        safe_int(sdnn),
        safe_int(rmssd),
        nn50 if not np.isnan(nn50) else None,
    )
