"""
QRS removal for T-wave detection.

Replaces the QRS complex with a smooth sigmoid bridge between pre- and post-QRS
baseline levels so derivative-based T detection is less contaminated by
depolarization artifacts.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _local_mean(signal: np.ndarray, center: int, half_width: int) -> float:
    """Mean of signal in [center - half_width, center), clamped to array bounds."""
    if len(signal) == 0:
        return 0.0
    start = max(0, center - half_width)
    end = max(start + 1, min(len(signal), center))
    segment = signal[start:end]
    if segment.size == 0:
        return float(signal[min(center, len(signal) - 1)])
    return float(np.mean(segment))


def remove_qrs_sigmoid(
    signal: np.ndarray,
    r_peak_idx: int,
    sampling_rate: float,
    *,
    s_peak_idx: Optional[int] = None,
    qrs_end_idx: Optional[int] = None,
    pre_rr_frac: float = 1.0 / 3.0,
    post_qrs_ms: float = 80.0,
    baseline_window_ms: float = 10.0,
) -> np.ndarray:
    """
    Replace the QRS region with a sigmoid connecting local baselines.

    Parameters
    ----------
    signal : np.ndarray
        1D detrended ECG cycle segment.
    r_peak_idx : int
        R-peak index within ``signal``.
    sampling_rate : float
        Sampling rate in Hz.
    s_peak_idx, qrs_end_idx : int, optional
        Used to extend the replaced region past the S wave / QRS end when available.
    pre_rr_frac : float
        Fraction of cycle length to extend before R when defining QRS start.
    post_qrs_ms : float
        Minimum extension after R when S/QRS end are unavailable.
    baseline_window_ms : float
        Window for estimating pre/post replacement baseline levels.

    Returns
    -------
    np.ndarray
        Copy of ``signal`` with the QRS segment replaced (unchanged if region invalid).
    """
    if len(signal) < 3:
        return np.asarray(signal, dtype=float).copy()

    out = np.asarray(signal, dtype=float).copy()
    n = len(out)
    r_peak_idx = int(np.clip(r_peak_idx, 0, n - 1))

    rr_samples = max(n - 1, 1)
    qrs_start = max(0, r_peak_idx - int(round(pre_rr_frac * rr_samples)))

    post_samples = int(round(post_qrs_ms * sampling_rate / 1000.0))
    qrs_end_candidates = [r_peak_idx + post_samples]
    if s_peak_idx is not None:
        kdis = int(round(20.0 * sampling_rate / 1000.0))
        qrs_end_candidates.append(int(s_peak_idx) + kdis)
    if qrs_end_idx is not None:
        qrs_end_candidates.append(int(qrs_end_idx))
    qrs_end = min(n - 1, max(qrs_end_candidates))

    if qrs_end <= qrs_start + 2:
        return out

    baseline_half = max(1, int(round(baseline_window_ms * sampling_rate / 1000.0)))
    y0 = _local_mean(out, qrs_start, baseline_half)
    post_start = qrs_end + 1
    post_end = min(n, post_start + baseline_half)
    if post_end > post_start:
        y1 = float(np.mean(out[post_start:post_end]))
    else:
        y1 = float(out[-1])

    length = qrs_end - qrs_start + 1
    t = np.linspace(-6.0, 6.0, length)
    sigmoid = y0 + (y1 - y0) / (1.0 + np.exp(-t))
    out[qrs_start : qrs_end + 1] = sigmoid
    return out
