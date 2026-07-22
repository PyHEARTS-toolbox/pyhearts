"""
Phase B: signal conditioning for P/T delineation.

- Two-stage median baseline removal (windows fs and 2×fs).
- Savitzky–Golay smoothing on search segments (shape-preserving; off for record-T search).
- Optional delineation bandpass (optional; record-level T search uses median baseline only).
  R detection bandpass is configured separately via ``rpeak_highpass_hz`` / ``rpeak_lowpass_hz``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, savgol_filter

from pyhearts.config import ProcessCycleConfig


def _odd_kernel(n_samples: int) -> int:
    k = max(3, int(n_samples))
    if k % 2 == 0:
        k += 1
    return k


def median_baseline_removal(
    ecg: np.ndarray,
    sampling_rate: float,
    window1_s: float = 1.0,
    window2_s: float = 2.0,
) -> np.ndarray:
    """
    Subtract cascaded median-filter baseline (record-level stage).

    First median window ≈ fs, second ≈ 2×fs (sample counts).
    """
    x = np.asarray(ecg, dtype=float)
    if x.size < 5:
        return x.copy()
    k1 = _odd_kernel(window1_s * sampling_rate)
    k2 = _odd_kernel(window2_s * sampling_rate)
    baseline = median_filter(x, size=k1, mode="nearest")
    baseline = median_filter(baseline, size=k2, mode="nearest")
    return x - baseline


def light_bandpass_delineation(
    ecg: np.ndarray,
    sampling_rate: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> np.ndarray:
    """Butterworth bandpass for P/T search (Rahul-style; applied after baseline removal)."""
    x = np.asarray(ecg, dtype=float)
    if x.size < max(8, order * 3):
        return x.copy()
    nyq = sampling_rate / 2.0
    if high_hz >= nyq:
        high_hz = nyq * 0.95
    b, a = butter(int(order), [low_hz / nyq, high_hz / nyq], btype="band")
    try:
        return filtfilt(b, a, x)
    except ValueError:
        return x.copy()


def prepare_record_delineation_signal(
    ecg: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> np.ndarray:
    """Full-record signal for P/T search (median baseline; optional bandpass)."""
    if cfg.delineation_baseline_method == "median_record":
        w1, w2 = cfg.delineation_median_baseline_windows_s
        x = median_baseline_removal(ecg, sampling_rate, w1, w2)
    else:
        x = np.asarray(ecg, dtype=float).copy()
    if cfg.delineation_bandpass:
        x = light_bandpass_delineation(
            x,
            sampling_rate,
            cfg.delineation_bandpass_low_hz,
            cfg.delineation_bandpass_high_hz,
            cfg.delineation_bandpass_order,
        )
    return x


def extract_cycle_delineation_signal(
    one_cycle,
    sig_detrended: np.ndarray,
    full_delineation_ecg: Optional[np.ndarray],
    cfg: ProcessCycleConfig,
    sampling_rate: float,
) -> np.ndarray:
    """
    Per-cycle delineation signal: slice from full record or median-baseline the epoch.
    """
    if full_delineation_ecg is not None and "index" in one_cycle.columns:
        xs = one_cycle["index"].values.astype(int)
        lo, hi = int(xs[0]), int(xs[-1])
        if 0 <= lo < full_delineation_ecg.size and hi < full_delineation_ecg.size:
            seg = full_delineation_ecg[lo : hi + 1]
            if seg.size == len(sig_detrended):
                return seg.astype(float, copy=False)
    if cfg.delineation_baseline_method == "median_record":
        w1, w2 = cfg.delineation_median_baseline_windows_s
        x = median_baseline_removal(sig_detrended, sampling_rate, w1, w2)
    else:
        x = np.asarray(sig_detrended, dtype=float)
    if cfg.delineation_bandpass:
        x = light_bandpass_delineation(
            x,
            sampling_rate,
            cfg.delineation_bandpass_low_hz,
            cfg.delineation_bandpass_high_hz,
            cfg.delineation_bandpass_order,
        )
    return x


def savgol_search_segment(
    segment: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> np.ndarray:
    """Optional Savitzky–Golay smoothing on a P/T search band."""
    seg = np.asarray(segment, dtype=float)
    if not cfg.p_t_search_savgol or seg.size < 5:
        return seg
    win_ms = cfg.p_t_savgol_window_ms
    poly = int(cfg.p_t_savgol_polyorder)
    win = _odd_kernel(win_ms * sampling_rate / 1000.0)
    if win >= seg.size:
        win = seg.size - 1 if (seg.size - 1) % 2 == 1 else seg.size - 2
    if win < poly + 2:
        return seg
    try:
        return savgol_filter(seg, window_length=win, polyorder=poly, mode="interp")
    except ValueError:
        return seg


def smooth_search_window(
    signal: np.ndarray,
    start: int,
    end: int,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> Tuple[np.ndarray, int, int]:
    """Return (smoothed_segment, start, end) for indices [start, end)."""
    n = len(signal)
    start = int(np.clip(start, 0, max(0, n - 1)))
    end = int(np.clip(end, start + 1, n))
    if end <= start:
        end = min(n, start + 3)
    segment = signal[start:end]
    return savgol_search_segment(segment, sampling_rate, cfg), start, end
