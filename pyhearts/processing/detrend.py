from typing import Optional, Tuple
import numpy as np
from scipy.stats import linregress
from pyhearts.plots import plot_detrended_cycle


def detrend_signal(
    xs: Optional[np.ndarray],
    signal: np.ndarray,
    sampling_rate: float,
    window_ms: float,
    *,
    cycle: Optional[int] = None,
    plot: bool = False
) -> Tuple[np.ndarray, float]:
    """
    Remove baseline offset and linear trend from an ECG (or other) signal.

    Parameters
    ----------
    xs : np.ndarray or None
        Time vector for the signal, required only if `plot=True`.
    signal : np.ndarray
        Input signal values.
    sampling_rate : float
        Sampling rate of the signal in Hz.
    window_ms : float
        Baseline window length in milliseconds.
    cycle : int, optional
        Cycle index (used for plotting titles/labels).
    plot : bool, default=False
        If True, plots the baseline and detrended signal.

    Returns
    -------
    detrended_signal : np.ndarray
        Signal after baseline and linear trend removal.
    slope : float
        Estimated linear trend slope (change per sample).

    Raises
    ------
    ValueError
        If the baseline window exceeds the signal length.
        If `plot=True` and `xs` or `cycle` is not provided.
    """
    signal = np.asarray(signal, dtype=float)

    # --- Validation ---
    baseline_samples = int(sampling_rate * (window_ms / 1000))
    if baseline_samples > len(signal):
        raise ValueError("Baseline window exceeds signal length.")
    if plot and (xs is None or cycle is None):
        raise ValueError("To plot, `xs` and `cycle` must be provided.")

    # --- Baseline correction ---
    baseline = np.median(signal[:baseline_samples])
    signal_baseline_corrected = signal - baseline

    # --- Trend estimation ---
    slope, intercept, *_ = linregress(
        [0, len(signal_baseline_corrected) - 1],
        [signal_baseline_corrected[0], signal_baseline_corrected[-1]]
    )
    trend = slope * np.arange(len(signal_baseline_corrected)) + intercept

    # --- Detrend ---
    detrended_signal = signal_baseline_corrected - trend

    # --- Plotting ---
    if plot:
        plot_detrended_cycle(xs, signal, detrended_signal, cycle)

    return detrended_signal, slope