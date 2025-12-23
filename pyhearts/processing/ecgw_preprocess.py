import numpy as np
from scipy.signal import butter, filtfilt
from typing import Tuple


def _butter_band(signal: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 2) -> np.ndarray:
    """
    Simple Butterworth band filter (low/high in Hz) with filtfilt.

    Used to approximate a QRS-focused passband (high-pass around 1 Hz,
    low-pass around 60 Hz) in a stable, concise way.
    """
    if signal.size == 0:
        return signal.astype(float)

    nyq = fs / 2.0
    low = max(0.01, low_hz / nyq)
    high = min(0.99, high_hz / nyq)
    if high <= low + 1e-3:
        high = min(0.99, low + 0.05)

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal.astype(float))


def _moving_window_integral(x: np.ndarray, fs: float, window_ms: float) -> np.ndarray:
    """
    Moving-window integration over x using a simple rectangular window.

    - window_ms: integration window in milliseconds
    """
    if x.size == 0:
        return x.astype(float)

    win_samples = int(round(window_ms * fs / 1000.0))
    win_samples = max(1, win_samples)
    window = np.ones(win_samples, dtype=float) / win_samples
    return np.convolve(x, window, mode="same")


def qrs_energy_preprocess(
    ecg: np.ndarray,
    fs: float,
    *,
    highpass_hz: float = 1.0,
    lowpass_hz: float = 60.0,
    mwi_window_ms: float = 150.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocessing chain for QRS/edge work based on band-pass and energy:

    1. Band-pass filter (≈1–60 Hz) using a 2nd-order Butterworth filter.
    2. First derivative of the bandpassed signal.
    3. Squaring of the derivative (energy).
    4. Moving-window integration (rectangular window in ms).

    Returns
    -------
    ecg_band : np.ndarray
        Band-passed ECG signal.
    deriv : np.ndarray
        First derivative of ecg_band.
    mwi : np.ndarray
        Moving-window integrated energy signal.
    """
    ecg = np.asarray(ecg, dtype=float).ravel()
    if ecg.size == 0 or fs <= 0:
        return ecg, ecg, ecg

    # 1) Band-pass
    ecg_band = _butter_band(ecg, fs, low_hz=highpass_hz, high_hz=lowpass_hz, order=2)

    # 2) Derivative
    deriv = np.gradient(ecg_band)

    # 3) Squared derivative (energy)
    energy = deriv ** 2

    # 4) Moving-window integration (~150 ms by default)
    mwi = _moving_window_integral(energy, fs, window_ms=mwi_window_ms)

    return ecg_band, deriv, mwi


