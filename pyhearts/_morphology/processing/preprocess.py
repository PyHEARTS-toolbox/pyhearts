from typing import Optional, Union

import numpy as np
from scipy import signal


def _as_1d_ecg(ecg_signal: Union[np.ndarray, list, tuple]) -> np.ndarray:
    """Coerce input to a float 1-D vector; accept column/row vectors only."""
    arr = np.asarray(ecg_signal, dtype=float)
    if arr.ndim == 0:
        raise ValueError("ECG signal must be 1-D; got a scalar.")
    if arr.ndim == 1:
        return arr.copy()
    if arr.ndim == 2 and 1 in arr.shape:
        return np.ravel(arr)
    raise ValueError(
        f"ECG signal must be 1-D (or a column/row vector); got shape {arr.shape}."
    )


def _validate_filter_args(
    *,
    sampling_rate: float,
    highpass_cutoff: Optional[float],
    lowpass_cutoff: Optional[float],
    filter_order: Optional[int],
    notch_frequency: Optional[float],
    quality_factor: Optional[float],
) -> None:
    if sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be positive; got {sampling_rate}.")

    wants_band = highpass_cutoff is not None or lowpass_cutoff is not None
    if wants_band and filter_order is None:
        raise ValueError(
            "filter_order is required when highpass_cutoff or lowpass_cutoff is set."
        )
    if filter_order is not None and not wants_band:
        raise ValueError(
            "filter_order was set but neither highpass_cutoff nor lowpass_cutoff was provided."
        )
    if filter_order is not None and int(filter_order) < 1:
        raise ValueError(f"filter_order must be >= 1; got {filter_order}.")

    if notch_frequency is not None and quality_factor is None:
        raise ValueError(
            "quality_factor is required when notch_frequency is set."
        )
    if quality_factor is not None and notch_frequency is None:
        raise ValueError(
            "quality_factor was set but notch_frequency was not provided."
        )
    if quality_factor is not None and float(quality_factor) <= 0:
        raise ValueError(f"quality_factor must be positive; got {quality_factor}.")

    nyquist = float(sampling_rate) / 2.0
    for name, value in (
        ("highpass_cutoff", highpass_cutoff),
        ("lowpass_cutoff", lowpass_cutoff),
        ("notch_frequency", notch_frequency),
    ):
        if value is None:
            continue
        if not (0.0 < float(value) < nyquist):
            raise ValueError(
                f"{name} must satisfy 0 < f < fs/2 "
                f"(fs={sampling_rate} -> fs/2={nyquist}); got {value}."
            )


def preprocess_ecg(
    ecg_signal: Union[np.ndarray, list, tuple],
    sampling_rate: Union[int, float],
    highpass_cutoff: Optional[float] = None,
    filter_order: Optional[int] = None,
    lowpass_cutoff: Optional[float] = None,
    notch_frequency: Optional[float] = None,
    quality_factor: Optional[float] = None,
    poly_degree: Optional[int] = None,
    max_nan_frac: float = 0.01,
) -> np.ndarray:
    """
    Preprocess an ECG signal by applying optional detrending and filters.

    The preprocessing pipeline consists of:
        1. Coerce to a 1-D float vector (column/row vectors are raveled).
        2. Sparse-NaN interpolation (or reject if too many NaNs).
        3. Optional polynomial detrending.
        4. Optional high-pass filtering.
        5. Optional notch filtering.
        6. Optional low-pass filtering.

    Parameters
    ----------
    ecg_signal : array-like
        The raw ECG signal. Must be 1-D, or a column/row vector with one axis of
        length 1. May contain NaNs.
    sampling_rate : int or float
        Sampling rate of the ECG signal in Hz.
    highpass_cutoff : float, optional
        High-pass filter cutoff frequency in Hz. If set, ``filter_order`` is required.
    filter_order : int, optional
        Filter order for Butterworth filters. Required when either bandpass cutoff
        is set; must not be set alone.
    lowpass_cutoff : float, optional
        Low-pass filter cutoff frequency in Hz. If set, ``filter_order`` is required.
    notch_frequency : float, optional
        Frequency in Hz for notch filtering (e.g., 50 or 60 Hz). If set,
        ``quality_factor`` is required.
    quality_factor : float, optional
        Quality factor for notch filter. Required if ``notch_frequency`` is set;
        must not be set alone.
    poly_degree : int, optional
        Degree of polynomial for detrending. If None, detrending is skipped.
    max_nan_frac : float, default=0.01
        Maximum allowable fraction of NaN values. Below this threshold, NaNs are
        linearly interpolated from neighboring finite samples (e.g., 0.01 = 1%).

    Returns
    -------
    np.ndarray
        Preprocessed ECG signal as a 1-D float NumPy array.

    Raises
    ------
    ValueError
        On invalid shape, unpaired filter arguments, out-of-range cutoffs,
        excessive NaNs, or an all-NaN signal.
    """
    _validate_filter_args(
        sampling_rate=float(sampling_rate),
        highpass_cutoff=highpass_cutoff,
        lowpass_cutoff=lowpass_cutoff,
        filter_order=filter_order,
        notch_frequency=notch_frequency,
        quality_factor=quality_factor,
    )

    ecg_processed = _as_1d_ecg(ecg_signal)
    if ecg_processed.size == 0:
        raise ValueError("ECG signal is empty.")

    # --- Step 0: Handle sparse NaNs via linear interpolation ---
    nan_mask = np.isnan(ecg_processed)
    if nan_mask.all():
        raise ValueError("All values are NaN.")
    if nan_mask.any():
        nan_frac = float(nan_mask.mean())
        if nan_frac > float(max_nan_frac):
            raise ValueError(
                f"NaN fraction {nan_frac:.4f} exceeds max_nan_frac={max_nan_frac}."
            )
        good = ~nan_mask
        x = np.arange(ecg_processed.size)
        ecg_processed[nan_mask] = np.interp(
            x[nan_mask], x[good], ecg_processed[good]
        )

    # --- Step 1: Polynomial Detrending ---
    if poly_degree is not None:
        if int(poly_degree) < 0:
            raise ValueError(f"poly_degree must be >= 0; got {poly_degree}.")
        x = np.arange(ecg_processed.size)
        coeffs = np.polyfit(x, ecg_processed, deg=poly_degree)
        ecg_processed -= np.polyval(coeffs, x)

    # --- Step 2: High-Pass Filter ---
    if highpass_cutoff is not None:
        b_hp, a_hp = signal.butter(
            filter_order, highpass_cutoff, btype="high", fs=sampling_rate
        )
        ecg_processed = signal.filtfilt(b_hp, a_hp, ecg_processed)

    # --- Step 3: Notch Filter ---
    if notch_frequency is not None:
        b_notch, a_notch = signal.iirnotch(
            notch_frequency, quality_factor, sampling_rate
        )
        ecg_processed = signal.filtfilt(b_notch, a_notch, ecg_processed)

    # --- Step 4: Low-Pass Filter ---
    if lowpass_cutoff is not None:
        b_lp, a_lp = signal.butter(
            filter_order, lowpass_cutoff, btype="low", fs=sampling_rate
        )
        ecg_processed = signal.filtfilt(b_lp, a_lp, ecg_processed)

    return ecg_processed
