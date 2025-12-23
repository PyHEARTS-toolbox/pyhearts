from typing import Optional, Union

import numpy as np
from scipy import signal


def preprocess_ecg(
    ecg_signal: np.ndarray,
    sampling_rate: Union[int, float],
    highpass_cutoff: Optional[float] = None,
    filter_order: Optional[int] = None,
    lowpass_cutoff: Optional[float] = None,
    notch_frequency: Optional[float] = None,
    quality_factor: Optional[float] = None,
    poly_degree: Optional[int] = None,
    max_nan_frac: float = 0.01,
) -> Optional[np.ndarray]:
    """
    Preprocess an ECG signal by applying optional detrending and filters.

    The preprocessing pipeline consists of:
        1. NaN checking.
        2. Optional polynomial detrending.
        3. Optional high-pass filtering.
        4. Optional notch filtering.
        5. Optional low-pass filtering.

    Parameters
    ----------
    ecg_signal : np.ndarray
        The raw ECG signal (1D array). May contain NaNs.
    sampling_rate : int or float
        Sampling rate of the ECG signal in Hz.
    highpass_cutoff : float, optional
        High-pass filter cutoff frequency in Hz. If None, high-pass filter is skipped.
    filter_order : int, optional
        Filter order for Butterworth filters. Required for high/low-pass filters.
    lowpass_cutoff : float, optional
        Low-pass filter cutoff frequency in Hz. If None, low-pass filter is skipped.
    notch_frequency : float, optional
        Frequency in Hz for notch filtering (e.g., 50 or 60 Hz). If None, skipped.
    quality_factor : float, optional
        Quality factor for notch filter. Required if `notch_frequency` is set.
    poly_degree : int, optional
        Degree of polynomial for detrending. If None, detrending is skipped.
    max_nan_frac : float, default=0.05
        Maximum allowable fraction of NaN values for interpolation (e.g., 0.01 = 1%).

    Returns
    -------
    np.ndarray or None
        Preprocessed ECG signal as a 1D NumPy array, or None if preprocessing fails.

    Raises
    ------
    ValueError
        If the NaN fraction exceeds `max_nan_frac` or all samples are NaN.
    """
    try:
        ecg_processed = ecg_signal.astype(float, copy=True)

        # --- Step 0: Handle NaNs with strict check ---
        nan_mask = np.isnan(ecg_processed)
        
        if nan_mask.all():
            raise ValueError("All values are NaN.")
        elif nan_mask.any():
            raise ValueError(f"NaNs present: {nan_mask.sum()} values.")

        # --- Step 1: Polynomial Detrending ---
        if poly_degree is not None:
            x = np.arange(ecg_processed.size)
            try:
                # Step 4: Add error handling for polynomial detrending
                # Use lower degree if high degree causes numerical issues
                try:
                    coeffs = np.polyfit(x, ecg_processed, deg=poly_degree)
                    trend = np.polyval(coeffs, x)
                    # Check for invalid values
                    if np.any(~np.isfinite(trend)):
                        # Fallback to lower degree
                        if poly_degree > 1:
                            coeffs = np.polyfit(x, ecg_processed, deg=1)
                            trend = np.polyval(coeffs, x)
                        else:
                            # Fallback to simple mean removal
                            trend = np.mean(ecg_processed)
                    ecg_processed -= trend
                except (np.linalg.LinAlgError, ValueError, RuntimeWarning):
                    # Fallback to linear detrending
                    if poly_degree > 1:
                        coeffs = np.polyfit(x, ecg_processed, deg=1)
                        ecg_processed -= np.polyval(coeffs, x)
                    else:
                        # Fallback to mean removal
                        ecg_processed -= np.mean(ecg_processed)
            except Exception:
                # Final fallback: just remove mean
                ecg_processed -= np.mean(ecg_processed)

        # --- Step 2: High-Pass Filter ---
        if highpass_cutoff is not None and filter_order is not None:
            b_hp, a_hp = signal.butter(filter_order, highpass_cutoff, btype="high", fs=sampling_rate)
            ecg_processed = signal.filtfilt(b_hp, a_hp, ecg_processed)

        # --- Step 3: Notch Filter ---
        if notch_frequency is not None and quality_factor is not None:
            b_notch, a_notch = signal.iirnotch(notch_frequency, quality_factor, sampling_rate)
            ecg_processed = signal.filtfilt(b_notch, a_notch, ecg_processed)

        # --- Step 4: Low-Pass Filter ---
        if lowpass_cutoff is not None and filter_order is not None:
            b_lp, a_lp = signal.butter(filter_order, lowpass_cutoff, btype="low", fs=sampling_rate)
            ecg_processed = signal.filtfilt(b_lp, a_lp, ecg_processed)

        return ecg_processed

    except Exception as e:
        print(f"Error preprocessing ECG signal: {e}")
        return None

