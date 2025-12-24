"""
ECGPUWAVE-style T wave detection using derivative-based approach.

This module implements T wave detection following ECGPUWAVE's method:
- Uses low-pass filtered derivative signal (like dbuf)
- Finds T peak as zero-crossing in derivative
- Uses adaptive threshold for boundary detection (not peak validation)
"""

import numpy as np
from typing import Optional, Tuple
from scipy.signal import butter, filtfilt, savgol_filter


def compute_filtered_derivative(
    signal: np.ndarray,
    sampling_rate: float,
    lowpass_cutoff: float = 40.0,
) -> np.ndarray:
    """
    Compute low-pass filtered derivative signal (ECGPUWAVE's dbuf).
    
    ECGPUWAVE applies a low-pass filter (40 Hz) to the signal, then computes
    the derivative. This creates a smoothed derivative that emphasizes T wave
    transitions while reducing noise.
    
    Parameters
    ----------
    signal : np.ndarray
        Input ECG signal.
    sampling_rate : float
        Sampling rate in Hz.
    lowpass_cutoff : float, default 40.0
        Low-pass filter cutoff frequency in Hz (ECGPUWAVE uses 40 Hz).
    
    Returns
    -------
    np.ndarray
        Filtered derivative signal (like ECGPUWAVE's dbuf).
    """
    if len(signal) < 3:
        return np.zeros_like(signal)
    
    # Apply low-pass filter (40 Hz) - ECGPUWAVE's fpb filter
    nyq = sampling_rate / 2.0
    if lowpass_cutoff >= nyq:
        # If cutoff too high, just compute derivative
        filtered = signal
    else:
        low = lowpass_cutoff / nyq
        low = max(0.01, min(low, 0.99))
        b, a = butter(2, low, btype='low')
        filtered = filtfilt(b, a, signal)
    
    # Compute derivative
    # Use gradient for better edge handling
    derivative = np.gradient(filtered) * sampling_rate  # Scale by sampling rate
    
    return derivative


def detect_zero_crossing(
    signal: np.ndarray,
    start_idx: int,
    direction: str = "left",
) -> Optional[int]:
    """
    Detect zero-crossing in signal (ECGPUWAVE's detectar_cero).
    
    Finds the first zero-crossing from start_idx in the specified direction.
    
    Parameters
    ----------
    signal : np.ndarray
        Signal to search (typically derivative).
    start_idx : int
        Starting index for search.
    direction : {"left", "right"}, default "left"
        Direction to search ("left" = decreasing index, "right" = increasing).
    
    Returns
    -------
    int or None
        Index of zero-crossing, or None if not found.
    """
    if start_idx < 0 or start_idx >= len(signal):
        return None
    
    if direction == "left":
        # Search left (decreasing index)
        i = start_idx - 1
        while i > 0:
            if signal[i] * signal[i - 1] <= 0:  # Sign change or zero
                # Return the index closer to zero
                if abs(signal[i]) < abs(signal[i - 1]):
                    return i
                else:
                    return i - 1
            i -= 1
    else:  # direction == "right"
        # Search right (increasing index)
        i = start_idx + 1
        while i < len(signal) - 1:
            if signal[i] * signal[i + 1] <= 0:  # Sign change or zero
                # Return the index closer to zero
                if abs(signal[i]) < abs(signal[i + 1]):
                    return i
                else:
                    return i + 1
            i += 1
    
    return None


def compute_adaptive_kte(derivative_amplitude: float) -> float:
    """
    Compute adaptive threshold kte based on derivative amplitude (ECGPUWAVE style).
    
    ECGPUWAVE scales kte based on the absolute value of the derivative minimum/maximum:
    - abs(ymin) >= 0.41: kte = 7
    - abs(ymin) >= 0.35: kte = 6
    - abs(ymin) >= 0.25: kte = 5
    - abs(ymin) >= 0.10: kte = 4
    - abs(ymin) < 0.10:  kte = 3.5
    
    Parameters
    ----------
    derivative_amplitude : float
        Absolute value of derivative min or max.
    
    Returns
    -------
    float
        Adaptive threshold kte.
    """
    abs_amp = abs(derivative_amplitude)
    
    if abs_amp >= 0.41:
        return 7.0
    elif abs_amp >= 0.35:
        return 6.0
    elif abs_amp >= 0.25:
        return 5.0
    elif abs_amp >= 0.10:
        return 4.0
    else:
        return 3.5


def find_t_wave_boundary(
    derivative: np.ndarray,
    peak_idx: int,
    peak_amplitude: float,
    direction: str = "right",
    back_factor: float = 1.0,
) -> Optional[int]:
    """
    Find T wave boundary using adaptive threshold (ECGPUWAVE's creuar_umbral).
    
    Uses adaptive threshold kte to find where derivative crosses threshold,
    which defines the T wave boundary (onset or offset).
    
    Parameters
    ----------
    derivative : np.ndarray
        Filtered derivative signal.
    peak_idx : int
        Index of T peak (in derivative domain).
    peak_amplitude : float
        Amplitude at peak (derivative value).
    direction : {"left", "right"}, default "right"
        Direction to search ("left" for onset, "right" for offset).
    back_factor : float, default 1.0
        Back-off factor (ECGPUWAVE's back array).
    
    Returns
    -------
    int or None
        Index of boundary, or None if not found.
    """
    if peak_idx < 0 or peak_idx >= len(derivative):
        return None
    
    # Compute adaptive kte
    kte = compute_adaptive_kte(peak_amplitude)
    
    # Compute threshold (ECGPUWAVE logic)
    if kte / back_factor >= 1.1:
        threshold = peak_amplitude * back_factor / kte
    else:
        threshold = peak_amplitude / 1.1
    
    # Search for threshold crossing
    if direction == "right":
        # Search right (for T end)
        i = peak_idx + 1
        while i < len(derivative) - 1:
            if peak_amplitude < 0:  # Negative peak
                if derivative[i] > threshold:
                    # Find closest point to threshold
                    if abs(derivative[i - 1] - threshold) < abs(derivative[i] - threshold):
                        return i - 1
                    return i
            else:  # Positive peak
                if derivative[i] < threshold:
                    if abs(derivative[i - 1] - threshold) < abs(derivative[i] - threshold):
                        return i - 1
                    return i
            i += 1
    else:  # direction == "left"
        # Search left (for T start)
        i = peak_idx - 1
        while i > 0:
            if peak_amplitude < 0:  # Negative peak
                if derivative[i] > threshold:
                    if abs(derivative[i + 1] - threshold) < abs(derivative[i] - threshold):
                        return i + 1
                    return i
            else:  # Positive peak
                if derivative[i] < threshold:
                    if abs(derivative[i + 1] - threshold) < abs(derivative[i] - threshold):
                        return i + 1
                    return i
            i -= 1
    
    return None


def detect_t_wave_ecgpuwave_style(
    signal: np.ndarray,
    derivative: np.ndarray,
    search_start: int,
    search_end: int,
    s_end_idx: Optional[int] = None,
    sampling_rate: float = 250.0,
    verbose: bool = False,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[float]]:
    """
    Detect T wave using ECGPUWAVE-style derivative-based method.
    
    This function mirrors ECGPUWAVE's `onat` subroutine:
    1. Finds min/max in derivative signal
    2. Determines T wave morphology
    3. Finds T peak as zero-crossing in derivative
    4. Uses adaptive threshold to find boundaries
    
    Parameters
    ----------
    signal : np.ndarray
        Original ECG signal (for amplitude).
    derivative : np.ndarray
        Filtered derivative signal (dbuf).
    search_start : int
        Start index of search window (after S end + refractory).
    search_end : int
        End index of search window (before next R).
    s_end_idx : int, optional
        S wave end index (for boundary constraints).
    sampling_rate : float, default 250.0
        Sampling rate in Hz.
    verbose : bool, default False
        If True, print diagnostic messages.
    
    Returns
    -------
    tuple
        (t_peak_idx, t_start_idx, t_end_idx, t_peak_amplitude, morphology)
        - t_peak_idx: T peak position in signal
        - t_start_idx: T wave onset
        - t_end_idx: T wave offset
        - t_peak_amplitude: T peak amplitude in signal
        - morphology: T wave type (0=normal, 1=inverted, 6=not found)
    """
    if search_end <= search_start or search_start < 0 or search_end > len(derivative):
        return None, None, None, None, 6
    
    # Extract search window
    deriv_seg = derivative[search_start:search_end]
    if len(deriv_seg) < 3:
        return None, None, None, None, 6
    
    # Find min and max in derivative (ECGPUWAVE's buscamaxmin)
    imin = search_start + int(np.argmin(deriv_seg))
    imax = search_start + int(np.argmax(deriv_seg))
    ymin = derivative[imin]
    ymax = derivative[imax]
    
    if verbose:
        print(f"  Derivative min: {ymin:.4f} at {imin}, max: {ymax:.4f} at {imax}")
    
    # Check if we have valid T wave (must have both positive and negative)
    # Relaxed: allow T waves even if derivative is mostly one-sided (ECGPUWAVE handles this)
    # Only reject if derivative is completely monotonic in one direction
    if ymin > 0 and ymax > 0:
        # All positive - no T wave
        if verbose:
            print(f"  No valid T wave - all positive derivative (ymin={ymin:.4f}, ymax={ymax:.4f})")
        return None, None, None, None, 6
    if ymin < 0 and ymax < 0:
        # All negative - no T wave
        if verbose:
            print(f"  No valid T wave - all negative derivative (ymin={ymin:.4f}, ymax={ymax:.4f})")
        return None, None, None, None, 6
    
    # Determine morphology and find T peak
    # For simplicity, we'll handle normal and inverted T waves
    # (ECGPUWAVE handles 7 types, but we'll focus on the main ones)
    
    # Handle case where we have both positive and negative, or mostly one-sided
    # If mostly positive, treat as normal T; if mostly negative, treat as inverted
    if ymax > 0 and ymin <= 0:
        # Normal T wave (positive peak) - has positive derivative
        morphology = 0
        # Use the positive peak (max) as reference
        peak_ref_idx = imax
        peak_ref_amp = ymax
    elif ymin < 0 and ymax <= 0:
        # Inverted T wave (negative peak) - all negative derivative
        morphology = 1
        peak_ref_idx = imin
        peak_ref_amp = ymin
    elif abs(ymax) > abs(ymin):
        # Normal T wave (positive peak)
        morphology = 0
        peak_ref_idx = imax
        peak_ref_amp = ymax
    else:
        # Inverted T wave (negative peak)
        morphology = 1
        peak_ref_idx = imin
        peak_ref_amp = ymin
    
    # Find zero-crossing for T peak
    # Try to find zero-crossings around the peak reference
    if morphology == 0:
        # Normal T: find zero-crossing between min and max
        icero1 = detect_zero_crossing(derivative, imin, direction="right") if imin < imax else None
        icero2 = detect_zero_crossing(derivative, imax, direction="left") if imax > imin else None
    else:
        # Inverted T: find zero-crossing between max and min
        icero1 = detect_zero_crossing(derivative, imax, direction="right") if imax < imin else None
        icero2 = detect_zero_crossing(derivative, imin, direction="left") if imin > imax else None
    
    if icero1 is not None and icero2 is not None:
        t_peak_deriv_idx = (icero1 + icero2) // 2
    elif icero1 is not None:
        t_peak_deriv_idx = icero1
    elif icero2 is not None:
        t_peak_deriv_idx = icero2
    else:
        # Fallback: use the peak reference index (where derivative is max/min)
        # Then find actual signal peak near that location
        t_peak_deriv_idx = peak_ref_idx
        # Find actual signal peak in a small window around derivative peak
        search_win = min(10, (search_end - search_start) // 4)
        peak_search_start = max(search_start, t_peak_deriv_idx - search_win)
        peak_search_end = min(search_end, t_peak_deriv_idx + search_win)
        if morphology == 0:
            # Normal T: find max in signal
            peak_seg = signal[peak_search_start:peak_search_end]
            if len(peak_seg) > 0:
                t_peak_deriv_idx = peak_search_start + int(np.argmax(peak_seg))
        else:
            # Inverted T: find min in signal
            peak_seg = signal[peak_search_start:peak_search_end]
            if len(peak_seg) > 0:
                t_peak_deriv_idx = peak_search_start + int(np.argmin(peak_seg))
    
    # Get signal amplitude at peak
    if 0 <= t_peak_deriv_idx < len(signal):
        t_peak_amplitude = signal[t_peak_deriv_idx]
    else:
        t_peak_amplitude = signal[peak_ref_idx] if 0 <= peak_ref_idx < len(signal) else 0.0
    
    # Find boundaries using adaptive threshold
    if morphology == 0:
        t_start = find_t_wave_boundary(derivative, imax, ymax, direction="left", back_factor=1.0)
        t_end = find_t_wave_boundary(derivative, imin, ymin, direction="right", back_factor=1.0) if ymin < 0 else None
    else:
        t_start = find_t_wave_boundary(derivative, imin, ymin, direction="left", back_factor=1.0)
        t_end = find_t_wave_boundary(derivative, imax, ymax, direction="right", back_factor=1.0) if ymax > 0 else None
    
    # Ensure boundaries are within valid range
    if s_end_idx is not None:
        if t_start is not None and t_start < s_end_idx:
            t_start = s_end_idx + 2  # ECGPUWAVE's fallback
    
    if t_start is not None and t_start < search_start:
        t_start = search_start
    if t_end is not None and t_end > search_end:
        t_end = search_end
    
    # Validate T peak position
    if t_peak_deriv_idx < search_start or t_peak_deriv_idx >= search_end:
        if verbose:
            print(f"  T peak out of bounds: {t_peak_deriv_idx} not in [{search_start}, {search_end})")
        return None, None, None, None, 6
    
    if verbose:
        print(f"  T wave detected: peak={t_peak_deriv_idx}, start={t_start}, end={t_end}, "
              f"amplitude={t_peak_amplitude:.4f}, morphology={morphology}")
    
    return t_peak_deriv_idx, t_start, t_end, t_peak_amplitude, morphology

