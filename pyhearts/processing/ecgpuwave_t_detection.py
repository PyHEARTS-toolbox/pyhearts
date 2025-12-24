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
    max_search: Optional[int] = None,
) -> Optional[int]:
    """
    Detect zero-crossing in signal (ECGPUWAVE's detectar_cero).
    
    Finds the first zero-crossing from start_idx in the specified direction.
    More robust than simple sign change - uses interpolation for sub-sample accuracy.
    
    Parameters
    ----------
    signal : np.ndarray
        Signal to search (typically derivative).
    start_idx : int
        Starting index for search.
    direction : {"left", "right"}, default "left"
        Direction to search ("left" = decreasing index, "right" = increasing).
    max_search : int, optional
        Maximum number of samples to search (None = search until boundary).
    
    Returns
    -------
    int or None
        Index of zero-crossing, or None if not found.
    """
    if start_idx < 0 or start_idx >= len(signal):
        return None
    
    if direction == "left":
        # Search left (decreasing index)
        i = start_idx
        search_count = 0
        while i > 0:
            if max_search is not None and search_count >= max_search:
                break
            if i > 0 and signal[i] * signal[i - 1] <= 0:  # Sign change or zero
                # Use linear interpolation for sub-sample accuracy
                if abs(signal[i]) < 1e-10:  # Already at zero
                    return i
                elif abs(signal[i - 1]) < 1e-10:  # Previous sample at zero
                    return i - 1
                else:
                    # Linear interpolation to find exact zero-crossing
                    # y = y0 + (y1 - y0) * t, solve for t where y = 0
                    # t = -y0 / (y1 - y0)
                    t = -signal[i - 1] / (signal[i] - signal[i - 1])
                    # Return the closer integer index
                    if t < 0.5:
                        return i - 1
                    else:
                        return i
            i -= 1
            search_count += 1
    else:  # direction == "right"
        # Search right (increasing index)
        i = start_idx
        search_count = 0
        while i < len(signal) - 1:
            if max_search is not None and search_count >= max_search:
                break
            if i < len(signal) - 1 and signal[i] * signal[i + 1] <= 0:  # Sign change or zero
                # Use linear interpolation for sub-sample accuracy
                if abs(signal[i]) < 1e-10:  # Already at zero
                    return i
                elif abs(signal[i + 1]) < 1e-10:  # Next sample at zero
                    return i + 1
                else:
                    # Linear interpolation to find exact zero-crossing
                    t = -signal[i] / (signal[i + 1] - signal[i])
                    # Return the closer integer index
                    if t < 0.5:
                        return i
                    else:
                        return i + 1
            i += 1
            search_count += 1
    
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
    
    # Check if we have valid T wave
    # ECGPUWAVE is very permissive - it will detect T waves even with small derivative variations
    # Only reject if derivative is completely flat (no variation)
    deriv_range = ymax - ymin
    if deriv_range < 0.01:  # Very small variation - likely noise or flat signal
        if verbose:
            print(f"  No valid T wave - derivative too flat (ymin={ymin:.4f}, ymax={ymax:.4f}, range={deriv_range:.4f})")
        return None, None, None, None, 6
    
    # Allow T waves even if derivative is mostly one-sided (ECGPUWAVE handles monophasic T waves)
    # ECGPUWAVE detects T waves with morphology types 2 (only up) and 3 (only down)
    
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
    
    # Find T peak using ECGPUWAVE's method
    # ECGPUWAVE uses zero-crossings in the derivative, but the key is finding
    # the correct zero-crossing that corresponds to the T wave peak, not an
    # early ST segment feature. We do this by:
    # 1. Identifying the T wave region using derivative min/max
    # 2. Finding signal maximum/minimum in that region (most reliable)
    # 3. Using zero-crossings as refinement, but only if they're in the T wave region
    
    # Define T wave region: between derivative min and max
    # This identifies where the T wave actually is
    t_region_start = min(imin, imax)
    t_region_end = max(imin, imax)
    
    # Expand region to capture full T wave (ECGPUWAVE uses wider search)
    # Add ~30ms on each side to ensure we capture the peak
    expansion_samples = int(round(30.0 * sampling_rate / 1000.0))
    t_region_start = max(search_start, t_region_start - expansion_samples)
    t_region_end = min(search_end - 1, t_region_end + expansion_samples)
    
    if t_region_end <= t_region_start:
        t_region_start = search_start
        t_region_end = search_end - 1
    
    # Primary method: Find signal maximum/minimum in T wave region
    # This is the most reliable indicator of the true T peak
    t_region_signal = signal[t_region_start:t_region_end + 1]
    
    if len(t_region_signal) == 0:
        signal_peak_idx = None
    elif morphology == 0:
        # Normal T: find maximum in signal
        t_peak_rel = int(np.argmax(t_region_signal))
        signal_peak_idx = t_region_start + t_peak_rel
    else:
        # Inverted T: find minimum in signal
        t_peak_rel = int(np.argmin(t_region_signal))
        signal_peak_idx = t_region_start + t_peak_rel
    
    # Secondary method: Find zero-crossings in derivative (ECGPUWAVE's approach)
    # But constrain search to avoid early ST segment detections
    # Limit search distance to ~50ms from min/max positions
    max_search_distance = int(round(50.0 * sampling_rate / 1000.0))
    
    if morphology == 0:
        # Normal T: icero1 from imin going left, icero2 from imax going right
        # For normal T: imax is before peak (rising), imin is after peak (falling)
        # Zero-crossing should be between imax and imin
        icero1 = detect_zero_crossing(derivative, imin, direction="left", max_search=max_search_distance)
        icero2 = detect_zero_crossing(derivative, imax, direction="right", max_search=max_search_distance)
        
        # Critical: Only use zero-crossings that are BETWEEN imax and imin
        # This ensures we find the T peak zero-crossing, not an early ST segment feature
        # For normal T: imax (before peak) < zero-crossing < imin (after peak)
        if icero1 is not None:
            # icero1 should be between imax and imin (searching left from imin)
            if icero1 < imax or icero1 > imin or icero1 < t_region_start or icero1 > t_region_end:
                icero1 = None  # Reject if outside valid T wave region
        if icero2 is not None:
            # icero2 should be between imax and imin (searching right from imax)
            if icero2 < imax or icero2 > imin or icero2 < t_region_start or icero2 > t_region_end:
                icero2 = None  # Reject if outside valid T wave region
    else:
        # Inverted T: icero1 from imax going left, icero2 from imin going right
        # For inverted T: imin is before peak (falling), imax is after peak (rising)
        # Zero-crossing should be between imin and imax
        icero1 = detect_zero_crossing(derivative, imax, direction="left", max_search=max_search_distance)
        icero2 = detect_zero_crossing(derivative, imin, direction="right", max_search=max_search_distance)
        
        # Critical: Only use zero-crossings that are BETWEEN imin and imax
        # For inverted T: imin (before peak) < zero-crossing < imax (after peak)
        if icero1 is not None:
            # icero1 should be between imin and imax (searching left from imax)
            if icero1 < imin or icero1 > imax or icero1 < t_region_start or icero1 > t_region_end:
                icero1 = None
        if icero2 is not None:
            # icero2 should be between imin and imax (searching right from imin)
            if icero2 < imin or icero2 > imax or icero2 < t_region_start or icero2 > t_region_end:
                icero2 = None
    
    # Choose best T peak: prefer signal max/min, but use zero-crossing average if available and close
    if signal_peak_idx is not None:
        # Use signal peak as primary
        t_peak_deriv_idx = signal_peak_idx
        
        # Refine using zero-crossings if they're close to signal peak (within ~40ms)
        refine_window = int(round(40.0 * sampling_rate / 1000.0))
        if icero1 is not None and icero2 is not None:
            zero_crossing_avg = (icero1 + icero2) // 2
            # If zero-crossing average is close to signal peak, use it (more precise)
            if abs(zero_crossing_avg - signal_peak_idx) <= refine_window:
                t_peak_deriv_idx = zero_crossing_avg
        elif icero2 is not None:
            # Use single zero-crossing if close to signal peak
            if abs(icero2 - signal_peak_idx) <= refine_window:
                t_peak_deriv_idx = icero2
    else:
        # Fallback: use zero-crossing average if available
        if icero1 is not None and icero2 is not None:
            t_peak_deriv_idx = (icero1 + icero2) // 2
        elif icero1 is not None:
            t_peak_deriv_idx = icero1
        elif icero2 is not None:
            t_peak_deriv_idx = icero2
        else:
            # Last resort: use peak reference
            t_peak_deriv_idx = peak_ref_idx
    
    # Find boundaries using adaptive threshold (needed for fallback)
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
    
    # If zero-crossings weren't found, use midpoint of boundaries (ECGPUWAVE's fallback)
    if t_peak_deriv_idx is None:
        if t_start is not None and t_end is not None:
            t_peak_deriv_idx = (t_start + t_end) // 2
        elif t_start is not None:
            t_peak_deriv_idx = t_start + 10  # Small offset from start
        elif t_end is not None:
            t_peak_deriv_idx = t_end - 10  # Small offset from end
        else:
            # Last resort: use peak reference
            t_peak_deriv_idx = peak_ref_idx
    
    # Validate T peak position (ECGPUWAVE checks if peak is outside boundaries)
    if t_start is not None and t_end is not None:
        if t_peak_deriv_idx >= t_end or t_peak_deriv_idx <= t_start:
            # Use midpoint (ECGPUWAVE's fallback: icero=itbeg(n)+(itend(n)-itbeg(n))/2)
            t_peak_deriv_idx = (t_start + t_end) // 2
            if verbose:
                print(f"  T peak outside boundaries, using midpoint: {t_peak_deriv_idx}")
    
    # Final bounds check
    if t_peak_deriv_idx < search_start or t_peak_deriv_idx >= search_end:
        if verbose:
            print(f"  T peak out of search window: {t_peak_deriv_idx} not in [{search_start}, {search_end})")
        return None, None, None, None, 6
    
    # Get signal amplitude at peak
    if 0 <= t_peak_deriv_idx < len(signal):
        t_peak_amplitude = signal[t_peak_deriv_idx]
    else:
        t_peak_amplitude = signal[peak_ref_idx] if 0 <= peak_ref_idx < len(signal) else 0.0
    
    if verbose:
        print(f"  T wave detected: peak={t_peak_deriv_idx}, start={t_start}, end={t_end}, "
              f"amplitude={t_peak_amplitude:.4f}, morphology={morphology}")
    
    return t_peak_deriv_idx, t_start, t_end, t_peak_amplitude, morphology

