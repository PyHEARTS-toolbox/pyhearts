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
        search_limit = max(0, start_idx - max_search) if max_search is not None else 0
        while i > search_limit:
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
        search_limit = min(len(signal) - 1, start_idx + max_search) if max_search is not None else len(signal) - 1
        while i < search_limit:
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
    r_peak_idx: Optional[int] = None,
    r_peak_value: Optional[float] = None,
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
    
    # Determine morphology based on SIGNAL values relative to baseline/R peak
    # For detrended signals, we need to be careful - detrending can shift baseline
    # Use the signal values relative to the search window baseline to determine morphology
    
    # First, define the T wave region to check signal values
    t_region_start_temp = min(imin, imax)
    t_region_end_temp = max(imin, imax)
    # Expand slightly to capture full T wave
    expansion_temp = int(round(50.0 * sampling_rate / 1000.0))
    t_region_start_temp = max(search_start, t_region_start_temp - expansion_temp)
    t_region_end_temp = min(search_end - 1, t_region_end_temp + expansion_temp)
    
    # Check signal values in T region to determine morphology
    t_region_signal_temp = signal[t_region_start_temp:t_region_end_temp + 1]
    if len(t_region_signal_temp) > 0:
        signal_max_in_region = np.max(t_region_signal_temp)
        signal_min_in_region = np.min(t_region_signal_temp)
        signal_max_idx_in_region = t_region_start_temp + int(np.argmax(t_region_signal_temp))
        signal_min_idx_in_region = t_region_start_temp + int(np.argmin(t_region_signal_temp))
        
        # For detrended signals, morphology determination is tricky
        # Detrending can shift baseline, making positive T waves appear negative
        # Use a more robust approach: prefer positive peak unless negative is clearly dominant
        
        # Calculate prominence of each peak relative to the median of the T region
        # This helps account for baseline shifts
        t_region_median = np.median(t_region_signal_temp)
        max_prominence = signal_max_in_region - t_region_median
        min_prominence = t_region_median - signal_min_in_region  # Note: inverted for negative
        
        # Also check if peaks are above/below the search window median
        search_window_median = np.median(signal[search_start:search_end])
        max_above_search_median = signal_max_in_region > search_window_median
        min_below_search_median = signal_min_in_region < search_window_median
        
        # Determine morphology: prefer positive (normal) unless negative is clearly dominant
        # Use multiple criteria to be more robust
        positive_score = 0
        negative_score = 0
        
        # Criterion 1: Prominence relative to T region median
        if max_prominence > 0:
            positive_score += 1
        if min_prominence > 0:
            negative_score += 1
        
        # Criterion 2: Above/below search window median
        if max_above_search_median:
            positive_score += 1
        if min_below_search_median:
            negative_score += 1
        
        # Criterion 3: Absolute prominence (but be lenient - prefer positive)
        if abs(max_prominence) > abs(min_prominence) * 0.7:  # Prefer positive unless negative is 30%+ larger
            positive_score += 1
        elif abs(min_prominence) > abs(max_prominence) * 1.3:  # Only prefer negative if it's 30%+ larger
            negative_score += 1
        
        # Determine morphology based on scores
        if positive_score >= negative_score:
            # Normal T wave (positive peak is preferred)
            morphology = 0
            peak_ref_idx = signal_max_idx_in_region
            peak_ref_amp = signal_max_in_region
        else:
            # Inverted T wave (negative peak is clearly dominant)
            morphology = 1
            peak_ref_idx = signal_min_idx_in_region
            peak_ref_amp = signal_min_in_region
    else:
        # Fallback: use derivative-based determination
        if ymax > 0 and ymin <= 0:
            morphology = 0
            peak_ref_idx = imax
            peak_ref_amp = ymax
        elif ymin < 0 and ymax <= 0:
            morphology = 1
            peak_ref_idx = imin
            peak_ref_amp = ymin
        elif abs(ymax) > abs(ymin):
            morphology = 0
            peak_ref_idx = imax
            peak_ref_amp = ymax
        else:
            morphology = 1
            peak_ref_idx = imin
            peak_ref_amp = ymin
    
    # Define T wave region: between derivative min and max
    # This identifies where the T wave actually is
    t_region_start = min(imin, imax)
    t_region_end = max(imin, imax)
    
    # Expand region to capture full T wave (ECGPUWAVE uses wider search)
    # The T wave can extend well beyond the derivative min/max region
    # Use a generous expansion, but also ensure we cover most of the search window
    # to catch late T peaks
    expansion_samples = int(round(200.0 * sampling_rate / 1000.0))  # Increased to 200ms
    t_region_start = max(search_start, t_region_start - expansion_samples)
    t_region_end = min(search_end - 1, t_region_end + expansion_samples)
    
    # Also ensure we cover at least 80% of the search window to catch late T peaks
    search_window_size = search_end - search_start
    min_region_size = int(round(0.8 * search_window_size))
    if (t_region_end - t_region_start) < min_region_size:
        # Expand to cover more of the search window
        center = (t_region_start + t_region_end) // 2
        t_region_start = max(search_start, center - min_region_size // 2)
        t_region_end = min(search_end - 1, center + min_region_size // 2)
    
    # Ensure region is valid and has minimum size
    if t_region_end <= t_region_start or (t_region_end - t_region_start) < 10:
        # Fallback: use most of the search window
        t_region_start = search_start + int(round(50.0 * sampling_rate / 1000.0))  # Start 50ms into window
        t_region_end = search_end - 1
    
    # Primary method: Find signal maximum/minimum in T wave region
    # This is the most reliable indicator of the true T peak
    t_region_signal = signal[t_region_start:t_region_end + 1]
    
    if len(t_region_signal) == 0:
        signal_peak_idx = None
    elif morphology == 0:
        # Normal T: find maximum in signal within T wave region
        t_peak_rel = int(np.argmax(t_region_signal))
        signal_peak_idx = t_region_start + t_peak_rel
    else:
        # Inverted T: find minimum in signal within T wave region
        t_peak_rel = int(np.argmin(t_region_signal))
        signal_peak_idx = t_region_start + t_peak_rel
    
    # Secondary method: Find ALL zero-crossings in T wave region (ECGPUWAVE's approach)
    # ECGPUWAVE searches for zero-crossings throughout the T wave region, not just near min/max
    # Find all zero-crossings in the T wave region
    all_zero_crossings = []
    
    # Search for all zero-crossings in t_region
    for i in range(t_region_start, min(t_region_end, len(derivative) - 1)):
        if derivative[i] * derivative[i + 1] <= 0:  # Sign change or zero
            # Return the index closer to zero
            if abs(derivative[i]) < abs(derivative[i + 1]):
                zc_idx = i
            else:
                zc_idx = i + 1
            all_zero_crossings.append(zc_idx)
    
    # Also search from derivative min/max positions (original ECGPUWAVE method)
    # This helps find zero-crossings that might be just outside t_region
    max_search_distance = int(round(100.0 * sampling_rate / 1000.0))  # Increased to 100ms
    
    if morphology == 0:
        # Normal T: imin comes before imax (derivative goes negative then positive)
        # Search from imin going right, and from imax going left
        icero1 = detect_zero_crossing(derivative, imin, direction="right", max_search=max_search_distance) if imin < imax else None
        icero2 = detect_zero_crossing(derivative, imax, direction="left", max_search=max_search_distance) if imax > imin else None
        
        # Add these to the list if they're valid and not already included
        if icero1 is not None and t_region_start <= icero1 <= t_region_end:
            if icero1 not in all_zero_crossings:
                all_zero_crossings.append(icero1)
        if icero2 is not None and t_region_start <= icero2 <= t_region_end:
            if icero2 not in all_zero_crossings:
                all_zero_crossings.append(icero2)
    else:
        # Inverted T: imax comes before imin (derivative goes positive then negative)
        # Search from imax going right, and from imin going left
        icero1 = detect_zero_crossing(derivative, imax, direction="right", max_search=max_search_distance) if imax < imin else None
        icero2 = detect_zero_crossing(derivative, imin, direction="left", max_search=max_search_distance) if imin > imax else None
        
        # Add these to the list if they're valid and not already included
        if icero1 is not None and t_region_start <= icero1 <= t_region_end:
            if icero1 not in all_zero_crossings:
                all_zero_crossings.append(icero1)
        if icero2 is not None and t_region_start <= icero2 <= t_region_end:
            if icero2 not in all_zero_crossings:
                all_zero_crossings.append(icero2)
    
    # Remove duplicates and sort
    all_zero_crossings = sorted(list(set(all_zero_crossings)))
    
    # For compatibility with existing code, keep icero1 and icero2 as the ones closest to signal apex
    # But we'll use all_zero_crossings in the selection logic
    icero1 = all_zero_crossings[0] if len(all_zero_crossings) > 0 else None
    icero2 = all_zero_crossings[-1] if len(all_zero_crossings) > 1 else None
    
    # ECGPUWAVE approach: Find zero-crossing closest to signal apex (wave apex)
    # ECGPUWAVE detects T peak as zero-crossing in derivative, but selects the
    # zero-crossing that is closest to the signal maximum/minimum (apex)
    # Analysis shows ECGPUWAVE T peaks are within 1-18 samples of signal apex
    
    # Step 1: Find signal apex (max/min) in T wave region - this is the reference point
    if signal_peak_idx is not None:
        signal_apex_idx = signal_peak_idx
    else:
        # Fallback: use peak reference index
        signal_apex_idx = peak_ref_idx
    
    # Step 2: Find zero-crossings and select the one closest to signal apex
    # Use all zero-crossings found in the T wave region
    zero_crossings = all_zero_crossings
    
    if len(zero_crossings) > 0:
        # Find zero-crossing closest to signal apex
        distances_to_apex = [abs(zc - signal_apex_idx) for zc in zero_crossings]
        closest_zc_idx = np.argmin(distances_to_apex)
        closest_zc = zero_crossings[closest_zc_idx]
        distance_to_apex = distances_to_apex[closest_zc_idx]
        
        # ECGPUWAVE uses zero-crossing if it's close to apex
        # Analysis shows ECGPUWAVE T peaks are within 1-18 samples (up to ~72ms at 250Hz)
        # Use a reasonable threshold: ~20ms or 5 samples minimum
        max_apex_distance_samples = max(5, int(round(20.0 * sampling_rate / 1000.0)))
        # But also allow up to ~80ms if zero-crossing is the only one found
        max_apex_distance_lenient = int(round(80.0 * sampling_rate / 1000.0))
        
        if distance_to_apex <= max_apex_distance_samples:
            # Zero-crossing is very close to apex - use it (ECGPUWAVE's primary method)
            t_peak_deriv_idx = closest_zc
        elif distance_to_apex <= max_apex_distance_lenient and len(zero_crossings) == 1:
            # Only one zero-crossing found and it's reasonably close - use it
            t_peak_deriv_idx = closest_zc
        else:
            # Zero-crossing is far from apex - prefer signal apex
            # This handles cases where zero-crossing is at wrong location (e.g., early ST segment)
            t_peak_deriv_idx = signal_apex_idx
    else:
        # No zero-crossings found - use signal apex directly
        t_peak_deriv_idx = signal_apex_idx
    
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

