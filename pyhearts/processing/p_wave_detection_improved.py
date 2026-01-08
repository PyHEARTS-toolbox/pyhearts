"""
Improved P wave detection based on established delineation algorithm.

Key improvements over fixed_window version:
1. Better baseline estimation (15ms before QRS)
2. Improved noise level validation
3. Proper derivative sign checks (ymax > 0, ymin < 0)
4. Iterative search window adjustment
5. Better zero-crossing detection for peak localization
6. More robust validation criteria
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.signal import butter, filtfilt

__all__ = ["detect_p_wave_improved"]


def bandpass_filter_p_wave(
    signal: np.ndarray,
    sampling_rate: float,
    lowcut: float = 1.0,
    highcut: float = 60.0,
    order: int = 2,
) -> np.ndarray:
    """Apply bandpass filter for P wave detection (1-60 Hz)."""
    nyq = sampling_rate / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    try:
        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        return filtered
    except Exception:
        return signal


def estimate_noise_level(signal_segment: np.ndarray) -> float:
    """Estimate noise level using MAD (Median Absolute Deviation)."""
    if len(signal_segment) == 0:
        return 0.0
    median = np.median(signal_segment)
    mad = np.median(np.abs(signal_segment - median))
    return mad * 1.4826  # Convert MAD to standard deviation estimate


def find_zero_crossing(derivative: np.ndarray, start_idx: int, direction: str = 'forward') -> Optional[int]:
    """
    Find zero-crossing in derivative.
    
    Parameters
    ----------
    derivative : np.ndarray
        Derivative signal
    start_idx : int
        Starting index
    direction : str
        'forward' or 'backward'
    
    Returns
    -------
    Optional[int]
        Zero-crossing index, or None if not found
    """
    if direction == 'forward':
        segment = derivative[start_idx:]
        for i in range(len(segment) - 1):
            if segment[i] * segment[i + 1] <= 0:  # Sign change
                return start_idx + i
    else:  # backward
        segment = derivative[:start_idx + 1]
        for i in range(len(segment) - 1, 0, -1):
            if segment[i - 1] * segment[i] <= 0:  # Sign change
                return i - 1
    
    return None


def detect_p_wave_improved(
    signal: np.ndarray,
    qrs_onset_idx: int,
    r_peak_idx: Optional[int] = None,
    r_amplitude: Optional[float] = None,
    sampling_rate: float = 250.0,
    previous_t_end_idx: Optional[int] = None,
    previous_p_end_idx: Optional[int] = None,
    max_derivative: Optional[float] = None,
    verbose: bool = False,
    cycle_idx: Optional[int] = None,
) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[int]]:
    """
    Improved P wave detection based on established delineation algorithm.
    
    Parameters
    ----------
    signal : np.ndarray
        ECG signal (detrended).
    qrs_onset_idx : int
        QRS onset index (start of QRS complex).
    r_peak_idx : int, optional
        R peak index (for amplitude validation).
    r_amplitude : float, optional
        R peak amplitude (for validation).
    sampling_rate : float, default 250.0
        Sampling rate in Hz.
    previous_t_end_idx : int, optional
        Previous T wave end index (to avoid overlap).
    previous_p_end_idx : int, optional
        Previous P wave end index (to avoid overlap).
    max_derivative : float, optional
        Maximum derivative value in signal (for threshold calculation).
    verbose : bool, default False
        If True, print diagnostic messages.
    cycle_idx : int, optional
        Cycle index for logging.
    
    Returns
    -------
    Tuple[Optional[int], Optional[float], Optional[int], Optional[int]]
        (p_peak_idx, p_amplitude, p_onset_idx, p_offset_idx)
        Returns (None, None, None, None) if no P wave detected.
    """
    if qrs_onset_idx < 0 or qrs_onset_idx >= len(signal):
        if verbose:
            print(f"[Cycle {cycle_idx}]: Invalid QRS onset index: {qrs_onset_idx}")
        return None, None, None, None
    
    # Step 1: Signal preprocessing
    filtered_signal = bandpass_filter_p_wave(signal, sampling_rate, lowcut=1.0, highcut=60.0, order=2)
    
    # Compute derivative
    derivative = np.diff(filtered_signal)
    
    # Step 2: Define search window (200ms before QRS to 30ms before QRS)
    search_start_ms = 200.0
    search_end_ms = 30.0
    
    search_start_samples = int(round(search_start_ms * sampling_rate / 1000.0))
    search_end_samples = int(round(search_end_ms * sampling_rate / 1000.0))
    
    ibw = max(0, qrs_onset_idx - search_start_samples)  # Initial begin window
    iew = max(0, qrs_onset_idx - search_end_samples)    # Initial end window
    
    # Avoid overlap with previous T wave end
    if previous_t_end_idx is not None and previous_t_end_idx > ibw:
        ibw = min(previous_t_end_idx + 1, iew - 1)
    
    # Avoid overlap with previous P wave end
    if previous_p_end_idx is not None and previous_p_end_idx > ibw:
        ibw = min(previous_p_end_idx + 1, iew - 1)
    
    if iew <= ibw or iew - ibw < 3:
        if verbose:
            print(f"[Cycle {cycle_idx}]: P search window too small: start={ibw}, end={iew}")
        return None, None, None, None
    
    # Step 3: Baseline estimation (15ms before QRS)
    baseline_start = max(0, qrs_onset_idx - int(round(15.0 * sampling_rate / 1000.0)))
    baseline_end = qrs_onset_idx
    if baseline_end > baseline_start:
        baseline = np.mean(filtered_signal[baseline_start:baseline_end])
    else:
        baseline = 0.0
    
    # Step 4: Find extrema in derivative
    deriv_start = max(0, ibw)
    deriv_end = min(len(derivative), iew - 1)
    
    if deriv_end <= deriv_start:
        return None, None, None, None
    
    search_window_derivative = derivative[deriv_start:deriv_end]
    search_window_signal = filtered_signal[ibw:iew]
    
    if len(search_window_derivative) < 2:
        return None, None, None, None
    
    # Find maximum and minimum in derivative
    deriv_max = float(np.max(search_window_derivative))
    deriv_min = float(np.min(search_window_derivative))
    max_idx = int(np.argmax(search_window_derivative))
    min_idx = int(np.argmin(search_window_derivative))
    
    max_idx_abs = deriv_start + max_idx
    min_idx_abs = deriv_start + min_idx
    
    # Step 5: Validation - derivative sign check (ymax > 0, ymin < 0)
    if deriv_max <= 0 or deriv_min >= 0:
        if verbose:
            print(f"[Cycle {cycle_idx}]: P wave rejected - invalid derivative signs (max={deriv_max:.4f}, min={deriv_min:.4f})")
        return None, None, None, None
    
    # Step 6: Check if minimum comes before maximum (inverted P wave)
    is_inverted = min_idx < max_idx
    
    # Step 7: Amplitude validation
    # Calculate maximum amplitude in filtered signal relative to baseline
    ecgpbmax = float(np.max(np.abs(search_window_signal - baseline)))
    
    # Get R peak value for comparison (if available)
    # PKni is the R peak position relative to search window start (ibw)
    pkni = None
    if r_peak_idx is not None and 0 <= r_peak_idx < len(filtered_signal):
        # PKni is relative to search window start
        if r_peak_idx >= ibw and r_peak_idx < iew:
            pkni = r_peak_idx - ibw
            r_value = filtered_signal[r_peak_idx]
            r_amplitude_abs = abs(r_value - baseline)
        else:
            r_value = filtered_signal[r_peak_idx]
            r_amplitude_abs = abs(r_value - baseline)
            pkni = None  # R peak outside search window
    elif r_amplitude is not None:
        r_amplitude_abs = abs(r_amplitude)
        pkni = None
    else:
        r_amplitude_abs = None
        pkni = None
    
    # Step 8: Combined validation check (derivative-based amplitude and slope validation)
    # Check if derivative extrema are too small relative to max derivative
    if max_derivative is None:
        max_derivative = float(np.max(np.abs(derivative))) if len(derivative) > 0 else 1.0
    
    slope_threshold = max_derivative / 100.0
    
    # Derivative-based validation: (ecgpbmax<=abs(Xpb(PKni)-base)/30) | 
    # ((ymax<dermax/(100)&abs(ymin)<dermax/(100))&(ymax<abs(ymin)/1.5|ymax>abs(ymin)*1.5)) | 
    # (ymax<0 | ymin>0)
    
    # Check 1: Amplitude check (if R peak available and in window)
    amplitude_check_failed = False
    if pkni is not None and pkni >= 0 and pkni < len(search_window_signal):
        r_value_in_window = search_window_signal[pkni]
        r_amplitude_in_window = abs(r_value_in_window - baseline)
        min_amplitude = r_amplitude_in_window / 30.0
        if ecgpbmax <= min_amplitude:
            amplitude_check_failed = True
            if verbose:
                print(f"[Cycle {cycle_idx}]: P wave rejected - amplitude {ecgpbmax:.4f} <= {min_amplitude:.4f} (1/30 R in window)")
    
    # Check 2: Both extrema small AND invalid ratio
    both_small = abs(deriv_max) < slope_threshold and abs(deriv_min) < slope_threshold
    invalid_ratio = abs(deriv_max) < abs(deriv_min) / 1.5 or abs(deriv_max) > abs(deriv_min) * 1.5
    slope_ratio_check_failed = both_small and invalid_ratio
    
    # Check 3: Derivative sign check (already done above, but keep for clarity)
    sign_check_failed = deriv_max <= 0 or deriv_min >= 0
    
    # Combined rejection
    if amplitude_check_failed or slope_ratio_check_failed or sign_check_failed:
        if verbose:
            if amplitude_check_failed:
                print(f"[Cycle {cycle_idx}]: P wave rejected - amplitude check failed")
            elif slope_ratio_check_failed:
                print(f"[Cycle {cycle_idx}]: P wave rejected - slope/ratio check failed (both small and invalid ratio)")
            elif sign_check_failed:
                print(f"[Cycle {cycle_idx}]: P wave rejected - invalid derivative signs")
        return None, None, None, None
    
    # Step 9: Adjust for inverted P wave
    if is_inverted:
        # Swap max and min for inverted P
        iaux = min_idx
        min_idx = max_idx
        max_idx = iaux
        yaux = deriv_min
        deriv_min = deriv_max
        deriv_max = yaux
        max_idx_abs, min_idx_abs = min_idx_abs, max_idx_abs
    
    # Step 10: Find P wave onset using threshold crossing
    # Threshold: ymax / Kpb (Kpb = 1.35)
    Kpb = 1.35
    onset_threshold = deriv_max / Kpb
    
    # Search backward from maximum
    imax = max_idx_abs
    iumb = imax
    
    # Create reversed derivative for backward search
    deriv_reversed = derivative[:imax + 1][::-1]
    threshold_crossed = False
    
    for i, dval in enumerate(deriv_reversed):
        if dval <= onset_threshold:
            iumb = imax - i
            threshold_crossed = True
            break
    
    if not threshold_crossed:
        iumb = imax
    
    # Step 11: Iterative adjustment if onset is too far from QRS (>240ms)
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        p_qrs_distance_ms = ((qrs_onset_idx - iumb) / sampling_rate) * 1000.0
        
        # Check if too far from QRS or overlaps with previous T/P end
        too_far = p_qrs_distance_ms >= 240.0
        overlaps_prev = (previous_t_end_idx is not None and iumb <= previous_t_end_idx) or \
                       (previous_p_end_idx is not None and iumb <= previous_p_end_idx)
        
        if not too_far and not overlaps_prev:
            break
        
        # Adjust search window
        ibw = ibw + int(round(20.0 * sampling_rate / 1000.0))
        
        if ibw >= iew - int(round(20.0 * sampling_rate / 1000.0)):
            if verbose:
                print(f"[Cycle {cycle_idx}]: P wave rejected - search window exhausted")
            return None, None, None, None
        
        # Recalculate extrema in new window
        deriv_start = max(0, ibw)
        deriv_end = min(len(derivative), iew - 1)
        
        if deriv_end <= deriv_start:
            return None, None, None, None
        
        search_window_derivative = derivative[deriv_start:deriv_end]
        
        if len(search_window_derivative) < 2:
            return None, None, None, None
        
        deriv_max = float(np.max(search_window_derivative))
        deriv_min = float(np.min(search_window_derivative))
        max_idx = int(np.argmax(search_window_derivative))
        min_idx = int(np.argmin(search_window_derivative))
        
        max_idx_abs = deriv_start + max_idx
        min_idx_abs = deriv_start + min_idx
        
        # Recalculate onset
        imax = max_idx_abs
        onset_threshold = deriv_max / Kpb
        deriv_reversed = derivative[:imax + 1][::-1]
        threshold_crossed = False
        
        for i, dval in enumerate(deriv_reversed):
            if dval <= onset_threshold:
                iumb = imax - i
                threshold_crossed = True
                break
        
        if not threshold_crossed:
            iumb = imax
        
        iteration += 1
    
    p_onset_idx = iumb
    
    # Step 12: Noise level check before P onset
    noise_start = max(0, p_onset_idx - int(round(40.0 * sampling_rate / 1000.0)))
    noise_end = max(0, p_onset_idx - int(round(5.0 * sampling_rate / 1000.0)))
    
    if noise_end > noise_start:
        noise_segment = filtered_signal[noise_start:noise_end]
        noise_level = estimate_noise_level(noise_segment)
    else:
        noise_level = 0.0
    
    # Step 13: Find P wave peak using zero-crossings
    # Find zero-crossing to the right of maximum
    icero1 = find_zero_crossing(derivative, imax, direction='forward')
    if icero1 is None:
        icero1 = imax
    
    # Find zero-crossing to the left of minimum
    icero2 = find_zero_crossing(derivative, min_idx_abs, direction='backward')
    if icero2 is None:
        icero2 = min_idx_abs
    
    # P peak is midpoint of zero-crossings
    p_peak_idx = int(round((icero1 + icero2) / 2.0))
    p_peak_idx = np.clip(p_peak_idx, ibw, iew - 1)
    p_amplitude = float(filtered_signal[p_peak_idx])
    
    # Step 14: Final noise level validation
    p_amplitude_relative = abs(filtered_signal[p_onset_idx] - filtered_signal[p_peak_idx])
    p_duration_samples = p_peak_idx - p_onset_idx
    p_duration_ms = (p_duration_samples / sampling_rate) * 1000.0
    
    if p_amplitude_relative < 1.5 * noise_level and p_duration_ms < 40.0:
        if verbose:
            print(f"[Cycle {cycle_idx}]: P wave rejected - amplitude {p_amplitude_relative:.4f} < 1.5× noise {1.5*noise_level:.4f}")
        return None, None, None, None
    
    # Step 15: Find P wave offset
    Kpe = 2.0
    offset_threshold = abs(deriv_min) / Kpe
    
    # Search forward from minimum
    imin = min_idx_abs
    iumb_offset = imin
    
    deriv_forward = derivative[imin:]
    for i, dval in enumerate(deriv_forward):
        if abs(dval) <= offset_threshold:
            iumb_offset = imin + i
            break
    
    # If offset extends beyond QRS onset, use minimum in that region
    if iumb_offset >= qrs_onset_idx:
        if p_peak_idx < qrs_onset_idx:
            offset_segment = filtered_signal[p_peak_idx:qrs_onset_idx]
            if len(offset_segment) > 0:
                min_idx_rel = int(np.argmin(np.abs(offset_segment)))
                iumb_offset = p_peak_idx + min_idx_rel
            else:
                iumb_offset = qrs_onset_idx - 1
    
    p_offset_idx = min(iumb_offset, qrs_onset_idx - 1)
    
    # Step 16: Final validation checks
    # Temporal order
    if not (p_onset_idx < p_peak_idx < p_offset_idx):
        if verbose:
            print(f"[Cycle {cycle_idx}]: P wave rejected - invalid temporal order")
        return None, None, None, None
    
    # Duration check (< 180ms, but allow up to 150ms for strict validation)
    duration_ms = ((p_offset_idx - p_onset_idx) / sampling_rate) * 1000.0
    if duration_ms > 180.0:
        if verbose:
            print(f"[Cycle {cycle_idx}]: P wave rejected - duration {duration_ms:.1f}ms > 180ms")
        return None, None, None, None
    
    # Final noise check: P peak to offset amplitude
    p_offset_amplitude = abs(filtered_signal[p_peak_idx] - filtered_signal[p_offset_idx])
    if p_offset_amplitude <= 1.5 * noise_level:
        if verbose:
            print(f"[Cycle {cycle_idx}]: P wave rejected - offset amplitude too small")
        return None, None, None, None
    
    if verbose:
        print(f"[Cycle {cycle_idx}]: P wave detected - peak={p_peak_idx}, amplitude={p_amplitude:.4f}, "
              f"onset={p_onset_idx}, offset={p_offset_idx}, duration={duration_ms:.1f}ms")
    
    return p_peak_idx, p_amplitude, p_onset_idx, p_offset_idx

