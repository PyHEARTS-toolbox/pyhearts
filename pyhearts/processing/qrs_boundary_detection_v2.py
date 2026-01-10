"""
Improved QRS boundary detection using signal amplitude and derivative.

This version uses a simpler, more direct approach to match established QRS boundary detection methods.
"""

import numpy as np
from typing import Optional


def detect_qrs_onset_derivative(
    signal: np.ndarray,
    q_peak_idx: Optional[int],
    r_peak_idx: int,
    sampling_rate: float,
    search_window_ms: float = 100.0,
    verbose: bool = False,
    cycle_idx: Optional[int] = None,
) -> int:
    """
    Detect QRS onset using signal amplitude and derivative.
    
    Uses a simpler approach: find where signal starts deflecting from baseline.
    """
    if r_peak_idx < 0 or r_peak_idx >= len(signal):
        return max(0, r_peak_idx - int(round(40 * sampling_rate / 1000.0)))
    
    # Determine search start point
    if q_peak_idx is not None and 0 <= q_peak_idx < r_peak_idx:
        search_start = q_peak_idx
        search_window_ms = 60.0  # QRS onset is typically 20-40ms before Q
    else:
        search_start = r_peak_idx
        search_window_ms = 100.0
    
    search_window_samples = int(round(search_window_ms * sampling_rate / 1000.0))
    search_begin = max(0, search_start - search_window_samples)
    
    if search_begin >= search_start or search_start - search_begin < 3:
        return max(0, search_start - int(round(40 * sampling_rate / 1000.0)))
    
    search_segment = signal[search_begin:search_start]
    if len(search_segment) < 3:
        return max(0, search_start - int(round(40 * sampling_rate / 1000.0)))
    
    # Estimate baseline from early part of segment
    baseline_window = max(5, min(20, len(search_segment)//3))
    baseline_est = np.median(search_segment[:baseline_window])
    baseline_std = np.std(search_segment[:baseline_window])
    
    # Get signal value at Q (or R if no Q)
    q_signal_val = search_segment[-1]  # Last point is at search_start (Q or R)
    
    # Compute derivative
    derivative = np.diff(search_segment)
    if len(derivative) < 2:
        return max(0, search_start - int(round(40 * sampling_rate / 1000.0)))
    
    # Find QRS onset: point where signal starts deflecting from baseline
    # Look for point where:
    # 1. Signal is close to baseline (within 1.5 std)
    # 2. Derivative magnitude is small (< threshold based on QRS derivative)
    # 3. Moving forward from this point, signal starts changing
    
    # Use QRS region to estimate derivative threshold
    qrs_window_ms = 50.0
    qrs_window_samples = int(round(qrs_window_ms * sampling_rate / 1000.0))
    qrs_start = max(0, r_peak_idx - qrs_window_samples)
    qrs_end_ref = min(len(signal), r_peak_idx + qrs_window_samples)
    qrs_segment = signal[qrs_start:qrs_end_ref]
    qrs_derivative = np.diff(qrs_segment) if len(qrs_segment) > 1 else np.array([0])
    max_qrs_derivative = float(np.max(np.abs(qrs_derivative))) if len(qrs_derivative) > 0 else 1.0
    
    # Threshold: 8% of max QRS derivative (same as end detection for consistency)
    deriv_threshold = max_qrs_derivative * 0.08
    
    qrs_onset_idx = search_start  # Default
    
    # Search backwards for onset point
    for i in range(len(search_segment) - 2, max(0, len(search_segment)//4), -1):
        signal_val = search_segment[i]
        signal_at_baseline = abs(signal_val - baseline_est) < baseline_std * 1.5
        
        if i < len(derivative):
            deriv_mag = abs(derivative[i])
            deriv_small = deriv_mag < deriv_threshold
            
            # Check if signal ahead is starting to change (moving toward Q)
            if i + 1 < len(search_segment):
                signal_change = abs(search_segment[i+1] - signal_val)
                signal_changing = signal_change > baseline_std * 0.5
            else:
                signal_changing = False
            
            if signal_at_baseline and deriv_small:
                # Found candidate - this is likely QRS onset
                qrs_onset_idx = search_begin + i
                # Continue searching backwards to find earliest valid point (but limit search)
                max_backward_search = int(round(15 * sampling_rate / 1000.0))  # 15ms max
                for j in range(i - 1, max(0, i - max_backward_search), -1):
                    if j < len(search_segment) and j < len(derivative):
                        if abs(search_segment[j] - baseline_est) < baseline_std * 1.5 and abs(derivative[j]) < deriv_threshold:
                            qrs_onset_idx = search_begin + j
                        else:
                            break
                break
    
    # Ensure onset is before R peak
    if qrs_onset_idx >= r_peak_idx:
        qrs_onset_idx = max(0, r_peak_idx - int(round(40 * sampling_rate / 1000.0)))
    
    # Ensure not too far before R (max 150ms)
    max_distance_samples = int(round(150 * sampling_rate / 1000.0))
    if r_peak_idx - qrs_onset_idx > max_distance_samples:
        qrs_onset_idx = max(0, r_peak_idx - max_distance_samples)
    
    if verbose and cycle_idx is not None:
        print(f"[Cycle {cycle_idx}]: QRS onset detected at {qrs_onset_idx} (Q={q_peak_idx}, R={r_peak_idx}, distance={r_peak_idx - qrs_onset_idx} samples)")
    
    return int(qrs_onset_idx)


def detect_qrs_end_derivative(
    signal: np.ndarray,
    s_peak_idx: Optional[int],
    r_peak_idx: int,
    sampling_rate: float,
    search_window_ms: float = 100.0,
    verbose: bool = False,
    cycle_idx: Optional[int] = None,
) -> int:
    """
    Detect QRS end using derivative-based threshold crossing (original approach).
    """
    if r_peak_idx < 0 or r_peak_idx >= len(signal):
        return min(len(signal) - 1, r_peak_idx + int(round(40 * sampling_rate / 1000.0)))
    
    # Determine search start point
    if s_peak_idx is not None and r_peak_idx < s_peak_idx < len(signal):
        search_start = s_peak_idx
    else:
        search_start = r_peak_idx
    
    search_window_samples = int(round(search_window_ms * sampling_rate / 1000.0))
    search_end = min(len(signal) - 1, search_start + search_window_samples)
    
    if search_start >= search_end:
        return min(len(signal) - 1, search_start + int(round(40 * sampling_rate / 1000.0)))
    
    search_segment = signal[search_start:search_end + 1]
    if len(search_segment) < 3:
        return min(len(signal) - 1, search_start + int(round(40 * sampling_rate / 1000.0)))
    
    # Compute derivative
    derivative = np.diff(search_segment)
    if len(derivative) < 2:
        return min(len(signal) - 1, search_start + int(round(40 * sampling_rate / 1000.0)))
    
    # Find maximum absolute derivative in QRS region (for threshold)
    qrs_window_ms = 50.0
    qrs_window_samples = int(round(qrs_window_ms * sampling_rate / 1000.0))
    qrs_start = max(0, r_peak_idx - qrs_window_samples)
    qrs_end_ref = min(len(signal), r_peak_idx + qrs_window_samples)
    qrs_segment = signal[qrs_start:qrs_end_ref]
    qrs_derivative = np.diff(qrs_segment) if len(qrs_segment) > 1 else np.array([0])
    max_qrs_derivative = float(np.max(np.abs(qrs_derivative))) if len(qrs_derivative) > 0 else 1.0
    
    # Also compute baseline derivative (after QRS) for adaptive threshold
    baseline_window_ms = 50.0
    baseline_window_samples = int(round(baseline_window_ms * sampling_rate / 1000.0))
    baseline_start = search_end
    baseline_end = min(len(signal), search_end + baseline_window_samples)
    if baseline_end > baseline_start and baseline_end - baseline_start > 3:
        baseline_segment = signal[baseline_start:baseline_end]
        baseline_derivative = np.diff(baseline_segment) if len(baseline_segment) > 1 else np.array([0])
        baseline_deriv_mag = float(np.median(np.abs(baseline_derivative))) if len(baseline_derivative) > 0 else 0.0
    else:
        baseline_deriv_mag = 0.0
    
    # Adaptive threshold: use higher of baseline noise or fraction of QRS derivative
    threshold_fraction = 0.08  # 8% of max QRS derivative
    threshold_from_qrs = max_qrs_derivative * threshold_fraction
    threshold_from_baseline = baseline_deriv_mag * 2.0 if baseline_deriv_mag > 0 else 0.0
    threshold = max(threshold_from_qrs, threshold_from_baseline)
    
    # Ensure minimum threshold
    min_threshold = max_qrs_derivative * 0.05  # At least 5% of max
    threshold = max(threshold, min_threshold)
    
    # Search forwards from search_start for derivative threshold crossing
    # QRS end is where derivative crosses threshold going from high to low
    qrs_end_idx = search_start  # Default: at search_start
    
    # Look for threshold crossing: find where derivative magnitude crosses threshold
    # going forwards (from high derivative in QRS to low derivative after QRS)
    found_crossing = False
    for i in range(len(derivative) - 1):
        deriv_mag = abs(derivative[i])
        deriv_mag_next = abs(derivative[i+1]) if i+1 < len(derivative) else deriv_mag
        
        # Look for crossing: derivative goes from above threshold to below threshold
        # This indicates transition from QRS complex to baseline
        if deriv_mag >= threshold and deriv_mag_next < threshold:
            # Found crossing: QRS end is at the point where derivative drops below threshold
            qrs_end_idx = search_start + i + 1
            found_crossing = True
            break
        elif deriv_mag < threshold and not found_crossing:
            # Already in baseline region
            qrs_end_idx = search_start + i
            found_crossing = True
            break
    
    # If no crossing found, use point where derivative becomes consistently small
    if not found_crossing:
        # Find first point going forwards where derivative is consistently below threshold
        consecutive_below = 0
        required_consecutive = max(3, int(round(5 * sampling_rate / 1000.0)))  # 5ms of low derivative
        for i in range(len(derivative)):
            if abs(derivative[i]) < threshold:
                consecutive_below += 1
                if consecutive_below >= required_consecutive:
                    qrs_end_idx = search_start + i - required_consecutive + 1
                    found_crossing = True
                    break
            else:
                consecutive_below = 0
    
    # Ensure end is after R peak
    if qrs_end_idx <= r_peak_idx:
        qrs_end_idx = min(len(signal) - 1, r_peak_idx + int(round(40 * sampling_rate / 1000.0)))
    
    # Ensure not too far after R (max 150ms)
    max_distance_samples = int(round(150 * sampling_rate / 1000.0))
    if qrs_end_idx - r_peak_idx > max_distance_samples:
        qrs_end_idx = min(len(signal) - 1, r_peak_idx + max_distance_samples)
    
    if verbose and cycle_idx is not None:
        print(f"[Cycle {cycle_idx}]: QRS end detected at {qrs_end_idx} (R={r_peak_idx}, S={s_peak_idx}, distance={qrs_end_idx - r_peak_idx} samples)")
    
    return int(qrs_end_idx)

