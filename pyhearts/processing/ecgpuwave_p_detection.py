"""
Derivative-based P wave detection with fixed search window.

Implements P wave detection using bandpass filtering, derivative computation, and fixed search windows:
1. Bandpass filter (1-60 Hz) to create ecgpb
2. Compute derivative (dbuf) to emphasize slopes
3. Fixed search window: 200ms before QRS to 30ms before QRS
4. Find extrema in derivative (max and min)
5. Locate peak using zero-crossings of derivative (midpoint method)
6. Validation: amplitude > 1/30 R, slope > dermax/100
7. Onset/Offset: fixed thresholds (ymax/1.35, ymin/kpe)
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.signal import butter, filtfilt

__all__ = ["detect_p_wave_ecgpuwave_style"]  # Kept for backward compatibility


def bandpass_filter_ecgpuwave(
    signal: np.ndarray,
    sampling_rate: float,
    lowcut: float = 1.0,
    highcut: float = 60.0,
    order: int = 2,
) -> np.ndarray:
    """
    Apply bandpass filter (1-60 Hz) for P-wave enhancement.
    
    Parameters
    ----------
    signal : np.ndarray
        Input ECG signal.
    sampling_rate : float
        Sampling rate in Hz.
    lowcut : float, default 1.0
        Low cutoff frequency in Hz.
    highcut : float, default 60.0
        High cutoff frequency in Hz.
    order : int, default 2
        Filter order.
    
    Returns
    -------
    np.ndarray
        Band-pass filtered signal.
    """
    nyq = sampling_rate / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    
    # Ensure cutoffs are valid
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    try:
        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        return filtered
    except Exception:
        # Fallback: return original signal if filtering fails
        return signal


def detect_p_wave_ecgpuwave_style(
    signal: np.ndarray,
    qrs_onset_idx: int,
    r_peak_idx: Optional[int] = None,
    r_amplitude: Optional[float] = None,
    sampling_rate: float = 250.0,
    previous_t_end_idx: Optional[int] = None,
    verbose: bool = False,
    cycle_idx: Optional[int] = None,
) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[int]]:
    """
    Detect P wave using derivative-based algorithm with fixed search window.
    
    Steps:
    1. Bandpass filter (1-60 Hz) to create ecgpb
    2. Compute derivative (dbuf) to emphasize slopes
    3. Define search window: 200ms before QRS to 30ms before QRS
    4. Find extrema in derivative (max and min)
    5. Validation: amplitude > 1/30 R, slope > dermax/100
    6. Locate peak using zero-crossings of derivative (midpoint method)
    7. Find onset/offset using fixed thresholds
    
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
    # Bandpass filter (1-60 Hz) - ~1 Hz high-pass, 60 Hz low-pass for P-wave enhancement
    ecgpb = bandpass_filter_ecgpuwave(signal, sampling_rate, lowcut=1.0, highcut=60.0, order=2)
    
    # Compute derivative (dbuf) to emphasize slopes
    dbuf = np.diff(ecgpb)
    
    # Step 2: Define search window
    # Start window (ibw): 200 ms before QRS onset
    # End window (iew): 30 ms before QRS onset
    ibw_ms = 200.0
    iew_ms = 30.0
    
    ibw_samples = int(round(ibw_ms * sampling_rate / 1000.0))
    iew_samples = int(round(iew_ms * sampling_rate / 1000.0))
    
    search_start = max(0, qrs_onset_idx - ibw_samples)
    search_end = max(0, qrs_onset_idx - iew_samples)
    
    # Avoid overlap with previous T wave end if present
    if previous_t_end_idx is not None and previous_t_end_idx > search_start:
        search_start = min(previous_t_end_idx + 1, search_end - 1)
    
    if search_end <= search_start or search_end - search_start < 3:
        if verbose:
            print(f"[Cycle {cycle_idx}]: P search window too small: start={search_start}, end={search_end}")
        return None, None, None, None
    
    # Extract search window
    # Note: dbuf is one sample shorter than ecgpb (derivative of ecgpb)
    search_window_ecgpb = ecgpb[search_start:search_end]
    
    # Derivative indices: dbuf[i] corresponds to derivative between ecgpb[i] and ecgpb[i+1]
    # So dbuf[search_start:search_end-1] corresponds to derivatives in the search window
    dbuf_start = max(0, search_start)
    dbuf_end = min(len(dbuf), search_end - 1)
    
    if dbuf_end <= dbuf_start:
        if verbose:
            print(f"[Cycle {cycle_idx}]: Derivative window too small: start={dbuf_start}, end={dbuf_end}")
        return None, None, None, None
    
    search_window_dbuf = dbuf[dbuf_start:dbuf_end]
    
    if len(search_window_dbuf) < 2:
        if verbose:
            print(f"[Cycle {cycle_idx}]: Derivative window too small: {len(search_window_dbuf)}")
        return None, None, None, None
    
    # Step 3: Find extrema in derivative
    # Find maximum (ymax, ima) and minimum (ymin, imi) in search window
    ymax = float(np.max(search_window_dbuf))
    ymin = float(np.min(search_window_dbuf))
    ima = int(np.argmax(search_window_dbuf))  # Index in derivative window
    imi = int(np.argmin(search_window_dbuf))  # Index in derivative window
    
    # Map to absolute indices in original signal
    # ima/imi are indices in search_window_dbuf, which starts at dbuf_start
    ima_abs = dbuf_start + ima
    imi_abs = dbuf_start + imi
    
    # Check if minimum comes before maximum (inverted P wave)
    is_inverted = imi < ima
    
    if verbose:
        print(f"[Cycle {cycle_idx}]: P detection - ymax={ymax:.4f} at {ima_abs}, ymin={ymin:.4f} at {imi_abs}, inverted={is_inverted}")
    
    # Step 4: Validation checks
    # Amplitude check: maximum amplitude in ecgpb must be > 1/30 of R wave amplitude
    max_amplitude_ecgpb = float(np.max(np.abs(search_window_ecgpb)))
    
    if r_amplitude is not None and r_amplitude > 0:
        min_amplitude_ratio = 1.0 / 30.0  # Amplitude validation threshold: P wave must be > 1/30 of R wave
        min_amplitude = min_amplitude_ratio * abs(r_amplitude)
        if max_amplitude_ecgpb < min_amplitude:
            if verbose:
                print(f"[Cycle {cycle_idx}]: P wave rejected - amplitude {max_amplitude_ecgpb:.4f} < {min_amplitude:.4f} (1/30 R)")
            return None, None, None, None
    
    # Slope check: derivative extrema must exceed thresholds (typically dermax/100)
    dermax = float(np.max(np.abs(dbuf))) if len(dbuf) > 0 else 1.0
    slope_threshold = dermax / 100.0
    
    if abs(ymax) < slope_threshold and abs(ymin) < slope_threshold:
        if verbose:
            print(f"[Cycle {cycle_idx}]: P wave rejected - slope thresholds not met (ymax={ymax:.4f}, ymin={ymin:.4f}, threshold={slope_threshold:.4f})")
        return None, None, None, None
    
    # Step 5: Locate P wave peak using zero-crossings
    # Find zero-crossings of derivative around maximum and minimum
    # P peak (ippos1) is the midpoint between these zero-crossings
    
    # For positive P wave: find zero-crossings around maximum
    # For inverted P wave: find zero-crossings around minimum
    if is_inverted:
        # Inverted P: use minimum
        center_deriv_idx = imi  # Index in search_window_dbuf
        center_signal_idx = imi_abs + 1  # Signal index is one after derivative index
    else:
        # Normal P: use maximum
        center_deriv_idx = ima  # Index in search_window_dbuf
        center_signal_idx = ima_abs + 1  # Signal index is one after derivative index
    
    # Find zero-crossings in derivative around the extrema
    # Search in a window around the extrema (±50ms)
    zero_crossing_window_ms = 50.0
    zero_crossing_window_samples = int(round(zero_crossing_window_ms * sampling_rate / 1000.0))
    
    # Extract derivative segment around extrema (in search_window_dbuf coordinates)
    deriv_start_rel = max(0, center_deriv_idx - zero_crossing_window_samples)
    deriv_end_rel = min(len(search_window_dbuf), center_deriv_idx + zero_crossing_window_samples + 1)
    deriv_segment = search_window_dbuf[deriv_start_rel:deriv_end_rel]
    
    if len(deriv_segment) < 3:
        # Fallback: use extrema position directly
        p_peak_idx = center_signal_idx
        p_amplitude = float(ecgpb[p_peak_idx])
    else:
        # Find zero-crossings: where derivative changes sign
        sign_changes = np.diff(np.sign(deriv_segment))
        zero_crossings = np.where(sign_changes != 0)[0]
        
        if len(zero_crossings) >= 2:
            # Use first and last zero-crossing, take midpoint
            zc1_rel = deriv_start_rel + zero_crossings[0]
            zc2_rel = deriv_start_rel + zero_crossings[-1]
            midpoint_deriv_rel = (zc1_rel + zc2_rel) // 2
            # Map back to absolute signal index: derivative index + 1 = signal index
            p_peak_idx = dbuf_start + midpoint_deriv_rel + 1
        elif len(zero_crossings) == 1:
            # Single zero-crossing, use it
            zc_rel = deriv_start_rel + zero_crossings[0]
            p_peak_idx = dbuf_start + zc_rel + 1
        else:
            # No zero-crossings, use extrema position
            p_peak_idx = center_signal_idx
        
        # Ensure valid index
        p_peak_idx = np.clip(p_peak_idx, search_start, search_end - 1)
        p_amplitude = float(ecgpb[p_peak_idx])
    
    # Check for two distinct peaks (>20 ms apart)
    # (Dual peak detection records both, but we'll use the first/larger one for now)
    # TODO: Implement dual peak detection if needed
    
    # Step 6: Find P wave onset (P1)
    # Threshold: umbral = ymax / 1.35 (for normal) or ymin / 1.35 (for inverted)
    if is_inverted:
        umbral_onset = abs(ymin) / 1.35
    else:
        umbral_onset = ymax / 1.35
    
    # Search backward from maximum/minimum until crossing threshold
    p_onset_idx = p_peak_idx
    for i in range(p_peak_idx - 1, search_start - 1, -1):
        if i < 0 or i >= len(ecgpb):
            break
        signal_value = abs(ecgpb[i])
        if signal_value < umbral_onset:
            p_onset_idx = i
            break
    
    # Ensure onset is not too far from QRS (<240 ms) and does not overlap with previous T wave
    p_qrs_distance_ms = ((qrs_onset_idx - p_onset_idx) / sampling_rate) * 1000.0
    if p_qrs_distance_ms > 240.0:
        # Adjust search window and retry (simplified: just use peak position)
        if verbose:
            print(f"[Cycle {cycle_idx}]: P onset too far from QRS ({p_qrs_distance_ms:.1f}ms > 240ms), using peak position")
        p_onset_idx = p_peak_idx
    
    if previous_t_end_idx is not None and p_onset_idx <= previous_t_end_idx:
        p_onset_idx = previous_t_end_idx + 1
    
    # Step 7: Find P wave offset (P2)
    # Threshold: umbral = ymin / kpe (kpe typically 2) for normal, or ymax / kpe for inverted
    kpe = 2.0
    if is_inverted:
        umbral_offset = abs(ymax) / kpe
    else:
        umbral_offset = abs(ymin) / kpe
    
    # Search forward from minimum/maximum until crossing threshold
    p_offset_idx = p_peak_idx
    for i in range(p_peak_idx + 1, min(search_end, qrs_onset_idx)):
        if i >= len(ecgpb):
            break
        signal_value = abs(ecgpb[i])
        if signal_value < umbral_offset:
            p_offset_idx = i
            break
    
    # If offset extends beyond QRS onset, use the minimum in that region instead
    if p_offset_idx >= qrs_onset_idx:
        # Find minimum between peak and QRS onset
        if p_peak_idx < qrs_onset_idx:
            offset_segment = ecgpb[p_peak_idx:qrs_onset_idx]
            if len(offset_segment) > 0:
                min_idx_rel = int(np.argmin(np.abs(offset_segment)))
                p_offset_idx = p_peak_idx + min_idx_rel
    
    # Step 8: Final validation
    # Check noise level: P wave amplitude must be > 1.5× noise level
    # (We'll use a simplified check: amplitude > threshold)
    if r_amplitude is not None and r_amplitude > 0:
        noise_level = abs(r_amplitude) / 30.0  # Approximate noise level
        if abs(p_amplitude) < 1.5 * noise_level:
            if verbose:
                print(f"[Cycle {cycle_idx}]: P wave rejected - amplitude {abs(p_amplitude):.4f} < 1.5× noise level {1.5*noise_level:.4f}")
            return None, None, None, None
    
    # Temporal checks: onset < peak < offset
    if not (p_onset_idx < p_peak_idx < p_offset_idx):
        if verbose:
            print(f"[Cycle {cycle_idx}]: P wave rejected - invalid temporal order: onset={p_onset_idx}, peak={p_peak_idx}, offset={p_offset_idx}")
        return None, None, None, None
    
    # Duration check: < 180 ms
    duration_ms = ((p_offset_idx - p_onset_idx) / sampling_rate) * 1000.0
    if duration_ms > 180.0:
        if verbose:
            print(f"[Cycle {cycle_idx}]: P wave rejected - duration {duration_ms:.1f}ms > 180ms")
        return None, None, None, None
    
    # No overlap with previous T wave (already checked in onset)
    
    if verbose:
        print(f"[Cycle {cycle_idx}]: P wave detected - peak={p_peak_idx}, amplitude={p_amplitude:.4f}, "
              f"onset={p_onset_idx}, offset={p_offset_idx}, duration={duration_ms:.1f}ms")
    
    return p_peak_idx, p_amplitude, p_onset_idx, p_offset_idx

