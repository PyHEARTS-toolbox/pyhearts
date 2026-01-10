"""
ST segment feature extraction.

Extracts ST elevation, slope, and deviation features from the ST segment.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy.signal import savgol_filter


def extract_st_segment_features(
    signal: np.ndarray,
    s_ri_idx: Optional[int],
    t_le_idx: Optional[int],
    p_ri_idx: Optional[int],
    q_le_idx: Optional[int],
    sampling_rate: float,
    j_point_offset_ms: float = 60.0,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Extract ST segment features: elevation, slope, and deviation.
    
    Parameters
    ----------
    signal : np.ndarray
        Detrended ECG signal for the cycle.
    s_ri_idx : Optional[int]
        Right index of S wave (J point, end of QRS complex). Cycle-relative index.
    t_le_idx : Optional[int]
        Left index of T wave (start of T wave). Cycle-relative index.
    p_ri_idx : Optional[int]
        Right index of P wave (end of P wave). Cycle-relative index.
    q_le_idx : Optional[int]
        Left index of Q wave (start of QRS complex). Cycle-relative index.
    sampling_rate : float
        Sampling rate in Hz.
    j_point_offset_ms : float
        Offset from J point (S_ri_idx) to measure ST elevation, in milliseconds.
        Default: 60ms (standard clinical measurement point).
    verbose : bool
        Whether to print debug information.
    
    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        - 'ST_elevation_mv': ST elevation at J+offset (mV)
        - 'ST_slope_mv_per_s': Slope of ST segment (mV/s)
        - 'ST_deviation_mv': ST deviation from baseline (mV)
        Values are np.nan if calculation is not possible.
    """
    result = {
        'ST_elevation_mv': np.nan,
        'ST_slope_mv_per_s': np.nan,
        'ST_deviation_mv': np.nan,
    }
    
    # Need S end (J point) and T start to define ST segment
    if s_ri_idx is None or t_le_idx is None:
        if verbose:
            print(f"ST segment features: Missing S_ri_idx={s_ri_idx} or T_le_idx={t_le_idx}")
        return result
    
    # Validate indices
    if s_ri_idx < 0 or s_ri_idx >= len(signal) or t_le_idx < 0 or t_le_idx >= len(signal):
        if verbose:
            print(f"ST segment features: Invalid indices - s_ri_idx={s_ri_idx}, t_le_idx={t_le_idx}, signal_len={len(signal)}")
        return result
    
    # Ensure ST segment is valid (S end before T start)
    if s_ri_idx >= t_le_idx:
        if verbose:
            print(f"ST segment features: Invalid ST segment - s_ri_idx={s_ri_idx} >= t_le_idx={t_le_idx}")
        return result
    
    # Calculate baseline from PR segment (isoelectric line)
    baseline = np.nan
    if p_ri_idx is not None and q_le_idx is not None:
        # PR segment: from P end to Q start
        if 0 <= p_ri_idx < len(signal) and 0 <= q_le_idx < len(signal) and p_ri_idx < q_le_idx:
            pr_segment = signal[p_ri_idx:q_le_idx+1]
            if len(pr_segment) > 0:
                baseline = float(np.median(pr_segment))
                if verbose:
                    print(f"ST segment features: Baseline from PR segment = {baseline:.4f} mV")
    
    # If PR segment not available, use median of signal around ST segment
    if np.isnan(baseline):
        # Use median of signal in a window before ST segment
        baseline_window_start = max(0, s_ri_idx - int(round(50 * sampling_rate / 1000.0)))
        baseline_window_end = s_ri_idx
        if baseline_window_end > baseline_window_start:
            baseline_signal = signal[baseline_window_start:baseline_window_end]
            if len(baseline_signal) > 0:
                baseline = float(np.median(baseline_signal))
                if verbose:
                    print(f"ST segment features: Baseline from pre-ST window = {baseline:.4f} mV")
    
    # If still no baseline, use global median
    if np.isnan(baseline):
        baseline = float(np.median(signal))
        if verbose:
            print(f"ST segment features: Baseline from global median = {baseline:.4f} mV")
    
    # Extract ST segment
    st_start_idx = s_ri_idx
    st_end_idx = t_le_idx
    st_segment = signal[st_start_idx:st_end_idx+1]
    
    if len(st_segment) < 2:
        if verbose:
            print(f"ST segment features: ST segment too short ({len(st_segment)} samples)")
        return result
    
    # Smooth ST segment to reduce noise
    if len(st_segment) >= 5:
        try:
            window_length = min(5, len(st_segment))
            if window_length % 2 == 0:
                window_length -= 1
            if window_length >= 3:
                st_segment_smooth = savgol_filter(st_segment, window_length=window_length, polyorder=2, mode='interp')
            else:
                st_segment_smooth = st_segment
        except Exception:
            st_segment_smooth = st_segment
    else:
        st_segment_smooth = st_segment
    
    # Calculate ST elevation at J point + offset
    j_point_offset_samples = int(round(j_point_offset_ms * sampling_rate / 1000.0))
    elevation_idx = st_start_idx + j_point_offset_samples
    
    # Ensure elevation point is within ST segment
    if elevation_idx > st_end_idx:
        # Use end of ST segment if offset exceeds segment
        elevation_idx = st_end_idx
    elif elevation_idx < st_start_idx:
        elevation_idx = st_start_idx
    
    # Get elevation relative to ST segment start
    elevation_idx_rel = elevation_idx - st_start_idx
    if 0 <= elevation_idx_rel < len(st_segment_smooth):
        st_elevation = float(st_segment_smooth[elevation_idx_rel])
        result['ST_elevation_mv'] = st_elevation
    else:
        if verbose:
            print(f"ST segment features: Elevation index out of bounds: {elevation_idx_rel} (segment_len={len(st_segment_smooth)})")
    
    # Calculate ST slope (linear fit to ST segment)
    if len(st_segment_smooth) >= 2:
        # Time points in seconds
        time_points = np.arange(len(st_segment_smooth)) / sampling_rate
        
        # Linear regression: y = mx + b
        # Slope = covariance(x, y) / variance(x)
        x_mean = np.mean(time_points)
        y_mean = np.mean(st_segment_smooth)
        
        numerator = np.sum((time_points - x_mean) * (st_segment_smooth - y_mean))
        denominator = np.sum((time_points - x_mean) ** 2)
        
        if abs(denominator) > 1e-10:
            slope = numerator / denominator  # mV/s
            result['ST_slope_mv_per_s'] = float(slope)
        else:
            if verbose:
                print("ST segment features: Cannot calculate slope (zero variance)")
    
    # Calculate ST deviation (elevation - baseline)
    if not np.isnan(result['ST_elevation_mv']) and not np.isnan(baseline):
        result['ST_deviation_mv'] = float(result['ST_elevation_mv'] - baseline)
    
    return result


def extract_st_segment_features_from_dict(
    signal: np.ndarray,
    all_peak_series: Dict[str, list],
    cycle_idx: int,
    sampling_rate: float,
    j_point_offset_ms: float = 60.0,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Extract ST segment features from peak series dictionary.
    
    Parameters
    ----------
    signal : np.ndarray
        Detrended ECG signal for the cycle.
    all_peak_series : Dict[str, list]
        Dictionary of component edge indices across all cycles.
    cycle_idx : int
        Index of the current cycle.
    sampling_rate : float
        Sampling rate in Hz.
    j_point_offset_ms : float
        Offset from J point to measure ST elevation, in milliseconds.
    verbose : bool
        Whether to print debug information.
    
    Returns
    -------
    Dict[str, float]
        Dictionary with ST segment features.
    """
    # Get indices for current cycle
    s_ri_idx = None
    t_le_idx = None
    p_ri_idx = None
    q_le_idx = None
    
    if 'S_ri_idx' in all_peak_series and cycle_idx < len(all_peak_series['S_ri_idx']):
        s_ri_val = all_peak_series['S_ri_idx'][cycle_idx]
        if not np.isnan(s_ri_val):
            s_ri_idx = int(s_ri_val)
    
    if 'T_le_idx' in all_peak_series and cycle_idx < len(all_peak_series['T_le_idx']):
        t_le_val = all_peak_series['T_le_idx'][cycle_idx]
        if not np.isnan(t_le_val):
            t_le_idx = int(t_le_val)
    
    if 'P_ri_idx' in all_peak_series and cycle_idx < len(all_peak_series['P_ri_idx']):
        p_ri_val = all_peak_series['P_ri_idx'][cycle_idx]
        if not np.isnan(p_ri_val):
            p_ri_idx = int(p_ri_val)
    
    if 'Q_le_idx' in all_peak_series and cycle_idx < len(all_peak_series['Q_le_idx']):
        q_le_val = all_peak_series['Q_le_idx'][cycle_idx]
        if not np.isnan(q_le_val):
            q_le_idx = int(q_le_val)
    
    return extract_st_segment_features(
        signal=signal,
        s_ri_idx=s_ri_idx,
        t_le_idx=t_le_idx,
        p_ri_idx=p_ri_idx,
        q_le_idx=q_le_idx,
        sampling_rate=sampling_rate,
        j_point_offset_ms=j_point_offset_ms,
        verbose=verbose,
    )


