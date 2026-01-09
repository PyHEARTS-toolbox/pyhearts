"""
QRS removal using sigmoid replacement.

This module implements QRS complex removal by replacing QRS regions
with sigmoid functions, leaving only P and T waves for easier detection.
This preprocessing step is essential for accurate T wave detection because
the high-amplitude, rapid deflections of the QRS complex can create artifacts
in derivative-based detection algorithms, particularly when T waves have low
amplitude or occur in close proximity to the QRS complex. The sigmoid replacement
preserves overall signal morphology while eliminating sharp transitions and
high-frequency content that would otherwise interfere with T wave localization.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple, Optional


def remove_qrs_sigmoid(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    q_onsets: Optional[np.ndarray] = None,
    s_offsets: Optional[np.ndarray] = None,
    sampling_rate: float = 250.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove QRS complexes using sigmoid replacement.
    
    Replaces QRS regions with smooth sigmoid functions that connect
    the signal before and after the QRS complex. The sigmoid function
    is constructed using a logistic curve scaled to span the QRS region,
    with endpoints matched to the mean signal values in windows immediately
    before and after the QRS region. This approach eliminates the sharp
    transitions and high-frequency content of the QRS complex that would
    otherwise contaminate derivative-based T wave detection, while preserving
    the overall signal baseline and morphology.
    
    Parameters
    ----------
    signal : np.ndarray
        Input ECG signal.
    r_peaks : np.ndarray
        Array of R-peak indices.
    q_onsets : np.ndarray, optional
        Array of Q-wave onset indices (if available).
    s_offsets : np.ndarray, optional
        Array of S-wave offset indices (if available).
    sampling_rate : float
        Sampling rate in Hz.
    
    Returns
    -------
    replaced_signal : np.ndarray
        Signal with QRS complexes replaced by sigmoid functions.
    intervals : np.ndarray
        Array of shape (n_beats, 2) with [start_idx, end_idx] for each replaced interval.
    """
    signal = np.asarray(signal, dtype=float)
    replaced_signal = signal.copy()
    intervals = []
    
    if len(r_peaks) == 0:
        return replaced_signal, np.array([], dtype=int).reshape(0, 2)
    
    # Remove baseline first (two-stage approach)
    # Stage 1: 750ms window
    if len(signal) > 100:
        baseline_window1_ms = 750
        baseline_window1 = int(round(baseline_window1_ms * sampling_rate / 1000.0))
        baseline_window1 = min(baseline_window1, len(signal) // 4)
        if baseline_window1 % 2 == 0:
            baseline_window1 += 1
        if baseline_window1 >= 5:
            try:
                baseline1 = savgol_filter(signal, baseline_window1, 3, mode='nearest')
                signal = signal - baseline1
            except:
                pass
    
    # Stage 2: 2000ms window
    if len(signal) > 100:
        baseline_window2_ms = 2000
        baseline_window2 = int(round(baseline_window2_ms * sampling_rate / 1000.0))
        baseline_window2 = min(baseline_window2, len(signal) // 2)
        if baseline_window2 % 2 == 0:
            baseline_window2 += 1
        if baseline_window2 >= 5:
            try:
                baseline2 = savgol_filter(signal, baseline_window2, 3, mode='nearest')
                signal = signal - baseline2
            except:
                pass
    
    for i, r_peak in enumerate(r_peaks):
        r_peak = int(r_peak)
        
        # Estimate QRS width
        if i == 0:
            # First beat: use 1/3 of RR interval to next R
            if len(r_peaks) > 1:
                rr_interval = r_peaks[1] - r_peak
            else:
                rr_interval = int(round(0.8 * sampling_rate))  # Default 800ms
            remove_left = int(round(rr_interval / 3))
        else:
            # Use 1/3 of previous RR interval
            rr_interval = r_peak - r_peaks[i - 1]
            remove_left = int(round(rr_interval / 3))
        
        # Right side: use S offset if available, otherwise default
        if s_offsets is not None and i < len(s_offsets) and not np.isnan(s_offsets[i]):
            remove_right = int(round(s_offsets[i] - r_peak + 50e-3 * sampling_rate))
        else:
            remove_right = int(round(80e-3 * sampling_rate))  # Default 80ms
        
        # Clamp to signal bounds
        start_idx = max(0, r_peak - remove_left)
        end_idx = min(len(signal), r_peak + remove_right)
        
        if start_idx >= end_idx:
            continue
        
        # Get QRS segment
        qrs_segment = replaced_signal[start_idx:end_idx]
        
        # Parameters for sigmoid replacement
        n1 = max(1, int(round(10e-3 * sampling_rate)))  # 10ms before
        n2 = max(1, int(round(10e-3 * sampling_rate)))  # 10ms after
        
        # Create sigmoid function
        segment_length = end_idx - start_idx
        xc = np.linspace(-6, 6, segment_length)
        c = 1.0 / (1.0 + np.exp(-xc))  # Sigmoid
        
        # Get baseline values before and after QRS
        if start_idx >= n1:
            y1 = np.mean(replaced_signal[start_idx - n1:start_idx])
        else:
            y1 = np.mean(replaced_signal[:start_idx]) if start_idx > 0 else replaced_signal[0]
        
        if end_idx + n2 <= len(replaced_signal):
            y2 = np.mean(replaced_signal[end_idx:end_idx + n2])
        else:
            y2 = np.mean(replaced_signal[end_idx:]) if end_idx < len(replaced_signal) else replaced_signal[-1]
        
        # Replace QRS with sigmoid
        replaced_signal[start_idx:end_idx] = (y2 - y1) * c + y1
        
        intervals.append([start_idx, end_idx])
    
    return replaced_signal, np.array(intervals, dtype=int)


def remove_baseline_wander(
    signal: np.ndarray,
    sampling_rate: float,
    window1_ms: float = 750.0,
    window2_ms: float = 2000.0,
) -> np.ndarray:
    """
    Remove baseline wander using two-stage approach.
    
    Parameters
    ----------
    signal : np.ndarray
        Input ECG signal.
    sampling_rate : float
        Sampling rate in Hz.
    window1_ms : float
        First stage window size in ms (default 750ms).
    window2_ms : float
        Second stage window size in ms (default 2000ms).
    
    Returns
    -------
    np.ndarray
        Signal with baseline removed.
    """
    signal = np.asarray(signal, dtype=float)
    
    # Stage 1: 750ms window
    window1 = int(round(window1_ms * sampling_rate / 1000.0))
    if window1 >= 5 and len(signal) > window1:
        if window1 % 2 == 0:
            window1 += 1
        try:
            baseline1 = savgol_filter(signal, window1, 3, mode='nearest')
            signal = signal - baseline1
        except:
            pass
    
    # Stage 2: 2000ms window
    window2 = int(round(window2_ms * sampling_rate / 1000.0))
    if window2 >= 5 and len(signal) > window2:
        if window2 % 2 == 0:
            window2 += 1
        try:
            baseline2 = savgol_filter(signal, window2, 3, mode='nearest')
            signal = signal - baseline2
        except:
            pass
    
    return signal

