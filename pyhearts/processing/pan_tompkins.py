"""
Pan-Tompkins algorithm for QRS detection.

This module implements the classic Pan-Tompkins algorithm for R-peak detection
as described in the original publication, using a standard preprocessing chain.
"""

import numpy as np
from scipy.signal import butter, filtfilt, lfilter
from typing import Optional


def pan_tompkins_qrs_detection(
    ecg: np.ndarray,
    sampling_rate: float,
    min_refrac_ms: float = 200.0,
) -> np.ndarray:
    """
    Pan-Tompkins algorithm for QRS complex detection.
    
    The algorithm consists of:
    1. Bandpass filtering (5-15 Hz)
    2. Derivative filtering
    3. Squaring
    4. Moving window integration
    5. Adaptive thresholding
    
    Parameters
    ----------
    ecg : np.ndarray
        1D ECG signal.
    sampling_rate : float
        Sampling rate in Hz.
    min_refrac_ms : float
        Minimum refractory period in ms (default 200ms).
    
    Returns
    -------
    np.ndarray
        Indices of detected R-peaks.
    """
    ecg = np.asarray(ecg, dtype=float)
    if ecg.ndim != 1 or ecg.size == 0:
        raise ValueError("`ecg` must be a non-empty 1D array.")
    if sampling_rate <= 0:
        raise ValueError("`sampling_rate` must be > 0.")
    
    # Step 1: Bandpass filter (5-15 Hz) for QRS enhancement
    nyq = sampling_rate / 2.0
    low = 5.0 / nyq
    high = 15.0 / nyq
    
    # Ensure valid cutoffs
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    b, a = butter(2, [low, high], btype='band')
    filtered = filtfilt(b, a, ecg)
    
    # Step 2: Derivative filter (5-point derivative)
    # y[n] = (-x[n-2] - x[n-1] + x[n+1] + x[n+2]) / 8
    derivative = np.zeros_like(filtered)
    derivative[2:-2] = (-filtered[:-4] - filtered[1:-3] + filtered[3:-1] + filtered[4:]) / 8.0
    # Handle edges
    if len(derivative) > 2:
        derivative[0] = (filtered[1] - filtered[0]) / 2.0
        derivative[1] = (filtered[2] - filtered[0]) / 4.0
        derivative[-2] = (filtered[-1] - filtered[-3]) / 4.0
        derivative[-1] = (filtered[-1] - filtered[-2]) / 2.0
    
    # Step 3: Squaring (emphasizes large derivatives)
    squared = derivative ** 2
    
    # Step 4: Moving window integration
    # Window size: ~150ms (typical QRS width)
    window_ms = 150.0
    window_samples = int(round(window_ms * sampling_rate / 1000.0))
    window_samples = max(1, window_samples)
    
    # Use convolution for moving average
    window = np.ones(window_samples) / window_samples
    integrated = np.convolve(squared, window, mode='same')
    
    # Step 5: Adaptive thresholding
    # Find peaks using adaptive thresholds
    min_distance = int(round(min_refrac_ms * sampling_rate / 1000.0))
    min_distance = max(1, min_distance)
    
    # Initial threshold: 50% of maximum
    threshold = np.max(integrated) * 0.5
    signal_peak = threshold
    noise_peak = threshold * 0.5
    
    # Find peaks above threshold
    peaks = []
    i = 0
    while i < len(integrated) - min_distance:
        if integrated[i] > threshold:
            # Find local maximum in the region
            search_end = min(i + min_distance, len(integrated))
            local_max_idx = i + np.argmax(integrated[i:search_end])
            peaks.append(local_max_idx)
            
            # Update thresholds
            signal_peak = 0.125 * integrated[local_max_idx] + 0.875 * signal_peak
            threshold = noise_peak + 0.25 * (signal_peak - noise_peak)
            
            i = search_end
        else:
            # Update noise peak
            if integrated[i] > 0:
                noise_peak = 0.125 * integrated[i] + 0.875 * noise_peak
            i += 1
    
    # Convert integrated signal peaks back to original signal peaks
    # Find actual R-peaks near the integrated peaks
    r_peaks = []
    search_window = int(round(50 * sampling_rate / 1000.0))  # ±50ms search window
    
    for peak_idx in peaks:
        # Search for actual R-peak in original filtered signal
        start = max(0, peak_idx - search_window)
        end = min(len(filtered), peak_idx + search_window)
        
        if end > start:
            # Find maximum absolute value (handles both positive and negative R-peaks)
            segment = filtered[start:end]
            local_peak = start + np.argmax(np.abs(segment))
            r_peaks.append(local_peak)
    
    # Remove duplicates and sort
    r_peaks = np.unique(np.array(r_peaks, dtype=int))
    
    # Apply refractory period
    if len(r_peaks) > 1:
        filtered_peaks = [r_peaks[0]]
        for i in range(1, len(r_peaks)):
            if r_peaks[i] - filtered_peaks[-1] >= min_distance:
                filtered_peaks.append(r_peaks[i])
        r_peaks = np.array(filtered_peaks, dtype=int)
    
    return r_peaks


def pan_tompkins_r_peak_detection(
    ecg: np.ndarray,
    sampling_rate: float,
    cfg,
    plot: bool = False,
    raw_ecg: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Pan-Tompkins R-peak detection with PyHEARTS integration.
    
    This is a wrapper that integrates Pan-Tompkins algorithm into PyHEARTS
    workflow, handling signal preprocessing and polarity detection.
    
    Parameters
    ----------
    ecg : np.ndarray
        1D ECG signal (typically preprocessed/filtered).
    sampling_rate : float
        Sampling rate in Hz.
    cfg : ProcessCycleConfig
        Configuration with detection parameters.
    plot : bool, default False
        Whether to plot detected peaks.
    raw_ecg : np.ndarray, optional
        Raw (unfiltered) ECG signal for polarity detection.
    
    Returns
    -------
    np.ndarray
        Indices of detected R-peaks.
    """
    from pyhearts.processing.rpeak import _bandpass_filter, _detect_signal_polarity
    
    ecg = np.asarray(ecg, dtype=float)
    
    # Optional preprocessing (same as prominence-based method)
    if cfg.rpeak_preprocess:
        ecg_filtered = _bandpass_filter(
            ecg,
            sampling_rate,
            highpass_hz=cfg.rpeak_highpass_hz,
            lowpass_hz=cfg.rpeak_lowpass_hz,
            order=cfg.rpeak_filter_order,
            notch_hz=cfg.rpeak_notch_hz,
        )
    else:
        ecg_filtered = ecg
    
    # Detect signal polarity
    polarity_signal = raw_ecg if raw_ecg is not None else ecg_filtered
    is_inverted = _detect_signal_polarity(
        polarity_signal,
        sampling_rate,
        min_refrac_ms=cfg.rpeak_min_refrac_ms
    )
    
    # Apply Pan-Tompkins algorithm
    r_peaks = pan_tompkins_qrs_detection(
        ecg_filtered,
        sampling_rate,
        min_refrac_ms=cfg.rpeak_min_refrac_ms,
    )
    
    # Handle inverted signals
    if is_inverted:
        # For inverted signals, find negative peaks (nadirs)
        # We can use the same algorithm on negated signal
        r_peaks_inv = pan_tompkins_qrs_detection(
            -ecg_filtered,
            sampling_rate,
            min_refrac_ms=cfg.rpeak_min_refrac_ms,
        )
        # Use whichever method finds more peaks (more reliable)
        if len(r_peaks_inv) > len(r_peaks):
            r_peaks = r_peaks_inv
    
    # Apply BPM bounds filtering
    if len(r_peaks) > 1:
        rr_intervals = np.diff(r_peaks) / sampling_rate * 1000.0  # ms
        bpm = 60000.0 / rr_intervals
        
        lo_bpm, hi_bpm = cfg.rpeak_bpm_bounds
        valid = np.ones(len(r_peaks), dtype=bool)
        
        for i in range(len(rr_intervals)):
            if bpm[i] < lo_bpm or bpm[i] > hi_bpm:
                # Mark the later peak as invalid
                if i + 1 < len(valid):
                    valid[i + 1] = False
        
        r_peaks = r_peaks[valid]
    
    return r_peaks


