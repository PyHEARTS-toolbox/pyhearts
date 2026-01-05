"""
P Wave Training Phase

Implements ECGpuwave-style training phase for P wave detection:
- Analyzes first 1-3 seconds to learn signal characteristics
- Separates P wave peaks from noise peaks
- Establishes adaptive thresholds for P wave validation
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from scipy.signal import find_peaks

__all__ = ["compute_p_training_phase_thresholds"]


def compute_p_training_phase_thresholds(
    ecg: np.ndarray,
    sampling_rate: float,
    training_start_sec: float = 1.0,
    training_end_sec: float = 3.0,
    bandpass_low_hz: float = 5.0,
    bandpass_high_hz: float = 15.0,
    bandpass_order: int = 2,
) -> Tuple[float, float]:
    """
    Compute P wave training phase thresholds (ECGPUWAVE-style).
    
    Analyzes first 1-3 seconds to learn P wave characteristics:
    - Signal peak: highest P wave peak in training window
    - Noise peak: highest peak below 75% of signal peak
    
    This establishes adaptive thresholds for P wave validation, making the
    detector more sensitive to small P waves while maintaining specificity.
    
    Parameters
    ----------
    ecg : np.ndarray
        ECG signal (detrended, in mV).
    sampling_rate : float
        Sampling rate in Hz.
    training_start_sec : float, default 1.0
        Start of training window in seconds.
    training_end_sec : float, default 3.0
        End of training window in seconds.
    bandpass_low_hz : float, default 5.0
        Low cutoff frequency for P wave band-pass filter (Hz).
    bandpass_high_hz : float, default 15.0
        High cutoff frequency for P wave band-pass filter (Hz).
    bandpass_order : int, default 2
        Filter order for band-pass filter.
    
    Returns
    -------
    tuple[float, float]
        (signal_peak, noise_peak) amplitudes in mV.
        These represent the trained thresholds for P wave validation.
    """
    training_start = int(training_start_sec * sampling_rate)
    training_end = int(training_end_sec * sampling_rate)
    training_end = min(training_end, len(ecg))
    
    if training_end <= training_start:
        # Fallback: use first 3 seconds
        training_end = min(int(3.0 * sampling_rate), len(ecg))
        training_start = 0
    
    if training_end <= training_start or training_end - training_start < 10:
        # Too short for training, use conservative defaults
        signal_peak = 0.1  # Conservative default: 0.1 mV
        noise_peak = 0.05  # 50% of signal peak
        return signal_peak, noise_peak
    
    training_segment = ecg[training_start:training_end]
    
    # Apply band-pass filter to enhance P waves (same as in P detection)
    try:
        from pyhearts.processing.processcycle import bandpass_filter_pwave
        training_filtered = bandpass_filter_pwave(
            training_segment,
            sampling_rate,
            lowcut=bandpass_low_hz,
            highcut=bandpass_high_hz,
            order=bandpass_order,
        )
    except Exception:
        # Fallback to generic bandpass if pwave-specific function not available
        from pyhearts.processing.preprocess import bandpass_filter
        training_filtered = bandpass_filter(
            training_segment,
            sampling_rate,
            highpass_cutoff=bandpass_low_hz,
            lowpass_cutoff=bandpass_high_hz,
            filter_order=bandpass_order,
        )
    
    if len(training_filtered) < 10:
        signal_peak = float(np.max(np.abs(training_filtered))) if len(training_filtered) > 0 else 0.1
        noise_peak = signal_peak * 0.5
        return signal_peak, noise_peak
    
    # Find peaks in training window using prominence (more robust than amplitude)
    # P waves should be spaced at least 100ms apart (physiological minimum)
    distance_samples = max(1, int(round(0.1 * sampling_rate)))  # 100ms minimum distance
    min_prominence = np.std(training_filtered) * 0.3  # Adaptive prominence threshold
    
    peaks, properties = find_peaks(
        np.abs(training_filtered),  # Use absolute value to detect both positive and negative P waves
        distance=distance_samples,
        prominence=min_prominence,
    )
    
    if len(peaks) == 0:
        # No peaks found, use amplitude-based estimate
        signal_peak = float(np.max(np.abs(training_filtered)))
        noise_peak = signal_peak * 0.5
        return signal_peak, noise_peak
    
    # Get peak prominences (not amplitudes) - this is what find_peaks uses
    peak_prominences = properties['prominences'] if 'prominences' in properties else [np.abs(training_filtered[p]) for p in peaks]
    
    # Signal peak: highest prominence (likely a P wave)
    signal_peak = float(np.max(peak_prominences))
    
    # Noise peak: highest prominence below 75% of signal peak
    # This separates P waves from noise/artifacts
    signal_75 = 0.75 * signal_peak
    noise_prominences = [prom for prom in peak_prominences if prom < signal_75]
    noise_peak = float(np.max(noise_prominences)) if len(noise_prominences) > 0 else signal_peak * 0.5
    
    return signal_peak, noise_peak

