"""
Signal quality assessment for ECG signals.

Provides functions to assess signal quality and determine if a signal
is suitable for analysis.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from typing import Tuple, Dict


def assess_signal_quality(
    ecg: np.ndarray,
    sampling_rate: float,
    min_snr_db: float = 15.0,
    min_amplitude_range_mv: float = 0.3,
    max_baseline_wander_mv: float = 0.2,
) -> Tuple[bool, Dict[str, float], str]:
    """
    Assess ECG signal quality and determine if it's suitable for analysis.
    
    Parameters
    ----------
    ecg : np.ndarray
        ECG signal array (in mV).
    sampling_rate : float
        Sampling rate in Hz.
    min_snr_db : float, default 15.0
        Minimum signal-to-noise ratio in dB for acceptable quality.
    min_amplitude_range_mv : float, default 0.3
        Minimum signal amplitude range in mV (peak-to-peak).
    max_baseline_wander_mv : float, default 0.2
        Maximum acceptable baseline wander standard deviation in mV.
    
    Returns
    -------
    is_acceptable : bool
        True if signal quality is acceptable for analysis.
    metrics : dict
        Dictionary of quality metrics:
        - snr_db: Signal-to-noise ratio in dB
        - amplitude_range_mv: Peak-to-peak amplitude in mV
        - baseline_wander_std_mv: Baseline wander standard deviation in mV
        - signal_std_mv: Signal standard deviation in mV
    reason : str
        Reason for rejection if is_acceptable is False, empty string otherwise.
    """
    if ecg.size < 100:
        return False, {}, "Signal too short (< 100 samples)"
    
    metrics = {}
    
    # Calculate SNR
    nyquist = sampling_rate / 2
    if nyquist > 40:
        # High-frequency noise estimate
        b, a = signal.butter(4, 40 / nyquist, btype='high')
        high_freq = signal.filtfilt(b, a, ecg)
        noise_estimate = np.std(high_freq)
    else:
        # For low sampling rates, use overall signal variation as noise proxy
        noise_estimate = np.std(np.diff(ecg))
    
    signal_power = np.std(ecg)
    snr_db = 20 * np.log10(signal_power / noise_estimate) if noise_estimate > 1e-9 else np.inf
    metrics['snr_db'] = snr_db
    
    # Calculate amplitude range
    amplitude_range = np.max(ecg) - np.min(ecg)
    metrics['amplitude_range_mv'] = amplitude_range
    
    # Calculate baseline wander
    if nyquist > 0.5:
        b_low, a_low = signal.butter(4, 0.5 / nyquist, btype='low')
        baseline = signal.filtfilt(b_low, a_low, ecg)
        baseline_wander_std = np.std(baseline)
    else:
        baseline_wander_std = 0.0
    metrics['baseline_wander_std_mv'] = baseline_wander_std
    
    # Signal standard deviation
    metrics['signal_std_mv'] = signal_power
    
    # Check quality criteria
    reasons = []
    
    if not np.isfinite(snr_db) or snr_db < min_snr_db:
        reasons.append(f"Low SNR ({snr_db:.1f} dB < {min_snr_db} dB)")
    
    if amplitude_range < min_amplitude_range_mv:
        reasons.append(f"Low amplitude ({amplitude_range:.3f} mV < {min_amplitude_range_mv} mV)")
    
    if baseline_wander_std > max_baseline_wander_mv:
        reasons.append(f"High baseline wander ({baseline_wander_std:.3f} mV > {max_baseline_wander_mv} mV)")
    
    is_acceptable = len(reasons) == 0
    reason = "; ".join(reasons) if reasons else ""
    
    return is_acceptable, metrics, reason

