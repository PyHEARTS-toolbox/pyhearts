"""
Signal quality assessment for ECG signals.

Provides functions to assess signal quality and determine if a signal
is suitable for analysis.
"""

from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np
from scipy import signal


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
    ecg = np.asarray(ecg, dtype=float)

    if ecg.size < 100:
        return False, {}, "Signal too short (< 100 samples)"
    
    metrics = {}
    
    # Calculate SNR
    nyquist = sampling_rate / 2
    if nyquist > 40:
        # High-frequency noise estimate
        try:
            # Use SOS+fs for numerical stability.
            sos = signal.butter(4, 40.0, btype="highpass", fs=sampling_rate, output="sos")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                high_freq = signal.sosfiltfilt(sos, ecg)
            noise_estimate = float(np.std(high_freq))
        except Exception:
            # Fallback: use diff as noise proxy
            noise_estimate = float(np.std(np.diff(ecg)))
    else:
        # For low sampling rates, use overall signal variation as noise proxy
        noise_estimate = float(np.std(np.diff(ecg)))
    
    signal_power = float(np.std(ecg))
    snr_db = 20 * np.log10(signal_power / noise_estimate) if noise_estimate > 1e-9 else np.inf
    metrics['snr_db'] = snr_db
    
    # Calculate amplitude range
    amplitude_range = float(np.max(ecg) - np.min(ecg))
    metrics['amplitude_range_mv'] = amplitude_range
    
    # Calculate baseline wander
    if nyquist > 0.5:
        try:
            # Very low cutoffs can be numerically unstable when expressed as normalized
            # frequency at high sampling rates; use SOS+fs.
            sos_low = signal.butter(4, 0.5, btype="lowpass", fs=sampling_rate, output="sos")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                baseline = signal.sosfiltfilt(sos_low, ecg)
            baseline_wander_std = float(np.std(baseline))
        except Exception:
            # Fallback: estimate baseline via moving-average baseline (2s window).
            win = int(max(3, round(2.0 * sampling_rate)))
            if win % 2 == 0:
                win += 1
            kernel = np.ones(win, dtype=float) / float(win)
            baseline = np.convolve(ecg, kernel, mode="same")
            baseline_wander_std = float(np.std(baseline))
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






