import numpy as np
import pyhearts as ph
from scipy.signal import find_peaks, butter, filtfilt, iirnotch
from typing import Optional, Union, Literal
from pyhearts.config import ProcessCycleConfig


def _bandpass_filter(
    ecg: np.ndarray,
    sampling_rate: float,
    highpass_hz: float,
    lowpass_hz: float,
    order: int = 2,
    notch_hz: Optional[float] = None,
) -> np.ndarray:
    """
    Apply bandpass (and optional notch) filter for R-peak detection.
    
    Enhanced for low-SNR signals with better noise reduction.
    """
    filtered = ecg.copy()
    
    # Highpass to remove baseline wander
    if highpass_hz > 0:
        nyq = sampling_rate / 2
        # Ensure cutoff is valid
        hp_norm = min(highpass_hz / nyq, 0.99)
        b, a = butter(order, hp_norm, btype='high')
        filtered = filtfilt(b, a, filtered)
    
    # Lowpass to remove high-frequency noise
    # For low-SNR signals, use slightly lower cutoff to reduce noise
    if lowpass_hz > 0:
        nyq = sampling_rate / 2
        lp_norm = min(lowpass_hz / nyq, 0.99)
        b, a = butter(order, lp_norm, btype='low')
        filtered = filtfilt(b, a, filtered)
    
    # Optional notch filter for power line interference
    # Use higher Q factor for better noise rejection in low-SNR signals
    if notch_hz is not None and notch_hz > 0:
        q = 35.0  # Increased from 30.0 for better noise rejection
        b, a = iirnotch(notch_hz, q, sampling_rate)
        filtered = filtfilt(b, a, filtered)
    
    return filtered


def _score_peak_set_qrs_characteristics(
    ecg: np.ndarray,
    peaks: np.ndarray,
    sampling_rate: float,
) -> float:
    """
    Score a set of peaks based on QRS-like characteristics.
    
    R peaks have distinctive features:
    - Sharp slopes (high derivative)
    - Narrow width (< 120ms typically)
    - High signal-to-noise ratio
    
    Parameters
    ----------
    ecg : np.ndarray
        ECG signal.
    peaks : np.ndarray
        Array of peak indices.
    sampling_rate : float
        Sampling rate in Hz.
    
    Returns
    -------
    float
        QRS-like score (higher = more R-peak-like).
    """
    if len(peaks) < 3:
        return 0.0
    
    # Calculate derivative for slope analysis
    derivative = np.gradient(ecg)
    
    # QRS width is typically < 120ms
    max_qrs_width_ms = 120.0
    max_qrs_width_samples = int(max_qrs_width_ms * sampling_rate / 1000.0)
    
    scores = []
    for peak_idx in peaks[:min(10, len(peaks))]:  # Sample up to 10 peaks for efficiency
        peak_idx = int(peak_idx)
        if peak_idx < 0 or peak_idx >= len(ecg):
            continue
        
        # 1. Slope score: R peaks have sharp slopes
        window_ms = 80.0
        window_samples = int(window_ms * sampling_rate / 1000.0)
        start_idx = max(0, peak_idx - window_samples)
        end_idx = min(len(derivative), peak_idx + window_samples)
        
        if end_idx > start_idx:
            max_slope = np.max(np.abs(derivative[start_idx:end_idx]))
            # Normalize slope (typical QRS slopes are > 0.1 mV/sample at 250 Hz)
            slope_score = min(1.0, max_slope / 0.1)
        else:
            slope_score = 0.0
        
        # 2. Width score: R peaks are narrow
        # Find width at 50% of peak height
        peak_val = ecg[peak_idx]
        baseline = np.median(ecg[max(0, peak_idx - max_qrs_width_samples):min(len(ecg), peak_idx + max_qrs_width_samples)])
        half_height = baseline + (peak_val - baseline) * 0.5
        
        # Find width
        left_idx = peak_idx
        right_idx = peak_idx
        for i in range(peak_idx, max(0, peak_idx - max_qrs_width_samples), -1):
            if ecg[i] <= half_height:
                left_idx = i
                break
        for i in range(peak_idx, min(len(ecg), peak_idx + max_qrs_width_samples)):
            if ecg[i] <= half_height:
                right_idx = i
                break
        
        width_samples = right_idx - left_idx
        width_ms = width_samples / sampling_rate * 1000.0
        # Narrow peaks (< 120ms) score higher
        width_score = max(0.0, 1.0 - (width_ms / max_qrs_width_ms))
        
        # 3. Prominence score: R peaks stand out from baseline
        prominence = abs(peak_val - baseline)
        # Normalize (typical R peaks have prominence > 0.3 mV)
        prominence_score = min(1.0, prominence / 0.3)
        
        # Combined score (weighted average)
        combined_score = (slope_score * 0.4 + width_score * 0.3 + prominence_score * 0.3)
        scores.append(combined_score)
    
    return np.median(scores) if len(scores) > 0 else 0.0


def _detect_signal_polarity(
    ecg: np.ndarray,
    sampling_rate: float,
    min_refrac_ms: float = 100.0,
) -> bool:
    """
    Detect if the ECG signal has inverted QRS complexes using multi-feature scoring.
    
    This robust approach evaluates both positive and negative peak sets using:
    1. QRS-like characteristics (slope, width, sharpness)
    2. Peak regularity (RR interval consistency)
    3. Signal quality (prominence, SNR)
    
    For inverted signals, the R-peaks are negative-going (nadirs) rather than
    positive peaks. This function uses a comprehensive scoring system to determine
    which peak set is more R-peak-like, rather than just comparing amplitudes.
    
    Parameters
    ----------
    ecg : np.ndarray
        Filtered ECG signal.
    sampling_rate : float
        Sampling rate in Hz.
    min_refrac_ms : float
        Minimum refractory period in ms for initial peak detection.
    
    Returns
    -------
    bool
        True if signal is inverted (R-peaks are negative), False otherwise.
    """
    if ecg.size < 100:  # Need enough samples for reliable detection
        return False
    
    # Use a simple prominence threshold based on signal variance
    mad = np.median(np.abs(ecg - np.median(ecg)))
    robust_std = 1.4826 * mad if mad > 1e-9 else float(np.std(ecg))
    prominence_threshold = 2.0 * robust_std
    
    # Minimum distance between peaks
    distance_samples = max(1, int(round(min_refrac_ms * sampling_rate / 1000.0)))
    
    # Find positive peaks (maxima)
    pos_peaks, pos_props = find_peaks(
        ecg, 
        distance=distance_samples, 
        prominence=prominence_threshold
    )
    
    # Find negative peaks (minima) by negating the signal
    neg_peaks, neg_props = find_peaks(
        -ecg, 
        distance=distance_samples, 
        prominence=prominence_threshold
    )
    
    # If we don't find enough peaks, can't determine polarity reliably
    if pos_peaks.size < 3 and neg_peaks.size < 3:
        # Fallback: check actual peak values, not just signal bias
        # Signal bias can be misleading (e.g., baseline wander)
        # Instead, check if we can find any clear peaks with lower threshold
        relaxed_threshold = prominence_threshold * 0.6  # More relaxed
        pos_peaks_relaxed, pos_props_relaxed = find_peaks(
            ecg, distance=distance_samples, prominence=relaxed_threshold
        )
        neg_peaks_relaxed, neg_props_relaxed = find_peaks(
            -ecg, distance=distance_samples, prominence=relaxed_threshold
        )
        
        if pos_peaks_relaxed.size >= 2 or neg_peaks_relaxed.size >= 2:
            # Compare peak amplitudes (use actual signal values, not absolute)
            if pos_peaks_relaxed.size > 0:
                pos_values = ecg[pos_peaks_relaxed]
                median_pos_val = np.median(pos_values)
                max_pos_val = np.max(pos_values)
            else:
                median_pos_val = 0
                max_pos_val = 0
            
            if neg_peaks_relaxed.size > 0:
                neg_values = ecg[neg_peaks_relaxed]  # Already negative
                median_neg_val = np.median(neg_values)
                min_neg_val = np.min(neg_values)
            else:
                median_neg_val = 0
                min_neg_val = 0
            
            # Check if positive peaks are clearly present and substantial
            # R-peaks should be the largest deflections
            if max_pos_val > 0.5 and max_pos_val > abs(min_neg_val) * 0.8:
                # Positive peaks are substantial - signal is normal
                return False
            elif abs(min_neg_val) > 0.5 and abs(min_neg_val) > max_pos_val * 1.2:
                # Negative peaks are clearly dominant - signal is inverted
                return True
            else:
                # Ambiguous - assume normal (conservative approach)
                return False
        else:
            # Can't determine - assume normal (conservative approach)
            # This prevents false inversion detection
            return False
    
    # Multi-feature scoring approach: evaluate which peak set is more R-peak-like
    if pos_peaks.size >= 3 and neg_peaks.size >= 3:
            baseline = np.median(ecg)
            
            # Score 1: QRS-like characteristics (slope, width, sharpness)
            # For negative peaks, we need to work with the inverted signal
            pos_qrs_score = _score_peak_set_qrs_characteristics(ecg, pos_peaks, sampling_rate)
            neg_qrs_score = _score_peak_set_qrs_characteristics(-ecg, neg_peaks, sampling_rate)
            
            # Score 2: Peak regularity and physiological validity
            # R peaks should have more consistent intervals than T waves
            # Also, R peaks should have intervals in the physiological range (60-120 bpm = 500-1000 ms)
            pos_intervals = np.diff(np.sort(pos_peaks)) / sampling_rate * 1000  # in ms
            neg_intervals = np.diff(np.sort(neg_peaks)) / sampling_rate * 1000  # in ms
            
            pos_interval_cv = np.std(pos_intervals) / (np.mean(pos_intervals) + 1e-9)
            neg_interval_cv = np.std(neg_intervals) / (np.mean(neg_intervals) + 1e-9)
            
            # Check if intervals are in physiological R-R range (500-1000 ms = 60-120 bpm)
            pos_median_interval = np.median(pos_intervals)
            neg_median_interval = np.median(neg_intervals)
            pos_in_physio_range = 500 <= pos_median_interval <= 1000
            neg_in_physio_range = 500 <= neg_median_interval <= 1000
            
            # Regularity score: lower CV = more regular = more R-like
            pos_regularity_score = max(0.0, 1.0 - pos_interval_cv)
            neg_regularity_score = max(0.0, 1.0 - neg_interval_cv)
            
            # Boost score if intervals are in physiological range (R peaks should be)
            if pos_in_physio_range:
                pos_regularity_score = min(1.0, pos_regularity_score + 0.2)
            if neg_in_physio_range:
                neg_regularity_score = min(1.0, neg_regularity_score + 0.2)
            
            # Score 3: Prominence/deflection from baseline
            pos_prominences = pos_props.get('prominences', None)
            neg_prominences = neg_props.get('prominences', None)
            
            if pos_prominences is None:
                pos_prominences = ecg[pos_peaks] - baseline
                pos_prominences = pos_prominences[pos_prominences > 0]
            if neg_prominences is None:
                neg_prominences = baseline - ecg[neg_peaks]
                neg_prominences = neg_prominences[neg_prominences > 0]
            
            median_pos_prom = np.median(pos_prominences) if len(pos_prominences) > 0 else 0
            median_neg_prom = np.median(neg_prominences) if len(neg_prominences) > 0 else 0
            
            # Normalize prominence scores (typical R peaks have prominence > 0.3 mV)
            pos_prom_score = min(1.0, median_pos_prom / 0.3) if median_pos_prom > 0 else 0
            neg_prom_score = min(1.0, median_neg_prom / 0.3) if median_neg_prom > 0 else 0
            
            # Combined score (weighted average)
            # QRS characteristics are most important (50%) - R peaks have distinctive QRS features
            # Regularity is less reliable (25%) - T waves can sometimes be more regular
            # Prominence is secondary (25%) - both R and T can be prominent
            pos_total_score = (pos_qrs_score * 0.5 + pos_regularity_score * 0.25 + pos_prom_score * 0.25)
            neg_total_score = (neg_qrs_score * 0.5 + neg_regularity_score * 0.25 + neg_prom_score * 0.25)
            
            # Signal is inverted if negative peaks score higher
            # Use physiological range as a strong indicator - R peaks should be in 60-120 bpm range
            # If one set is in range and the other isn't, that's a strong signal
            if neg_in_physio_range and not pos_in_physio_range:
                # Negative peaks are in physiological R-R range, positive are not - likely inverted
                is_inverted = True
            elif pos_in_physio_range and not neg_in_physio_range:
                # Positive peaks are in physiological R-R range, negative are not - likely normal
                is_inverted = False
            elif neg_qrs_score > pos_qrs_score * 1.2:
                # Negative peaks have significantly better QRS characteristics - likely inverted
                is_inverted = True
            elif pos_qrs_score > neg_qrs_score * 1.2:
                # Positive peaks have significantly better QRS characteristics - likely normal
                is_inverted = False
            elif neg_total_score > pos_total_score * 1.05:
                # Negative peaks score higher overall (lower threshold since we already checked physio range)
                is_inverted = True
            elif pos_total_score > neg_total_score * 1.05:
                # Positive peaks score higher overall
                is_inverted = False
            else:
                # Ambiguous case: use prominence as tie-breaker
                if median_neg_prom > median_pos_prom * 1.2:
                    is_inverted = True
                else:
                    is_inverted = False
            
            return is_inverted
    elif neg_peaks.size > 0 and pos_peaks.size == 0:
        # Only negative peaks found - likely inverted
        return True
    else:
        # Only positive peaks found or neither - assume normal
        return False


def _adaptive_prominence_threshold(
    ecg: np.ndarray,
    base_multiplier: float,
    sensitivity: Literal["standard", "high", "maximum"] = "standard",
) -> float:
    """
    Compute adaptive prominence threshold based on signal characteristics.
    
    Uses robust MAD-based estimation instead of pure std to handle noisy signals.
    Sensitivity modes allow trading off precision vs. recall.
    
    Parameters
    ----------
    ecg : np.ndarray
        Filtered ECG signal.
    base_multiplier : float
        Base prominence multiplier from config.
    sensitivity : {"standard", "high", "maximum"}
        Detection sensitivity level:
        - "standard": Uses base_multiplier as-is (balanced)
        - "high": Reduces threshold by 25% (higher recall, slightly lower precision)
        - "maximum": Reduces threshold by 40% (maximum recall, may include noise)
    
    Returns
    -------
    float
        Adaptive prominence threshold.
    """
    # Use MAD for robust noise estimation (less sensitive to R-peak outliers)
    median_val = np.median(ecg)
    mad = np.median(np.abs(ecg - median_val))
    robust_std = 1.4826 * mad  # MAD to std conversion for Gaussian
    
    # Fall back to regular std if MAD is degenerate
    if robust_std < 1e-9:
        robust_std = float(np.std(ecg))
    
    # Apply sensitivity adjustment
    sensitivity_factors = {
        "standard": 1.0,
        "high": 0.75,
        "maximum": 0.60,
    }
    factor = sensitivity_factors.get(sensitivity, 1.0)
    
    return base_multiplier * robust_std * factor


def _training_phase_signal_noise_separation(
    ecg: np.ndarray,
    derivative: np.ndarray,
    sampling_rate: float,
    training_start_sec: float = 1.0,
    training_end_sec: float = 3.0,
) -> tuple[float, float]:
    """
    ECGPUWAVE-style training phase: separate signal peaks from noise peaks.
    
    Analyzes first 1-3 seconds to learn signal characteristics:
    - Signal peak: highest peak in training window
    - Noise peak: highest peak below 75% of signal peak
    
    Parameters
    ----------
    ecg : np.ndarray
        ECG signal (possibly inverted).
    derivative : np.ndarray
        Derivative of ECG signal.
    sampling_rate : float
        Sampling rate in Hz.
    training_start_sec : float
        Start of training window in seconds.
    training_end_sec : float
        End of training window in seconds.
    
    Returns
    -------
    tuple[float, float]
        (signal_peak, noise_peak) amplitudes.
    """
    training_start = int(training_start_sec * sampling_rate)
    training_end = int(training_end_sec * sampling_rate)
    training_end = min(training_end, len(ecg))
    
    if training_end <= training_start:
        signal_peak = float(np.max(np.abs(ecg[:min(int(3.0 * sampling_rate), len(ecg))])))
        noise_peak = signal_peak * 0.5
        return signal_peak, noise_peak
    
    training_segment = ecg[training_start:training_end]
    
    if len(training_segment) < 10:
        signal_peak = float(np.max(np.abs(training_segment))) if len(training_segment) > 0 else 1.0
        noise_peak = signal_peak * 0.5
        return signal_peak, noise_peak
    
    # Find peaks in training window - use prominence, not amplitude
    distance_lo = max(1, int(round(0.1 * sampling_rate)))  # 100ms minimum distance
    min_prominence = np.std(training_segment) * 0.3
    peaks, properties = find_peaks(
        training_segment,
        distance=distance_lo,
        prominence=min_prominence
    )
    
    if len(peaks) == 0:
        # Fallback: use amplitude-based estimate
        signal_peak = float(np.max(np.abs(training_segment)))
        noise_peak = signal_peak * 0.5
        return signal_peak, noise_peak
    
    # Get peak prominences (not amplitudes) - this is what find_peaks uses
    peak_prominences = properties['prominences'] if 'prominences' in properties else [training_segment[p] for p in peaks]
    
    # Signal peak: highest prominence
    signal_peak = float(np.max(peak_prominences))
    
    # Noise peak: highest prominence below 75% of signal peak
    qrs75 = 0.75 * signal_peak
    noise_prominences = [prom for prom in peak_prominences if prom < qrs75]
    noise_peak = float(np.max(noise_prominences)) if len(noise_prominences) > 0 else signal_peak * 0.5
    
    return signal_peak, noise_peak


def _filter_rr_intervals(
    rr_intervals: np.ndarray,
    median_rr: float,
    lower_frac: float = 0.92,
    upper_frac: float = 1.16,
) -> np.ndarray:
    """
    Filter RR intervals to remove outliers (ECGPUWAVE-style).
    
    Only keeps RR intervals within [lower_frac * median, upper_frac * median].
    
    Parameters
    ----------
    rr_intervals : np.ndarray
        Array of RR intervals.
    median_rr : float
        Median RR interval for filtering.
    lower_frac : float
        Lower bound fraction (default: 0.92 = 92% of median).
    upper_frac : float
        Upper bound fraction (default: 1.16 = 116% of median).
    
    Returns
    -------
    np.ndarray
        Filtered RR intervals.
    """
    if len(rr_intervals) == 0:
        return rr_intervals
    
    lower_bound = lower_frac * median_rr
    upper_bound = upper_frac * median_rr
    
    mask = (rr_intervals >= lower_bound) & (rr_intervals <= upper_bound)
    return rr_intervals[mask]


def _calculate_maximum_slope(
    derivative: np.ndarray,
    peak_idx: int,
    sampling_rate: float,
    window_ms: float = 80.0,
) -> float:
    """
    Calculate maximum absolute derivative slope in window before peak.
    
    This is used for slope-based discrimination (ECGPUWAVE-style).
    
    Parameters
    ----------
    derivative : np.ndarray
        Derivative of ECG signal.
    peak_idx : int
        Peak index.
    sampling_rate : float
        Sampling rate in Hz.
    window_ms : float
        Window size in ms before peak (default: 80ms).
    
    Returns
    -------
    float
        Maximum absolute slope value.
    """
    window_samples = int(round(window_ms * sampling_rate / 1000.0))
    start_idx = max(0, peak_idx - window_samples)
    end_idx = min(len(derivative), peak_idx + 1)
    
    if end_idx <= start_idx:
        return 0.0
    
    max_slope = float(np.max(np.abs(derivative[start_idx:end_idx])))
    return max_slope


def _calculate_qrs_width(
    ecg: np.ndarray,
    peak_idx: int,
    sampling_rate: float,
    window_ms: float = 150.0,
) -> float:
    """
    Calculate QRS width (duration) around a peak.
    
    QRS width is measured as the full width at half maximum (FWHM) or
    as the distance between zero-crossings in the derivative.
    
    Parameters
    ----------
    ecg : np.ndarray
        ECG signal.
    peak_idx : int
        Peak index.
    sampling_rate : float
        Sampling rate in Hz.
    window_ms : float
        Maximum window size in ms to search (default: 150ms).
    
    Returns
    -------
    float
        QRS width in milliseconds.
    """
    window_samples = int(round(window_ms * sampling_rate / 1000.0))
    start_idx = max(0, peak_idx - window_samples // 2)
    end_idx = min(len(ecg), peak_idx + window_samples // 2)
    
    if end_idx <= start_idx:
        return 100.0  # Default reasonable width
    
    # Use absolute values for width calculation (works for both polarities)
    abs_ecg = np.abs(ecg)
    peak_val = abs_ecg[peak_idx]
    
    if peak_val < 1e-6:
        return 100.0  # Default if peak is too small
    
    # Use FWHM (Full Width at Half Maximum)
    half_max = peak_val / 2.0
    
    # Find left edge (where signal crosses half-max going outward from peak)
    left_idx = peak_idx
    for i in range(peak_idx, start_idx - 1, -1):
        if abs_ecg[i] >= half_max:
            left_idx = i
        else:
            break
    
    # Find right edge
    right_idx = peak_idx
    for i in range(peak_idx, end_idx):
        if abs_ecg[i] >= half_max:
            right_idx = i
        else:
            break
    
    qrs_width_samples = right_idx - left_idx
    
    # Ensure minimum width (at least 2 samples)
    if qrs_width_samples < 2:
        qrs_width_samples = 2
    
    qrs_width_ms = (qrs_width_samples / sampling_rate) * 1000.0
    
    # Clamp to reasonable range (10-300ms)
    qrs_width_ms = max(10.0, min(300.0, qrs_width_ms))
    
    return qrs_width_ms


def _validate_qrs_characteristics(
    peak_idx: int,
    ecg: np.ndarray,
    derivative: np.ndarray,
    sampling_rate: float,
    slope_threshold: float,
    min_slope: float = 0.01,
    max_width_ms: float = 200.0,
    min_width_ms: float = 20.0,
) -> tuple[bool, dict]:
    """
    Validate QRS characteristics for a peak.
    
    Checks slope, width, and other QRS characteristics to ensure
    the peak is a valid R peak (not a P or T wave).
    
    Uses same validation criteria for both polarities (ECGPUWave-style).
    
    Parameters
    ----------
    peak_idx : int
        Peak index to validate.
    ecg : np.ndarray
        ECG signal.
    derivative : np.ndarray
        Derivative of ECG signal.
    sampling_rate : float
        Sampling rate in Hz.
    slope_threshold : float
        Minimum slope threshold (from median QRS slope).
    min_slope : float
        Absolute minimum slope (default: 0.01).
    max_width_ms : float
        Maximum QRS width in ms (default: 200ms).
    min_width_ms : float
        Minimum QRS width in ms (default: 20ms).
    
    Returns
    -------
    tuple[bool, dict]
        (is_valid, characteristics_dict) where characteristics_dict contains
        slope, width, and other metrics.
    """
    # Calculate slope (use absolute values for consistency)
    slope = _calculate_maximum_slope(derivative, peak_idx, sampling_rate)
    
    # Calculate width (use absolute values for consistency)
    width_ms = _calculate_qrs_width(ecg, peak_idx, sampling_rate)
    
    # Validate characteristics - be more lenient for mixed-polarity signals
    is_valid = True
    reasons = []
    
    # Slope validation - use absolute slope threshold
    effective_threshold = max(min_slope, slope_threshold * 0.5)  # More lenient
    if slope < effective_threshold:
        is_valid = False
        reasons.append(f"slope_too_low ({slope:.4f} < {effective_threshold:.4f})")
    
    # Width validation - be more lenient (QRS can be 10-300ms)
    # Only reject if width is clearly out of physiological range
    if width_ms < 5.0 or width_ms > 500.0:
        is_valid = False
        reasons.append(f"width_out_of_range ({width_ms:.1f}ms, expected 5-500ms)")
    
    characteristics = {
        'slope': slope,
        'width_ms': width_ms,
        'is_valid': is_valid,
        'reasons': reasons,
    }
    
    return is_valid, characteristics


def _detect_r_peaks_single_polarity(
    ecg_for_detection: np.ndarray,
    ecg_filtered: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    sensitivity: Literal["standard", "high", "maximum"] = "standard",
) -> np.ndarray:
    """
    Detect R peaks in a single polarity (ECGPUWAVE-style multi-pass detection).
    
    This is a helper function that performs the full detection pipeline for one polarity.
    Applied to both positive and negative peaks separately, then results are merged.
    
    Key ECGPUWAVE principles:
    - Uses absolute values for threshold calculations (works with signal as-is)
    - Training phase should work on absolute signal values
    - Amplitude filtering should use absolute peak heights
    
    Parameters
    ----------
    ecg_for_detection : np.ndarray
        Signal prepared for peak detection (may be negated for inverted polarity).
    ecg_filtered : np.ndarray
        Original filtered signal (for threshold calculations).
    sampling_rate : float
        Sampling rate in Hz.
    cfg : ProcessCycleConfig
        Configuration with detection parameters.
    sensitivity : {"standard", "high", "maximum"}, default "standard"
        Detection sensitivity level.
    
    Returns
    -------
    np.ndarray
        Indices of detected R-peaks in this polarity.
    """
    # ----- Compute derivative for slope-based discrimination -----
    derivative = np.gradient(ecg_for_detection)

    # ----- ECGPUWAVE-style training phase: signal/noise separation -----
    # ECGPUWave works with signal as-is (doesn't negate)
    # For threshold calculation, we need to use the signal we're actually detecting peaks on
    # (ecg_for_detection), not the absolute values, because find_peaks uses the actual signal
    signal_peak, noise_peak = _training_phase_signal_noise_separation(
        ecg_for_detection, derivative, sampling_rate
    )
    
    # ECGPUWAVE threshold formula: noise + 0.25*(signal - noise)
    ecgpuwave_threshold = noise_peak + 0.25 * (signal_peak - noise_peak)
    
    # Use ECGPUWAVE threshold if it's reasonable, otherwise fall back to adaptive
    # Use the signal we're detecting on for adaptive threshold
    adaptive_threshold = _adaptive_prominence_threshold(
        ecg_for_detection, cfg.rpeak_prominence_multiplier, sensitivity
    )
    
    # Use the more conservative of the two thresholds
    # But ensure it's not too high - use minimum of the two if one is much larger
    prominence_threshold = max(ecgpuwave_threshold, adaptive_threshold * 0.5)
    
    # Safety check: if threshold is too high relative to signal, reduce it
    # This can happen if training phase finds very large peaks
    signal_max = float(np.max(np.abs(ecg_for_detection)))
    if prominence_threshold > signal_max * 0.8:
        prominence_threshold = signal_max * 0.3  # More reasonable threshold
    
    # ----- First pass: conservative distance -----
    distance_lo = max(1, int(round(cfg.rpeak_min_refrac_ms * sampling_rate / 1000.0)))
    peaks_lo, _ = find_peaks(ecg_for_detection, distance=distance_lo, prominence=prominence_threshold)

    # ----- Second pass: RR-adaptive distance with filtered RR intervals -----
    if peaks_lo.size < 3:
        initial_r_peaks = peaks_lo.astype(int)
        median_rr_samples = distance_lo  # fallback
    else:
        # Use filtered RR intervals (ECGPUWAVE-style)
        rr_intervals = np.diff(peaks_lo)
        filtered_rr_intervals = _filter_rr_intervals(
            rr_intervals,
            np.median(rr_intervals),
            lower_frac=0.92,
            upper_frac=1.16
        )
        
        if len(filtered_rr_intervals) > 0:
            median_rr_samples = float(np.median(filtered_rr_intervals))
        else:
            median_rr_samples = float(np.median(rr_intervals))

        min_bpm, max_bpm = cfg.rpeak_bpm_bounds
        rr_min = (60_000.0 / max_bpm) * sampling_rate / 1000.0
        rr_max = (60_000.0 / min_bpm) * sampling_rate / 1000.0
        median_rr_samples = float(np.clip(median_rr_samples, rr_min, rr_max))

        distance = max(1, int(round(cfg.rpeak_rr_frac_second_pass * median_rr_samples)))
        initial_r_peaks, _ = find_peaks(
            ecg_for_detection, distance=distance, prominence=prominence_threshold
        )
    
    # ----- QRS characteristic filtering: reject T-waves and P-waves (ECGPUWAVE-style) -----
    # Apply same validation criteria to both polarities
    if initial_r_peaks.size > 0:
        # Calculate slopes for all peaks
        peak_slopes = []
        for peak_idx in initial_r_peaks:
            slope = _calculate_maximum_slope(derivative, peak_idx, sampling_rate)
            peak_slopes.append(slope)
        peak_slopes = np.array(peak_slopes)
        
        # Estimate QRS slope from median of top 50% peaks (likely QRS)
        if len(peak_slopes) >= 2:
            sorted_slopes = np.sort(peak_slopes)
            median_qrs_slope = np.median(sorted_slopes[len(sorted_slopes)//2:])
            slope_threshold = 0.75 * median_qrs_slope
        else:
            # Fallback if not enough peaks
            median_qrs_slope = float(np.median(peak_slopes)) if len(peak_slopes) > 0 else 0.1
            slope_threshold = 0.5 * median_qrs_slope
        
        # Apply QRS characteristic validation to all peaks
        # Use more lenient criteria to avoid rejecting valid peaks
        # For now, use slope-only filtering (QRS width validation can be added later)
        # Reject peaks with slope < 0.75 * median QRS slope (likely T-waves)
        slope_mask = peak_slopes >= 0.75 * median_qrs_slope
        initial_r_peaks = initial_r_peaks[slope_mask]
        
        # TODO: Re-enable QRS width validation once width calculation is more robust
        # For now, we use slope-based filtering which is more reliable
    
    # ----- Amplitude filtering: remove P/T waves detected as R-peaks -----
    # ECGPUWave approach: Use absolute values for amplitude comparison
    # This ensures correct filtering for both positive and negative peaks
    # ECGPUWave is more lenient - it uses percentile-based filtering rather than
    # strict max-based filtering to handle mixed-polarity signals
    if initial_r_peaks.size > 0:
        peak_heights = np.abs(ecg_for_detection[initial_r_peaks])  # Use absolute values
        
        # ECGPUWave-style: Use percentile-based threshold instead of max-based
        # This is more robust for mixed-polarity signals
        if len(peak_heights) >= 3:
            # Use 60th percentile as threshold (more lenient than 50% of max)
            height_threshold = np.percentile(peak_heights, 40)  # Keep top 60%
        else:
            # Fallback to max-based if not enough peaks
            max_peak_height = np.max(peak_heights)
            height_threshold = 0.3 * max_peak_height  # Very lenient for small sets
        
        amplitude_mask = peak_heights >= height_threshold
        initial_r_peaks = initial_r_peaks[amplitude_mask]
    
    # ----- Third pass: aggressive gap-filling with backtracking (ECGPUWAVE-style) -----
    if initial_r_peaks.size >= 2:
        # Use filtered RR intervals for gap detection
        rr_intervals = np.diff(initial_r_peaks)
        filtered_rr_intervals = _filter_rr_intervals(
            rr_intervals,
            np.median(rr_intervals),
            lower_frac=0.92,
            upper_frac=1.16
        )
        
        if len(filtered_rr_intervals) > 0:
            median_rr_samples = float(np.median(filtered_rr_intervals))
        else:
            median_rr_samples = float(np.median(rr_intervals))
        
        gap_threshold = 1.8 * median_rr_samples
        large_gaps = np.where(rr_intervals > gap_threshold)[0]
        
        additional_peaks = []
        umb2 = prominence_threshold / 2.0  # Secondary threshold for backtracking
        
        for gap_idx in large_gaps:
            start_idx = initial_r_peaks[gap_idx]
            end_idx = initial_r_peaks[gap_idx + 1]
            gap_segment = ecg_for_detection[start_idx:end_idx]
            
            if gap_segment.size < 10:
                continue
            
            # Progressive threshold reduction (ECGPUWAVE-style backtracking)
            found_peak = False
            for threshold_level in range(5):  # Max 5 levels
                if threshold_level == 0:
                    search_threshold = umb2
                else:
                    search_threshold = umb2 / (2.0 ** threshold_level)
                
                gap_peaks, _ = find_peaks(
                    gap_segment,
                    distance=max(1, int(0.5 * median_rr_samples)),
                    prominence=search_threshold
                )
                gap_peaks = gap_peaks + start_idx
                
                # Check each gap peak with slope discrimination
                for gp in gap_peaks:
                    # Calculate slope
                    peak_slope = _calculate_maximum_slope(derivative, gp, sampling_rate)
                    
                    # Slope check (if we have QRS slope estimate)
                    if initial_r_peaks.size > 0:
                        # Estimate QRS slope from existing peaks
                        existing_slopes = []
                        for ep in initial_r_peaks[:min(10, len(initial_r_peaks))]:
                            existing_slopes.append(_calculate_maximum_slope(derivative, ep, sampling_rate))
                        if len(existing_slopes) > 0:
                            qrs_slope_estimate = np.median(existing_slopes)
                            if peak_slope < 0.65 * qrs_slope_estimate:
                                continue  # Likely T-wave
                    
                    # Height check (use absolute values)
                    peak_height = abs(ecg_for_detection[gp])
                    if peak_height < signal_peak * 0.4:
                        continue
                    
                    # Accept this peak
                    additional_peaks.append(gp)
                    found_peak = True
                    break
                
                if found_peak:
                    break
        
        if additional_peaks:
            initial_r_peaks = np.sort(np.concatenate([initial_r_peaks, additional_peaks]))
    
    return initial_r_peaks.astype(int)


def _detect_r_peaks_derivative_based(
    ecg_filtered: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    sensitivity: Literal["standard", "high", "maximum"] = "standard",
) -> np.ndarray:
    """
    Detect R peaks using derivative-based single-pass approach (ECGPUWave-style).
    
    This function works with the signal as-is (no negation), finds zero-crossings
    in the derivative to detect peaks in both directions, and filters by slope
    magnitude. This naturally handles both inverted and upright R peaks in a
    single pass without needing merge logic.
    
    Parameters
    ----------
    ecg_filtered : np.ndarray
        Filtered ECG signal (works as-is, no negation).
    sampling_rate : float
        Sampling rate in Hz.
    cfg : ProcessCycleConfig
        Configuration with detection parameters.
    sensitivity : {"standard", "high", "maximum"}, default "standard"
        Detection sensitivity level.
    
    Returns
    -------
    np.ndarray
        Indices of detected R-peaks (both polarities, single pass).
    """
    # Compute derivative (ECGPUWave uses derivative for peak detection)
    derivative = np.gradient(ecg_filtered)
    
    # Training phase: signal/noise separation (ECGPUWave-style)
    signal_peak, noise_peak = _training_phase_signal_noise_separation(
        ecg_filtered, derivative, sampling_rate
    )
    
    # ECGPUWave threshold formula: noise + 0.25*(signal - noise)
    # Use prominence threshold based on training phase
    ecgpuwave_threshold = noise_peak + 0.25 * (signal_peak - noise_peak)
    
    # Adaptive threshold as fallback
    adaptive_threshold = _adaptive_prominence_threshold(
        ecg_filtered, cfg.rpeak_prominence_multiplier, sensitivity
    )
    
    # Use the more conservative threshold
    prominence_threshold = max(ecgpuwave_threshold, adaptive_threshold * 0.5)
    
    # Safety check
    signal_max = float(np.max(np.abs(ecg_filtered)))
    if prominence_threshold > signal_max * 0.8:
        prominence_threshold = signal_max * 0.3
    
    # ECGPUWave-style: Find peaks using derivative magnitude (slope)
    # R peaks have high derivative slope regardless of polarity
    # Use absolute derivative to find high-slope regions
    abs_derivative = np.abs(derivative)
    
    # Convert prominence threshold to slope threshold
    # Training phase gives prominence, but we need slope threshold
    # Use a multiplier to convert: slope threshold ≈ prominence * multiplier
    # For now, use a reasonable fraction of max slope
    max_slope = float(np.max(abs_derivative))
    slope_threshold = max_slope * 0.1  # 10% of max slope as initial threshold
    
    # Apply minimum distance (refractory period) filtering
    distance_samples = max(1, int(round(cfg.rpeak_min_refrac_ms * sampling_rate / 1000.0)))
    
    # Find peaks in absolute derivative (high-slope regions)
    # This naturally finds peaks in both directions
    derivative_peaks, _ = find_peaks(
        abs_derivative,
        distance=distance_samples,
        prominence=slope_threshold
    )
    
    if len(derivative_peaks) == 0:
        return np.array([], dtype=int)
    
    # Map derivative peaks back to signal peaks
    # For each derivative peak, find the corresponding signal peak
    # (zero-crossing in derivative near the derivative peak)
    signal_peaks = []
    for deriv_peak in derivative_peaks:
        # Look for zero-crossing in derivative near this peak
        # Search within ±20ms
        search_window = int(20.0 * sampling_rate / 1000.0)
        start_idx = max(0, deriv_peak - search_window)
        end_idx = min(len(derivative), deriv_peak + search_window)
        
        # Find zero-crossing (where derivative changes sign)
        local_deriv = derivative[start_idx:end_idx]
        if len(local_deriv) < 2:
            continue
        
        # Find where derivative crosses zero
        sign_changes = np.diff(np.sign(local_deriv))
        zero_crossings = np.where(np.abs(sign_changes) > 0)[0]
        
        if len(zero_crossings) > 0:
            # Use the zero-crossing closest to the derivative peak
            zc_idx = start_idx + zero_crossings[np.argmin(np.abs(zero_crossings - (deriv_peak - start_idx)))]
            # Peak is at zero-crossing + 1
            signal_peak_idx = zc_idx + 1
            if 0 <= signal_peak_idx < len(ecg_filtered):
                signal_peaks.append(signal_peak_idx)
        else:
            # Fallback: use derivative peak location directly
            if 0 <= deriv_peak < len(ecg_filtered):
                signal_peaks.append(deriv_peak)
    
    if len(signal_peaks) == 0:
        return np.array([], dtype=int)
    
    # Remove duplicates and sort
    signal_peaks = np.unique(np.array(signal_peaks, dtype=int))
    signal_peaks = np.sort(signal_peaks)
    
    # Apply distance filtering again (greedy approach)
    filtered_peaks = [signal_peaks[0]]
    for peak_idx in signal_peaks[1:]:
        if peak_idx - filtered_peaks[-1] >= distance_samples:
            filtered_peaks.append(peak_idx)
    
    filtered_peaks = np.array(filtered_peaks, dtype=int)
    
    if len(filtered_peaks) < 3:
        return filtered_peaks
    
    # Second pass: RR-adaptive distance with filtered RR intervals
    if len(filtered_peaks) >= 3:
        rr_intervals = np.diff(filtered_peaks)
        filtered_rr_intervals = _filter_rr_intervals(
            rr_intervals,
            np.median(rr_intervals),
            lower_frac=0.92,
            upper_frac=1.16
        )
        
        if len(filtered_rr_intervals) > 0:
            median_rr_samples = float(np.median(filtered_rr_intervals))
        else:
            median_rr_samples = float(np.median(rr_intervals))
        
        min_bpm, max_bpm = cfg.rpeak_bpm_bounds
        rr_min = (60_000.0 / max_bpm) * sampling_rate / 1000.0
        rr_max = (60_000.0 / min_bpm) * sampling_rate / 1000.0
        median_rr_samples = float(np.clip(median_rr_samples, rr_min, rr_max))
        
        distance = max(1, int(round(cfg.rpeak_rr_frac_second_pass * median_rr_samples)))
        
        # Re-detect with adaptive distance using find_peaks on absolute derivative
        derivative_peaks, _ = find_peaks(
            abs_derivative,
            distance=distance,
            prominence=slope_threshold
        )
        
        # Map derivative peaks back to signal peaks (same logic as first pass)
        signal_peaks = []
        for deriv_peak in derivative_peaks:
            search_window = int(20.0 * sampling_rate / 1000.0)
            start_idx = max(0, deriv_peak - search_window)
            end_idx = min(len(derivative), deriv_peak + search_window)
            
            local_deriv = derivative[start_idx:end_idx]
            if len(local_deriv) < 2:
                continue
            
            sign_changes = np.diff(np.sign(local_deriv))
            zero_crossings = np.where(np.abs(sign_changes) > 0)[0]
            
            if len(zero_crossings) > 0:
                zc_idx = start_idx + zero_crossings[np.argmin(np.abs(zero_crossings - (deriv_peak - start_idx)))]
                signal_peak_idx = zc_idx + 1
                if 0 <= signal_peak_idx < len(ecg_filtered):
                    signal_peaks.append(signal_peak_idx)
            else:
                if 0 <= deriv_peak < len(ecg_filtered):
                    signal_peaks.append(deriv_peak)
        
        if len(signal_peaks) > 0:
            signal_peaks = np.unique(np.array(signal_peaks, dtype=int))
            signal_peaks = np.sort(signal_peaks)
            
            # Apply distance filtering
            filtered_peaks = [signal_peaks[0]]
            for peak_idx in signal_peaks[1:]:
                if peak_idx - filtered_peaks[-1] >= distance:
                    filtered_peaks.append(peak_idx)
            
            filtered_peaks = np.array(filtered_peaks, dtype=int)
    
    # QRS characteristic filtering: reject T-waves and P-waves by slope
    if len(filtered_peaks) > 0:
        peak_slopes = []
        for peak_idx in filtered_peaks:
            slope = _calculate_maximum_slope(derivative, peak_idx, sampling_rate)
            peak_slopes.append(slope)
        peak_slopes = np.array(peak_slopes)
        
        if len(peak_slopes) >= 2:
            sorted_slopes = np.sort(peak_slopes)
            median_qrs_slope = np.median(sorted_slopes[len(sorted_slopes)//2:])
            slope_threshold_qrs = 0.75 * median_qrs_slope
        else:
            median_qrs_slope = float(np.median(peak_slopes)) if len(peak_slopes) > 0 else 0.1
            slope_threshold_qrs = 0.5 * median_qrs_slope
        
        slope_mask_qrs = peak_slopes >= slope_threshold_qrs
        filtered_peaks = filtered_peaks[slope_mask_qrs]
    
    # Final RR interval validation
    if len(filtered_peaks) >= 2:
        min_physiological_rr = 0.2 * sampling_rate  # 200ms minimum
        valid_peaks = [filtered_peaks[0]]
        
        for i in range(1, len(filtered_peaks)):
            prev_rr = filtered_peaks[i] - filtered_peaks[i-1]
            if prev_rr >= min_physiological_rr:
                min_bpm, max_bpm = cfg.rpeak_bpm_bounds
                min_rr_samples = (60_000.0 / max_bpm) * sampling_rate / 1000.0
                max_rr_samples = (60_000.0 / min_bpm) * sampling_rate / 1000.0
                if min_rr_samples <= prev_rr <= max_rr_samples:
                    valid_peaks.append(filtered_peaks[i])
        
        filtered_peaks = np.array(valid_peaks, dtype=int)
    
    return filtered_peaks


def r_peak_detection(
    ecg: Union[np.ndarray, list[float]],
    sampling_rate: float,
    *,
    cfg: ProcessCycleConfig,
    plot: bool = False,
    plot_start: Optional[float] = None,   # seconds
    plot_end: Optional[float] = None,     # seconds
    crop_ms: Optional[int] = 3000,        # plotting convenience only
    sensitivity: Literal["standard", "high", "maximum"] = "standard",
    raw_ecg: Optional[np.ndarray] = None,  # Optional raw signal (kept for compatibility)
) -> np.ndarray:
    """
    Enhanced multi-pass R-peak detection with simultaneous dual-polarity detection.
    
    ECGPUWAVE-style improvements:
    1. Explicit training phase (1-3s) to separate signal from noise
    2. Filtered RR intervals (filter outliers: 92-116% of median) before calculating expected RR
    3. Slope-based discrimination to reject T-waves (reject if slope < 75% of median QRS slope)
    4. Aggressive gap-filling with progressive threshold reduction (backtracking)
    5. **Simultaneous dual-polarity detection** - detects peaks in both polarities from the start
    
    This function detects R-peaks in BOTH polarities simultaneously (no global polarity decision),
    then merges and validates the results. This allows handling of mixed-polarity signals where
    some R-peaks are inverted and some are upright (e.g., due to lead placement or signal quality).
    
    Optionally applies bandpass filtering (cfg.rpeak_preprocess) for noise robustness.
    First-pass uses a fixed refractory (cfg.rpeak_min_refrac_ms) to estimate RR.
    Second-pass uses cfg.rpeak_rr_frac_second_pass * filtered_median(RR) as refractory.
    Third-pass (gap-fill) searches for missed peaks in large RR gaps with progressive threshold reduction.
    RR estimate is clamped using cfg.rpeak_bpm_bounds.
    
    Parameters
    ----------
    ecg : array-like
        1D ECG signal (typically preprocessed/filtered).
    sampling_rate : float
        Sampling rate in Hz.
    cfg : ProcessCycleConfig
        Configuration with detection parameters.
    plot : bool, default False
        Whether to plot detected peaks.
    plot_start, plot_end : float, optional
        Time window (seconds) for plotting.
    crop_ms : int, optional
        Crop duration for plotting.
    sensitivity : {"standard", "high", "maximum"}, default "standard"
        Detection sensitivity level:
        - "standard": Balanced precision/recall (default)
        - "high": Higher recall (~+15%), slightly lower precision
        - "maximum": Maximum recall, may include some noise
    raw_ecg : array-like, optional
        Raw (unfiltered) ECG signal. Kept for compatibility but not used for polarity detection.
    
    Returns
    -------
    np.ndarray
        Indices of detected R-peaks (from both polarities, merged and validated).
    """
    ecg = np.asarray(ecg, dtype=float)
    if ecg.ndim != 1 or ecg.size == 0:
        raise ValueError("`ecg` must be a non-empty 1D array.")
    if sampling_rate <= 0:
        raise ValueError("`sampling_rate` must be > 0.")

    # ----- Optional preprocessing for noise robustness -----
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

    # ----- Derivative-based single-pass detection (ECGPUWave-style) -----
    # Works with signal as-is, finds zero-crossings in derivative,
    # filters by slope magnitude - naturally handles both polarities
    final_filtered_r_peaks = _detect_r_peaks_derivative_based(
        ecg_filtered,
        sampling_rate,
        cfg,
        sensitivity
    )

    # ----- Optional plotting -----
    if plot:
        if plot_start is not None and plot_end is not None:
            if plot_start < 0 or plot_end <= plot_start:
                raise ValueError("`plot_end` must be > `plot_start` and both ≥ 0.")
            start_idx = int(round(plot_start * sampling_rate))
            end_idx = int(round(plot_end * sampling_rate))
            start_idx = max(0, min(start_idx, ecg.size))
            end_idx = max(start_idx + 1, min(end_idx, ecg.size))

            ecg_segment = ecg[start_idx:end_idx]
            peaks_in_window = final_filtered_r_peaks[
                (final_filtered_r_peaks >= start_idx) & (final_filtered_r_peaks < end_idx)
            ] - start_idx

            ph.plots.plot_rpeaks(
                ecg_segment,
                sampling_rate,
                peaks_in_window,
                crop_ms=crop_ms,
                title="ECG (windowed) with R-peaks",
            )
        else:
            ph.plots.plot_rpeaks(
                ecg,
                sampling_rate,
                final_filtered_r_peaks,
                crop_ms=crop_ms,
                title="ECG with R-peaks",
            )

    return final_filtered_r_peaks
