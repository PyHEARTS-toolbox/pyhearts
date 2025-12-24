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


def _detect_signal_polarity(
    ecg: np.ndarray,
    sampling_rate: float,
    min_refrac_ms: float = 100.0,
) -> bool:
    """
    Detect if the ECG signal has inverted QRS complexes.
    
    For inverted signals, the R-peaks are negative-going (nadirs) rather than
    positive peaks. This function compares the magnitude of positive vs negative
    peaks to determine signal polarity.
    
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
    
    # Compare the magnitude of the most prominent peaks
    # For inverted signals, negative peaks should be more prominent
    if pos_peaks.size > 0 and neg_peaks.size > 0:
        # Get peak prominences if available, otherwise use peak amplitudes
        pos_prominences = pos_props.get('prominences', None)
        neg_prominences = neg_props.get('prominences', None)
        
        if pos_prominences is None:
            # Fallback: use peak heights above baseline
            pos_prominences = ecg[pos_peaks] - np.median(ecg)
            pos_prominences = pos_prominences[pos_prominences > 0]
        if neg_prominences is None:
            # Fallback: use peak depths below baseline
            neg_prominences = np.median(ecg) - ecg[neg_peaks]
            neg_prominences = neg_prominences[neg_prominences > 0]
        
        # Compare median prominence of positive vs negative peaks
        median_pos_prom = np.median(pos_prominences) if len(pos_prominences) > 0 else 0
        median_neg_prom = np.median(neg_prominences) if len(neg_prominences) > 0 else 0
        
        # Also compare peak amplitudes (absolute values)
        median_pos_amp = np.median(np.abs(ecg[pos_peaks])) if len(pos_peaks) > 0 else 0
        median_neg_amp = np.median(np.abs(ecg[neg_peaks])) if len(neg_peaks) > 0 else 0
        
        # Signal is inverted if negative peaks are more prominent
        # Use stricter thresholds to avoid false positives
        # Negative peaks must be clearly dominant (1.3x threshold) to be considered inverted
        # This prevents false inversion detection when signal has slight negative bias
        # but R-peaks are actually positive
        
        # Check actual peak values (not just prominence) - R-peaks should be the largest deflections
        max_pos_val = np.max(ecg[pos_peaks]) if len(pos_peaks) > 0 else 0
        min_neg_val = np.min(ecg[neg_peaks]) if len(neg_peaks) > 0 else 0
        
        # Primary check: compare actual peak amplitudes
        # If positive peaks are substantial (>0.5 mV) and comparable to negative peaks,
        # assume normal polarity (conservative)
        if max_pos_val > 0.5 and max_pos_val > abs(min_neg_val) * 0.7:
            is_inverted = False
        # Only invert if negative peaks are clearly and substantially dominant
        elif abs(min_neg_val) > 0.5 and abs(min_neg_val) > max_pos_val * 1.3:
            is_inverted = True
        # Secondary check: use prominence comparison with strict threshold
        elif median_neg_prom > 1.3 * median_pos_prom and median_pos_prom > 0:
            is_inverted = True
        elif median_neg_amp > 1.4 * median_pos_amp and median_pos_amp > 0:
            is_inverted = True
        else:
            # Default: assume normal (conservative)
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
    
    # Find peaks in training window
    distance_lo = max(1, int(round(0.1 * sampling_rate)))  # 100ms minimum distance
    peaks, _ = find_peaks(
        training_segment,
        distance=distance_lo,
        prominence=np.std(training_segment) * 0.3
    )
    
    if len(peaks) == 0:
        signal_peak = float(np.max(np.abs(training_segment)))
        noise_peak = signal_peak * 0.5
        return signal_peak, noise_peak
    
    # Get peak amplitudes
    peak_amplitudes = [training_segment[p] for p in peaks]
    
    # Signal peak: highest peak
    signal_peak = float(np.max(peak_amplitudes))
    
    # Noise peak: highest peak below 75% of signal peak
    qrs75 = 0.75 * signal_peak
    noise_peaks = [amp for amp in peak_amplitudes if amp < qrs75]
    noise_peak = float(np.max(noise_peaks)) if len(noise_peaks) > 0 else signal_peak * 0.5
    
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
    raw_ecg: Optional[np.ndarray] = None,  # Optional raw signal for polarity detection
) -> np.ndarray:
    """
    Enhanced multi-pass R-peak detection with ECGPUWAVE-style improvements.
    
    Key improvements:
    1. Explicit training phase (1-3s) to separate signal from noise
    2. Filtered RR intervals (filter outliers: 92-116% of median) before calculating expected RR
    3. Slope-based discrimination to reject T-waves (reject if slope < 75% of median QRS slope)
    4. Aggressive gap-filling with progressive threshold reduction (backtracking)
    
    Optionally applies bandpass filtering (cfg.rpeak_preprocess) for noise robustness.
    First-pass uses a fixed refractory (cfg.rpeak_min_refrac_ms) to estimate RR.
    Second-pass uses cfg.rpeak_rr_frac_second_pass * filtered_median(RR) as refractory.
    Third-pass (gap-fill) searches for missed peaks in large RR gaps with progressive threshold reduction.
    RR estimate is clamped using cfg.rpeak_bpm_bounds.
    
    Automatically detects inverted QRS complexes (where R-peaks are negative-going
    nadirs rather than positive peaks) and handles them appropriately. This ensures
    correct detection for signals with inverted polarity while maintaining performance
    on normal (non-inverted) signals.
    
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
        Raw (unfiltered) ECG signal for polarity detection. If provided, polarity
        is detected on the raw signal to avoid issues where preprocessing may
        change apparent signal polarity. If None, polarity is detected on the
        filtered signal.
    
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

    # ----- Detect signal polarity (inverted vs normal) -----
    # For inverted signals, R-peaks are negative-going (nadirs) rather than positive peaks
    # Use raw signal for polarity detection if provided (preprocessing may change apparent polarity)
    # Otherwise use filtered signal
    polarity_signal = raw_ecg if raw_ecg is not None else ecg_filtered
    is_inverted = _detect_signal_polarity(
        polarity_signal, 
        sampling_rate, 
        min_refrac_ms=cfg.rpeak_min_refrac_ms
    )
    
    # If inverted, negate the signal for peak detection
    # This allows us to use the same peak-finding logic for both cases
    ecg_for_peak_detection = -ecg_filtered if is_inverted else ecg_filtered

    # ----- Compute derivative for slope-based discrimination -----
    derivative = np.gradient(ecg_for_peak_detection)

    # ----- ECGPUWAVE-style training phase: signal/noise separation -----
    signal_peak, noise_peak = _training_phase_signal_noise_separation(
        ecg_for_peak_detection, derivative, sampling_rate
    )
    
    # ECGPUWAVE threshold formula: noise + 0.25*(signal - noise)
    ecgpuwave_threshold = noise_peak + 0.25 * (signal_peak - noise_peak)
    
    # Use ECGPUWAVE threshold if it's reasonable, otherwise fall back to adaptive
    adaptive_threshold = _adaptive_prominence_threshold(
        ecg_filtered, cfg.rpeak_prominence_multiplier, sensitivity
    )
    
    # Use the more conservative of the two thresholds
    prominence_threshold = max(ecgpuwave_threshold, adaptive_threshold * 0.5)
    
    # ----- First pass: conservative distance -----
    # Use the (possibly negated) signal for peak detection
    distance_lo = max(1, int(round(cfg.rpeak_min_refrac_ms * sampling_rate / 1000.0)))
    peaks_lo, _ = find_peaks(ecg_for_peak_detection, distance=distance_lo, prominence=prominence_threshold)

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
            ecg_for_peak_detection, distance=distance, prominence=prominence_threshold
        )
    
    # ----- Slope-based filtering: reject T-waves (ECGPUWAVE-style) -----
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
            
            # Reject peaks with slope < 0.75 * median QRS slope (likely T-waves)
            slope_mask = peak_slopes >= 0.75 * median_qrs_slope
            initial_r_peaks = initial_r_peaks[slope_mask]
    
    # ----- Amplitude filtering: remove P/T waves detected as R-peaks -----
    # R-peaks should be the tallest peaks. Filter out peaks that are much
    # smaller than the maximum (likely P or T waves).
    # Use the peak detection signal for height comparison
    if initial_r_peaks.size > 0:
        peak_heights = ecg_for_peak_detection[initial_r_peaks]
        max_peak_height = np.max(peak_heights)
        
        # Keep only peaks that are at least 50% of the max height (increased from 40%)
        # (R-peaks are typically 2-10x taller than P/T waves)
        # Higher threshold reduces false positives from P/T waves
        height_threshold = 0.5 * max_peak_height
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
            gap_segment = ecg_for_peak_detection[start_idx:end_idx]
            
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
                    
                    # Height check
                    peak_height = ecg_for_peak_detection[gp]
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

    final_filtered_r_peaks = np.asarray(initial_r_peaks, dtype=int)

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
