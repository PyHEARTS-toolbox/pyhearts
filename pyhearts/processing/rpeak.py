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
    """Apply bandpass (and optional notch) filter for R-peak detection."""
    filtered = ecg.copy()
    
    # Highpass to remove baseline wander
    if highpass_hz > 0:
        nyq = sampling_rate / 2
        # Ensure cutoff is valid
        hp_norm = min(highpass_hz / nyq, 0.99)
        b, a = butter(order, hp_norm, btype='high')
        filtered = filtfilt(b, a, filtered)
    
    # Lowpass to remove high-frequency noise
    if lowpass_hz > 0:
        nyq = sampling_rate / 2
        lp_norm = min(lowpass_hz / nyq, 0.99)
        b, a = butter(order, lp_norm, btype='low')
        filtered = filtfilt(b, a, filtered)
    
    # Optional notch filter for power line interference
    if notch_hz is not None and notch_hz > 0:
        q = 30.0  # Quality factor
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
        # Fallback: check overall signal bias
        # If signal is predominantly negative, it's likely inverted
        signal_median = np.median(ecg)
        signal_mean = np.mean(ecg)
        if signal_median < -0.1 or signal_mean < -0.1:
            return True
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
        # Use a more lenient threshold: negative peaks must be at least 1.1x more prominent
        # OR if negative amplitude is significantly larger (1.15x)
        # OR if negative prominence is at least equal and negative amplitude is larger
        # This handles cases where filtering affects prominence but amplitude is clear
        is_inverted = (
            (median_neg_prom > 1.1 * median_pos_prom) or 
            (median_neg_amp > 1.15 * median_pos_amp) or
            (median_neg_prom >= median_pos_prom and median_neg_amp > median_pos_amp)
        )
        
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
    Multi-pass, prominence-based R-peak detection driven by config.
    
    Optionally applies bandpass filtering (cfg.rpeak_preprocess) for noise robustness.
    First-pass uses a fixed refractory (cfg.rpeak_min_refrac_ms) to estimate RR.
    Second-pass uses cfg.rpeak_rr_frac_second_pass * median(RR) as refractory.
    Third-pass (gap-fill) searches for missed peaks in large RR gaps.
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

    # ----- Adaptive prominence threshold -----
    # Use the original filtered signal for threshold calculation to maintain consistency
    prominence_threshold = _adaptive_prominence_threshold(
        ecg_filtered, cfg.rpeak_prominence_multiplier, sensitivity
    )
    
    # ----- First pass: conservative distance -----
    # Use the (possibly negated) signal for peak detection
    distance_lo = max(1, int(round(cfg.rpeak_min_refrac_ms * sampling_rate / 1000.0)))
    peaks_lo, _ = find_peaks(ecg_for_peak_detection, distance=distance_lo, prominence=prominence_threshold)

    # ----- Second pass: RR-adaptive distance -----
    if peaks_lo.size < 3:
        initial_r_peaks = peaks_lo.astype(int)
        median_rr_samples = distance_lo  # fallback
    else:
        rr_samp = np.median(np.diff(peaks_lo))

        min_bpm, max_bpm = cfg.rpeak_bpm_bounds
        rr_min = (60_000.0 / max_bpm) * sampling_rate / 1000.0
        rr_max = (60_000.0 / min_bpm) * sampling_rate / 1000.0
        rr_samp = float(np.clip(rr_samp, rr_min, rr_max))
        median_rr_samples = rr_samp

        distance = max(1, int(round(cfg.rpeak_rr_frac_second_pass * rr_samp)))
        initial_r_peaks, _ = find_peaks(
            ecg_for_peak_detection, distance=distance, prominence=prominence_threshold
        )
    
    # ----- Amplitude filtering: remove P/T waves detected as R-peaks -----
    # R-peaks should be the tallest peaks. Filter out peaks that are much
    # smaller than the maximum (likely P or T waves).
    # Use the peak detection signal for height comparison
    if initial_r_peaks.size > 0:
        peak_heights = ecg_for_peak_detection[initial_r_peaks]
        max_peak_height = np.max(peak_heights)
        
        # Keep only peaks that are at least 40% of the max height
        # (R-peaks are typically 2-10x taller than P/T waves)
        height_threshold = 0.4 * max_peak_height
        amplitude_mask = peak_heights >= height_threshold
        initial_r_peaks = initial_r_peaks[amplitude_mask]
    
    # ----- Third pass: gap-filling for missed beats -----
    # Search for missed peaks in gaps > 1.8× median RR (likely missed beats)
    # Only enabled for "maximum" sensitivity to avoid T-peak false positives
    if initial_r_peaks.size >= 2 and sensitivity == "maximum":
        gap_threshold = 1.8 * median_rr_samples  # More conservative: 1.8x instead of 1.5x
        rr_intervals = np.diff(initial_r_peaks)
        large_gaps = np.where(rr_intervals > gap_threshold)[0]
        
        additional_peaks = []
        reduced_prominence = prominence_threshold * 0.8  # Less lenient: 0.8 instead of 0.7
        
        for gap_idx in large_gaps:
            start_idx = initial_r_peaks[gap_idx]
            end_idx = initial_r_peaks[gap_idx + 1]
            gap_segment = ecg_for_peak_detection[start_idx:end_idx]
            
            if gap_segment.size < 10:
                continue
            
            # Find peaks in the gap with reduced prominence
            gap_peaks, props = find_peaks(
                gap_segment,
                distance=max(1, int(0.5 * median_rr_samples)),  # Larger min distance
                prominence=reduced_prominence,
            )
            
            # Only keep peaks that are:
            # 1. Reasonably positioned (not too close to edges)
            # 2. Actually tall enough to be R-peaks (not T-peaks)
            margin = int(0.25 * median_rr_samples)
            r_peak_height_threshold = np.max(ecg_for_peak_detection[initial_r_peaks]) * 0.5
            
            for gp in gap_peaks:
                global_idx = start_idx + gp
                peak_height = gap_segment[gp]
                if margin < gp < len(gap_segment) - margin and peak_height > r_peak_height_threshold:
                    additional_peaks.append(global_idx)
        
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
