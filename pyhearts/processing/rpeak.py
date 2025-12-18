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
) -> np.ndarray:
    """
    Multi-pass, prominence-based R-peak detection driven by config.
    
    Optionally applies bandpass filtering (cfg.rpeak_preprocess) for noise robustness.
    First-pass uses a fixed refractory (cfg.rpeak_min_refrac_ms) to estimate RR.
    Second-pass uses cfg.rpeak_rr_frac_second_pass * median(RR) as refractory.
    Third-pass (gap-fill) searches for missed peaks in large RR gaps.
    RR estimate is clamped using cfg.rpeak_bpm_bounds.
    
    Parameters
    ----------
    ecg : array-like
        1D ECG signal.
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

    # ----- Adaptive prominence threshold -----
    prominence_threshold = _adaptive_prominence_threshold(
        ecg_filtered, cfg.rpeak_prominence_multiplier, sensitivity
    )
    
    # ----- First pass: conservative distance -----
    distance_lo = max(1, int(round(cfg.rpeak_min_refrac_ms * sampling_rate / 1000.0)))
    peaks_lo, _ = find_peaks(ecg_filtered, distance=distance_lo, prominence=prominence_threshold)

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
            ecg_filtered, distance=distance, prominence=prominence_threshold
        )
    
    # ----- Amplitude filtering: remove P/T waves detected as R-peaks -----
    # R-peaks should be the tallest peaks. Filter out peaks that are much
    # smaller than the maximum (likely P or T waves).
    if initial_r_peaks.size > 0:
        peak_heights = ecg_filtered[initial_r_peaks]
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
            gap_segment = ecg_filtered[start_idx:end_idx]
            
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
            r_peak_height_threshold = np.max(ecg_filtered[initial_r_peaks]) * 0.5
            
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
