from typing import Optional, Tuple, Dict, List, Any
from scipy.signal import savgol_filter
import numpy as np

from pyhearts.config import ProcessCycleConfig   



def compute_interdeflection_differences(
    peak_voltages: Dict[str, float], pairs: List[Tuple[str, str]], mode: str = "signed"
) -> Dict[str, float]:
    """
    Compute inter-deflection voltage differences from per-component peak amplitudes.

    Parameters
    ----------
    peak_voltages : dict
        Dictionary mapping wave labels to peak amplitude values.
    pairs : list of tuple
        List of wave label pairs to compare, e.g., [("R", "S"), ("T", "R")].
    mode : {"signed", "absolute"}
        Whether to return signed or absolute differences.

    Returns
    -------
    dict
        Dictionary mapping "WaveA_minus_WaveB" to the computed voltage difference.
    """
    differences = {}
    for wave_a, wave_b in pairs:
        amp_a = peak_voltages.get(wave_a)
        amp_b = peak_voltages.get(wave_b)
        key = f"{wave_a}_minus_{wave_b}"

        if amp_a is None or amp_b is None or not np.isfinite(amp_a) or not np.isfinite(amp_b):
            differences[key] = np.nan
            continue

        diff_value = amp_a - amp_b
        differences[key] = abs(diff_value) if mode == "absolute" else diff_value

    return differences



def compute_voltage_integrals(
    signal: np.ndarray,
    bounds: Dict[str, Tuple[int, int]],
    sampling_rate: float,
) -> Dict[str, float]:
    """
    Compute voltage integrals (area under the curve) for ECG components.

    Parameters
    ----------
    signal : np.ndarray
        1D detrended ECG cycle in millivolts (mV).
    bounds : dict[str, tuple[int, int]]
        Mapping from component label to (start_idx, end_idx), inclusive indices.
    sampling_rate : float
        Sampling rate in Hz (samples per second). Must be > 0.

    Returns
    -------
    dict[str, float]
        Mapping from "{label}_voltage_integral" to the area in microvolt·milliseconds (µV·ms).

    Notes
    -----
    - Integration uses the trapezoidal rule with a time step of 1 / sampling_rate seconds.
    - Unit conversion: (mV·s) → (µV·ms) via multiplication by 1e6.
    """
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be > 0")

    sig = np.asarray(signal, dtype=float).ravel()
    n = sig.size
    if n < 2 or not np.isfinite(sig).any():
        # Nothing integrable; return NaNs for all requested labels
        return {f"{label}_voltage_integral": np.nan for label in bounds.keys()}

    dt = 1.0 / sampling_rate  # seconds per sample
    result: Dict[str, float] = {}

    for label, idx_pair in bounds.items():
        key = f"{label}_voltage_integral"

        # Validate indices
        if (
            not isinstance(idx_pair, (tuple, list))
            or len(idx_pair) != 2
            or idx_pair[0] is None
            or idx_pair[1] is None
        ):
            result[key] = np.nan
            continue

        start_idx, end_idx = int(idx_pair[0]), int(idx_pair[1])

        # Require proper ordering and in-bounds indices
        if start_idx < 0 or end_idx < 0 or start_idx >= end_idx:
            result[key] = np.nan
            continue

        if start_idx >= n:
            result[key] = np.nan
            continue

        end_idx = min(end_idx, n - 1)
        segment = sig[start_idx : end_idx + 1]

        if segment.size < 2 or not np.isfinite(segment).all():
            result[key] = np.nan
            continue

        # Integrate in mV·s, then convert to µV·ms
        area_mv_s = np.trapezoid(segment, dx=dt)
        result[key] = float(area_mv_s * 1e6)  # µV·ms

    return result


def smooth_derivative(
    signal: np.ndarray,
    sampling_rate: float,
    window_ms: float = 20.0,
) -> np.ndarray:
    """
    Small helper: smooth signal and compute first derivative.

    Used as a building block for edge detection based on slope changes.
    """
    if signal.size == 0:
        return np.array([], dtype=float)

    fs = float(sampling_rate)
    win_samples = max(3, int(round(window_ms * fs / 1000.0)))
    if win_samples % 2 == 0:
        win_samples += 1

    # light smoothing to reduce high-frequency noise before derivative
    smoothed = savgol_filter(signal, window_length=win_samples, polyorder=2)
    deriv = np.gradient(smoothed)
    return deriv


def find_edge_by_derivative_and_baseline(
    signal: np.ndarray,
    center_idx: int,
    sampling_rate: float,
    *,
    direction: str,
    max_distance_ms: float = 200.0,
    slope_fraction: float = 0.2,
    baseline_window_ms: float = 60.0,
) -> int:
    """
    Generic edge finder:

    - Works on the detrended ECG signal.
    - Computes a smoothed derivative around the peak.
    - Uses a fraction of the local max derivative plus a baseline estimate
      to decide where the wave emerges from or returns to baseline.

    This is a generic slope- and baseline-based edge finder within a physiologic
    time window, suitable for locating waveform onsets/offsets around peaks.
    """
    n = signal.size
    if n == 0:
        return center_idx

    center_idx = int(center_idx)
    center_idx = max(0, min(center_idx, n - 1))

    fs = float(sampling_rate)
    max_dist = int(round(max_distance_ms * fs / 1000.0))
    if direction == "left":
        start = max(0, center_idx - max_dist)
        end = center_idx
    else:
        start = center_idx
        end = min(n - 1, center_idx + max_dist)

    seg = signal[start : end + 1]
    if seg.size < 3:
        return center_idx

    deriv = smooth_derivative(seg, sampling_rate)
    # local derivative threshold
    max_slope = float(np.max(np.abs(deriv)))
    if max_slope <= 0:
        return center_idx
    slope_thresh = slope_fraction * max_slope

    # baseline from outer window near segment edge
    base_win = int(round(baseline_window_ms * fs / 1000.0))
    if direction == "left":
        base_start = max(0, start - base_win)
        base_end = start
    else:
        base_start = end
        base_end = min(n - 1, end + base_win)
    base_segment = signal[base_start : base_end + 1]
    baseline = float(np.median(base_segment)) if base_segment.size > 0 else 0.0

    # walk away from center until both slope and amplitude get near baseline
    if direction == "left":
        it = np.arange(seg.size - 1, -1, -1)
    else:
        it = np.arange(seg.size)

    edge_idx = center_idx
    for k in it:
        s_val = seg[k]
        d_val = deriv[k]
        if abs(d_val) <= slope_thresh and abs(s_val - baseline) <= abs(max_slope) * 0.05:
            edge_idx = start + int(k)
            break

    return max(0, min(edge_idx, n - 1))



def _coerce_odd_window(win: int, poly: int) -> int:
    """
    Ensure the window size is valid for a Savitzky-Golay filter.

    The window must be:
      - At least `poly + 3`
      - An odd integer

    Parameters
    ----------
    win : int
        Desired window length.
    poly : int
        Polynomial order used in the Savitzky-Golay filter.

    Returns
    -------
    int
        Adjusted window length that is odd and >= `poly + 3`.
    """
    # ensure odd and >= poly+3 (Savitzky-Golay requirement)
    win = max(win, poly + 3)
    if win % 2 == 0:
        win += 1
    return win

    
import numpy as np
from typing import Optional
from scipy.signal import savgol_filter

def calc_sharpness_derivative(
    signal: np.ndarray,
    left_idx: int,
    right_idx: int,
    *,
    fs: float,
    cfg: Optional["ProcessCycleConfig"] = None,
) -> float:
    """
    Compute normalized sharpness of a segment using first derivatives.

    Only two user-facing knobs (via cfg) are respected:
      - cfg.sharp_stat:     {"mean","median","p95"}  (default "p95")
      - cfg.sharp_amp_norm: {"p2p","rms","mad"}      (default "p2p")

    Internal, fixed choices (not configurable):
      - Light smoothing: Savitzky–Golay (window=7, poly=3) when len(seg) >= 7
      - p2p percentiles: (5, 95)
      - epsilon guard:   1e-6

    Returns NaN if inputs are invalid or amplitude ≈ 0.
    """
    # --- Guardrails ---
    if left_idx is None or right_idx is None or left_idx >= right_idx:
        return np.nan
    if fs <= 0:
        return np.nan

    seg = np.asarray(signal[left_idx:right_idx + 1], dtype=float)
    if seg.ndim != 1 or seg.size < 3:
        return np.nan

    # --- Fixed internal smoothing (kept minimal and deterministic) ---
    if seg.size >= 7:
        seg = savgol_filter(seg, window_length=7, polyorder=3, mode="interp")

    # --- Derivative magnitude (per second) ---
    dt = 1.0 / float(fs)
    dseg = np.abs(np.diff(seg)) / dt

    # --- Summary statistic (cfg-controlled) ---
    stat = (cfg.sharp_stat if cfg is not None else "p95")
    if stat == "mean":
        sharp = float(np.mean(dseg))
    elif stat == "median":
        sharp = float(np.median(dseg))
    else:  # "p95"
        sharp = float(np.percentile(dseg, 95))

    # --- Amplitude normalization (cfg-controlled) ---
    norm = (cfg.sharp_amp_norm if cfg is not None else "p2p")
    if norm == "p2p":
        lo, hi = np.percentile(seg, [5.0, 95.0])  # fixed robust p2p
        amp = float(hi - lo)
    elif norm == "rms":
        amp = float(np.sqrt(np.mean(seg ** 2)))
    else:  # "mad"
        med = float(np.median(seg))
        amp = float(1.4826 * np.median(np.abs(seg - med)))

    # --- Stability guard (fixed epsilon) ---
    if not np.isfinite(amp) or amp < 1e-6:
        return np.nan

    val = sharp / amp
    return float(val) if np.isfinite(val) else np.nan




def estimate_local_baseline(
    sig: np.ndarray,
    center_idx: int,
    search_samples: int,
    direction: str = "both",
    cfg: Optional[ProcessCycleConfig] = None,
    comp_label: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Estimate local baseline and noise level around a wave.
    
    Uses robust statistics (median/MAD) on regions outside the wave itself.
    
    Parameters
    ----------
    sig : np.ndarray
        Input signal array.
    center_idx : int
        Index of the wave center.
    search_samples : int
        Total search window size in samples.
    direction : {"left", "right", "both"}
        Which direction(s) to use for baseline estimation.
    cfg : ProcessCycleConfig, optional
        Configuration object.
    
    Returns
    -------
    Tuple[float, float]
        (baseline_level, noise_level) where noise_level is MAD-based robust std.
    """
    cfg = cfg or ProcessCycleConfig()
    baseline_window_frac = cfg.local_baseline_window_fraction
    
    # For P-wave onset, use more of the pre-P region for baseline (ECGPUWAVE approach)
    if comp_label == "P" and direction in ("left", "both"):
        # Use larger window for P-wave baseline estimation
        baseline_window_frac = min(0.5, baseline_window_frac * 1.5)
    
    if direction in ("left", "both"):
        left_start = max(0, center_idx - search_samples)
        left_end = center_idx - int(search_samples * (1 - baseline_window_frac))
        left_segment = sig[left_start:left_end] if left_end > left_start else np.array([])
    else:
        left_segment = np.array([])
    
    if direction in ("right", "both"):
        right_start = center_idx + int(search_samples * (1 - baseline_window_frac))
        right_end = min(len(sig), center_idx + search_samples)
        right_segment = sig[right_start:right_end] if right_end > right_start else np.array([])
    else:
        right_segment = np.array([])
    
    # Combine segments
    if len(left_segment) > 0 and len(right_segment) > 0:
        baseline_segment = np.concatenate([left_segment, right_segment])
    elif len(left_segment) > 0:
        baseline_segment = left_segment
    elif len(right_segment) > 0:
        baseline_segment = right_segment
    else:
        # Fallback: use global median/MAD
        baseline_level = float(np.median(sig))
        noise_level = float(1.4826 * np.median(np.abs(sig - baseline_level)))
        return baseline_level, noise_level
    
    if len(baseline_segment) < 3:
        # Fallback: use global median/MAD
        baseline_level = float(np.median(sig))
        noise_level = float(1.4826 * np.median(np.abs(sig - baseline_level)))
    else:
        baseline_level = float(np.median(baseline_segment))
        noise_level = float(1.4826 * np.median(np.abs(baseline_segment - baseline_level)))
    
    return baseline_level, noise_level


def compute_adaptive_threshold(
    height: float,
    local_baseline: float,
    local_noise: float,
    cfg: ProcessCycleConfig,
    comp_label: str,
) -> float:
    """
    Compute adaptive threshold based on local SNR.
    
    For high SNR: use lower threshold (capture more of wave)
    For low SNR: use higher threshold (avoid noise)
    
    Parameters
    ----------
    height : float
        Peak amplitude.
    local_baseline : float
        Local baseline level.
    local_noise : float
        Local noise level (MAD-based).
    cfg : ProcessCycleConfig
        Configuration object.
    comp_label : str
        Component label (e.g., "P", "T").
    
    Returns
    -------
    float
        Adaptive threshold value.
    """
    peak_to_baseline = abs(height - local_baseline)
    if peak_to_baseline < 1e-6:
        return local_baseline  # fallback
    
    snr = peak_to_baseline / (local_noise + 1e-6)
    
    # Base threshold from config
    base_threshold_frac = cfg.threshold_fraction
    
    # Adjust based on SNR
    # High SNR (>10): reduce threshold by up to 30%
    # Low SNR (<3): increase threshold by up to 50%
    if snr > 10:
        adjustment = 0.7  # reduce threshold
    elif snr < 3:
        adjustment = 1.5  # increase threshold
    else:
        # Linear interpolation
        adjustment = 0.7 + (snr - 3) / (10 - 3) * (1.5 - 0.7)
    
    adaptive_frac = base_threshold_frac * adjustment
    adaptive_frac = np.clip(adaptive_frac, 0.05, 0.40)  # safety bounds
    
    if height >= local_baseline:
        threshold = local_baseline + adaptive_frac * (height - local_baseline)
    else:
        threshold = local_baseline - adaptive_frac * (local_baseline - height)
    
    return threshold


def find_waveform_limit_derivative(
    sig: np.ndarray,
    center_idx: int,
    height: float,
    std: float,
    *,
    sampling_rate: float,
    comp_label: str,
    cfg: ProcessCycleConfig,
    direction: str = "left",  # "left" for onset, "right" for offset
) -> int:
    """
    Find waveform limit using derivative-based approach.
    
    Detects the point where signal slope/curvature changes significantly
    relative to local baseline, rather than using fixed threshold.
    
    Parameters
    ----------
    sig : np.ndarray
        Input signal array.
    center_idx : int
        Index of the wave center.
    height : float
        Peak amplitude.
    std : float
        Estimated standard deviation of the peak shape.
    sampling_rate : float
        Sampling rate in Hz.
    comp_label : str
        Component label (e.g., "P", "T").
    cfg : ProcessCycleConfig
        Configuration object.
    direction : {"left", "right"}
        "left" for onset, "right" for offset.
    
    Returns
    -------
    int
        Index of the waveform limit.
    """
    # Search window: extend beyond Gaussian std estimate
    search_samples = int(round(cfg.shape_search_scale * std))
    if comp_label in cfg.shape_max_window_ms:
        max_window = int(round(cfg.shape_max_window_ms[comp_label] * sampling_rate / 1000.0))
        search_samples = min(search_samples, max_window)
    
    if direction == "left":
        search_start = max(0, center_idx - search_samples)
        search_end = center_idx
        search_slice = slice(search_start, search_end + 1)
    else:  # right/offset
        search_start = center_idx
        search_end = min(len(sig) - 1, center_idx + search_samples)
        search_slice = slice(search_start, search_end + 1)
    
    segment = sig[search_slice]
    if len(segment) < 5:
        return center_idx  # fallback
    
    # Step 1: Smooth signal to reduce noise
    # Use longer smoothing for T-offset
    smoothing_window = 7
    if comp_label == "T" and direction == "right":
        smoothing_window_ms = cfg.t_wave_offset_smoothing_window_ms
        smoothing_window = int(round(smoothing_window_ms * sampling_rate / 1000.0))
        smoothing_window = min(smoothing_window, len(segment) // 2)
        if smoothing_window < 7:
            smoothing_window = 7
        smoothing_window = smoothing_window if smoothing_window % 2 == 1 else smoothing_window + 1
    
    if len(segment) >= smoothing_window:
        segment_smooth = savgol_filter(segment, window_length=smoothing_window, polyorder=3, mode="interp")
    else:
        segment_smooth = segment
    
    # Step 2: Compute derivatives
    dt = 1.0 / sampling_rate
    first_deriv = np.diff(segment_smooth) / dt
    second_deriv = np.diff(first_deriv) / dt if len(first_deriv) > 1 else np.array([])
    
    # Step 3: Estimate local baseline and noise level
    local_baseline, local_noise = estimate_local_baseline(
        sig, center_idx, search_samples, direction=direction, cfg=cfg, comp_label=comp_label
    )
    
    # Step 4: Adjust sensitivity for P-waves and T-offset (ECGPUWAVE-style)
    deriv_multiplier = cfg.waveform_limit_deriv_multiplier
    baseline_multiplier = cfg.waveform_limit_baseline_multiplier
    
    if comp_label == "P" and direction == "left":
        # P-wave onset: more sensitive detection for gradual rise
        deriv_multiplier *= cfg.p_wave_deriv_sensitivity_multiplier * 0.6  # extra sensitivity
        baseline_multiplier *= 1.2  # more lenient baseline proximity for P onset
    elif comp_label == "T" and direction == "right":
        # T-wave offset: stricter baseline tolerance (ECGPUWAVE approach)
        # Require signal to be closer to baseline before detecting offset
        baseline_multiplier *= 0.7  # Stricter: 70% of default tolerance
        deriv_multiplier *= 1.3  # Less sensitive: require smaller derivative (130% of default)
    
    # Step 5: Adaptive threshold: derivative must drop to < noise_level × multiplier
    deriv_threshold = local_noise * deriv_multiplier
    
    # Step 6: Search from peak outward (ECGPUWAVE-style for P onset)
    limit_idx = center_idx
    
    # Map center_idx to segment coordinates
    if direction == "left":
        center_in_segment = center_idx - search_start
    else:  # right
        center_in_segment = 0  # center is at start of segment for right search
    
    # For P-wave onset, use more sophisticated detection (ECGPUWAVE-style)
    if comp_label == "P" and direction == "left":
        # ECGPUWAVE approach: find where signal transitions from baseline to rising
        # More conservative: require clear evidence of transition
        
        best_onset = center_idx
        min_deriv_rise = deriv_threshold * 0.4  # minimum derivative to indicate clear rise
        baseline_tolerance = local_noise * baseline_multiplier * 1.2  # slightly more lenient
        
        # Search backwards from peak, looking for transition point
        for i in range(len(segment_smooth) - 4):
            idx_in_segment = center_in_segment - i  # search backwards from peak
            
            if idx_in_segment < 4 or idx_in_segment >= len(segment_smooth) - 3:
                continue
            
            signal_at_idx = segment_smooth[idx_in_segment]
            
            # Check if we're at baseline level (with tolerance)
            at_baseline = abs(signal_at_idx - local_baseline) < baseline_tolerance
            
            if not at_baseline:
                continue  # Must be at baseline for P onset
            
            # Check derivatives: need to see transition from small to positive
            if idx_in_segment - 1 >= 0 and idx_in_segment - 1 < len(first_deriv):
                deriv_current = first_deriv[idx_in_segment - 1]
                deriv_current_mag = abs(deriv_current)
            else:
                continue
            
            # Check derivative ahead (2-3 samples ahead for more robust detection)
            deriv_ahead_positive = False
            if idx_in_segment + 1 < len(first_deriv):
                deriv_ahead = first_deriv[idx_in_segment + 1]
                deriv_ahead_positive = deriv_ahead > min_deriv_rise
            
            # Check signal values ahead to confirm sustained rise (more samples)
            signal_rising = False
            if idx_in_segment < len(segment_smooth) - 4:
                signal_ahead_1 = segment_smooth[idx_in_segment + 1]
                signal_ahead_2 = segment_smooth[min(idx_in_segment + 2, len(segment_smooth) - 1)]
                signal_ahead_3 = segment_smooth[min(idx_in_segment + 4, len(segment_smooth) - 1)]
                # Require sustained rise: each step should be higher
                signal_rising = (signal_ahead_1 > signal_at_idx and 
                                signal_ahead_2 > signal_ahead_1 and
                                signal_ahead_3 > signal_ahead_2)
            
            # P onset criteria (conservative ECGPUWAVE-style):
            # 1. At baseline ✓
            # 2. Current derivative is small (not yet rising significantly)
            # 3. Derivative ahead is clearly positive AND signal ahead shows sustained rise
            deriv_small = deriv_current_mag < deriv_threshold
            
            if deriv_small and deriv_ahead_positive and signal_rising:
                # Found good onset candidate: at baseline, derivative small, ahead is clearly rising
                best_onset = search_start + idx_in_segment
                # Don't break - continue searching backwards to find earliest valid onset
                # but update if we find a better (earlier) one that still meets criteria
        
        # Ensure we found a valid onset (not just the center)
        # Also ensure it's not too far from center (physiological constraint: P onset typically within 200ms of P peak)
        max_distance = int(round(200 * sampling_rate / 1000.0))
        if best_onset == center_idx or (center_idx - best_onset) > max_distance:
            # Didn't find a good onset or too far, will fall back to threshold method
            limit_idx = center_idx
        else:
            limit_idx = best_onset
    else:
        # Standard detection for other waves/offsets
        for i in range(len(segment_smooth) - 2):
            if direction == "left":
                idx_in_segment = center_in_segment - i  # search backwards
            else:
                idx_in_segment = i  # search forwards
            
            if idx_in_segment < 1 or idx_in_segment >= len(segment_smooth):
                continue
            
            # Check if we've reached baseline level
            signal_at_idx = segment_smooth[idx_in_segment]
            at_baseline = abs(signal_at_idx - local_baseline) < (local_noise * baseline_multiplier)
            
            # Check if derivative is below threshold (signal stopped changing)
            if idx_in_segment - 1 < len(first_deriv):
                deriv_mag = abs(first_deriv[idx_in_segment - 1])
                deriv_small = deriv_mag < deriv_threshold
            else:
                deriv_small = False
            
            # Check curvature change (transition indicator)
            curvature_change = False
            if len(second_deriv) > 0 and idx_in_segment - 1 < len(second_deriv) and idx_in_segment - 2 >= 0:
                prev_curvature = second_deriv[idx_in_segment - 2] if idx_in_segment - 2 < len(second_deriv) else 0
                curr_curvature = second_deriv[idx_in_segment - 1]
                curvature_change = (prev_curvature * curr_curvature < 0)  # sign change
            
            # Transition found if: at baseline AND derivative small
            # OR: curvature change detected (inflection point)
            if (at_baseline and deriv_small) or (curvature_change and at_baseline):
                # For T-offset, require sustained baseline (ECGPUWAVE approach)
                # T-waves have long tails - need to ensure signal stays at baseline
                if comp_label == "T" and direction == "right":
                    # Check if signal stays at baseline for next few samples
                    sustained_samples_required = int(round(20 * sampling_rate / 1000.0))  # 20ms
                    sustained_samples_required = min(sustained_samples_required, len(segment_smooth) - idx_in_segment - 1)
                    
                    if sustained_samples_required >= 3:
                        # Check next samples
                        all_at_baseline = True
                        for j in range(1, sustained_samples_required + 1):
                            check_idx = idx_in_segment + j
                            if check_idx >= len(segment_smooth):
                                all_at_baseline = False
                                break
                            
                            signal_check = segment_smooth[check_idx]
                            at_baseline_check = abs(signal_check - local_baseline) < (local_noise * baseline_multiplier)
                            
                            # Also check derivative
                            if check_idx - 1 < len(first_deriv):
                                deriv_check = abs(first_deriv[check_idx - 1])
                                deriv_small_check = deriv_check < deriv_threshold
                            else:
                                deriv_small_check = False
                            
                            if not (at_baseline_check and deriv_small_check):
                                all_at_baseline = False
                                break
                        
                        if all_at_baseline:
                            # Signal sustained at baseline - this is the offset
                            limit_idx = search_start + idx_in_segment
                            break
                        else:
                            # Not sustained, continue searching
                            continue
                    else:
                        # Not enough samples to check, use current point
                        limit_idx = search_start + idx_in_segment
                        break
                else:
                    # For other waves/onsets, use immediate detection
                    limit_idx = search_start + idx_in_segment
                    break
    
    # Step 7: For T-offset, check for U-wave
    if comp_label == "T" and direction == "right" and cfg.detect_u_wave:
        u_wave_check_window = int(round(0.1 * sampling_rate))  # 100ms after T peak
        if limit_idx + u_wave_check_window < len(sig):
            post_t_segment = sig[limit_idx:limit_idx + u_wave_check_window]
            if len(post_t_segment) > 3:
                # Look for secondary peak (U-wave)
                u_smooth_window = min(5, len(post_t_segment) // 2)
                if u_smooth_window >= 3 and u_smooth_window % 2 == 1:
                    post_t_smooth = savgol_filter(post_t_segment, window_length=u_smooth_window, polyorder=2)
                    post_t_deriv = np.diff(post_t_smooth)
                    # If derivative changes sign (potential U-wave), adjust search
                    if len(post_t_deriv) > 1 and np.any(np.diff(np.sign(post_t_deriv)) != 0):
                        # U-wave detected - be more conservative with T-offset
                        # Already found limit_idx should be fine, but we could refine
                        pass
    
    return limit_idx


def find_asymmetric_bounds_stdguided(
    sig: np.ndarray,
    center_idx: int,
    height: float,
    std: float,
    *,
    sampling_rate: int,
    comp_label: Optional[str] = None,
    cfg: Optional[ProcessCycleConfig] = None,
) -> Tuple[int, int]:
    """
    Find asymmetric left/right bounds around a signal peak using
    standard deviation and physiologic window constraints.
    
    Uses a hybrid approach:
    1. Primary: derivative-based waveform limit detection (ECGPUWAVE-style)
    2. Fallback: adaptive threshold crossing if derivative method fails
    
    The derivative-based method detects actual signal changes (slope/curvature)
    relative to local baseline, addressing systematic underestimation of
    PR and QT intervals.

    Parameters:
        sig (np.ndarray): Input signal array.
        center_idx (int): Index of the center peak.
        height (float): Peak amplitude (positive or negative).
        std (float): Estimated standard deviation of the peak shape.
        sampling_rate (int): Sampling rate of the signal in Hz.
        comp_label (str, optional): Component label (e.g., "P", "QRS", "T") 
            to use for physiologic window constraints.
        cfg (ProcessCycleConfig, optional): Configuration object. Defaults 
            to a new instance if None.

    Returns:
        Tuple[int, int]: Left and right indices marking the bounds of 
        the component.
    """
    cfg = cfg or ProcessCycleConfig()
    
    # Estimate local baseline
    search_samples = int(round(cfg.shape_search_scale * std)) if np.isfinite(std) else 0
    if comp_label and comp_label in cfg.shape_max_window_ms:
        physiol = int(round(cfg.shape_max_window_ms[comp_label] * sampling_rate / 1000.0))
        search_samples = physiol if search_samples == 0 else min(search_samples, physiol)
    
    local_baseline, local_noise = estimate_local_baseline(
        sig, center_idx, search_samples, direction="both", cfg=cfg, comp_label=comp_label
    )
    
    # Try derivative-based method first for T and (optionally) P waves.
    if cfg.use_derivative_based_limits and comp_label == "T":
        try:
            left_idx = find_waveform_limit_derivative(
                sig, center_idx, height, std,
                sampling_rate=float(sampling_rate),
                comp_label=comp_label,
                cfg=cfg,
                direction="left"
            )
            right_idx = find_waveform_limit_derivative(
                sig, center_idx, height, std,
                sampling_rate=float(sampling_rate),
                comp_label=comp_label,
                cfg=cfg,
                direction="right"
            )
            
            # Validate results - ensure reasonable bounds
            # For P-wave onset, ensure it's not too far from center (physiological constraint)
            if comp_label == "P":
                max_p_onset_distance = int(round(200 * sampling_rate / 1000.0))  # 200ms max
                if left_idx < center_idx - max_p_onset_distance:
                    # Too far back, likely wrong - use threshold fallback
                    left_idx = None
            
            if (left_idx is not None and right_idx is not None and 
                left_idx < center_idx < right_idx and 
                left_idx >= 0 and right_idx < len(sig)):
                return left_idx, right_idx
        except Exception:
            pass  # Fall back to threshold method
    
    # Fallback: adaptive threshold method
    adaptive_threshold = compute_adaptive_threshold(
        height, local_baseline, local_noise, cfg, comp_label or "R"
    )
    
    # Search with adaptive threshold
    max_offset = search_samples
    max_left = max(0, center_idx - max_offset)
    max_right = min(len(sig) - 1, center_idx + max_offset)
    
    left_idx = center_idx
    right_idx = center_idx
    
    if height >= local_baseline:
        while left_idx > max_left and sig[left_idx] > adaptive_threshold:
            left_idx -= 1
        while right_idx < max_right and sig[right_idx] > adaptive_threshold:
            right_idx += 1
    else:
        while left_idx > max_left and sig[left_idx] < adaptive_threshold:
            left_idx -= 1
        while right_idx < max_right and sig[right_idx] < adaptive_threshold:
            right_idx += 1
    
    return left_idx, right_idx



def extract_shape_features(
    signal: np.ndarray,
    gauss_centers: np.ndarray,
    gauss_stdevs: np.ndarray,
    gauss_heights: np.ndarray,
    component_labels: List[str],
    r_height: float,
    sampling_rate: int,
    *,
    cfg: Optional[ProcessCycleConfig] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Extract morphological shape features for labeled ECG components.

    For each Gaussian-parameterized component (P, Q, R, S, T), this function:
      - Finds asymmetric left/right bounds guided by standard deviation and
        physiologic constraints.
      - Computes duration, rise/decay times, rise-decay symmetry (RDSM),
        and sharpness (derivative-based, amplitude-normalized).
      - Performs concavity checks to ensure physiologic wave shapes.
      - Computes per-component voltage integrals and pairwise voltage differences.

    Parameters:
        signal (np.ndarray): Input 1D ECG signal array.
        gauss_centers (np.ndarray): Estimated Gaussian center indices for components.
        gauss_stdevs (np.ndarray): Estimated Gaussian standard deviations for components.
        gauss_heights (np.ndarray): Estimated Gaussian amplitudes (heights) for components.
        component_labels (List[str]): Labels for components (e.g., ["P", "Q", "R", "S", "T"]).
        r_height (float): R-peak amplitude, used for relative scaling.
        sampling_rate (int): Sampling frequency in Hz.
        cfg (ProcessCycleConfig, optional): Configuration object. Defaults to a new instance if None.
        verbose (bool, optional): If True, print debug information. Defaults to False.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - "valid_components" (List[str]): Labels of successfully processed components.
            - "per_component" (Dict[str, Dict[str, float]]): Per-wave feature dicts with:
                - duration_ms: Wave duration in ms
                - ri_idx, le_idx: Right/left indices
                - rise_ms, decay_ms: Rise and decay durations in ms
                - rdsm: Rise-decay symmetry ratio
                - sharpness: Derivative-based sharpness
                - voltage_integral_uv_ms: Area under the wave in µV·ms
            - "pairwise_differences" (Dict[str, float]): Inter-deflection voltage differences
              (e.g., R-T, R-S) according to cfg.shape_interdeflection_pairs.
    """
    cfg = cfg or ProcessCycleConfig()

    duration_min_samples = int(round(cfg.duration_min_ms * sampling_rate / 1000.0))

    shape_features_list: List[List[float]] = []
    valid_labels: List[str] = []

    peak_voltages_by_wave: Dict[str, float] = {
        label: float(h) if np.isfinite(h) else np.nan
        for label, h in zip(component_labels, gauss_heights)
    }
    bounds_by_wave: Dict[str, Tuple[int, int]] = {}

    for i, label in enumerate(component_labels):
        center = gauss_centers[i]
        height = gauss_heights[i]
        stdev  = gauss_stdevs[i]
        if not (np.isfinite(center) and np.isfinite(height) and np.isfinite(stdev)):
            continue

        approx_center_idx = int(round(center))
        left_idx, right_idx = find_asymmetric_bounds_stdguided(
            sig=signal,
            center_idx=approx_center_idx,
            height=height,
            std=stdev,
            sampling_rate=sampling_rate,
            comp_label=label,
            cfg=cfg,
        )
        left_idx = max(0, left_idx)
        right_idx = min(len(signal) - 1, right_idx)
        if right_idx <= left_idx:
            continue

        # refine center for Q/S as trough
        if label in {"Q", "S"}:
            segment = signal[left_idx:right_idx+1]
            if segment.size == 0:
                continue
            center_idx = left_idx + int(np.argmin(segment))
        else:
            center_idx = approx_center_idx

        duration_samples = right_idx - left_idx
        if duration_samples < duration_min_samples:
            continue
        rise_samples = center_idx - left_idx
        decay_samples = right_idx - center_idx
        if rise_samples <= 0 or decay_samples <= 0:
            continue

        # concavity checks
        if label in {"P","R","T"}:
            if (np.nanmin(signal[left_idx:center_idx+1]) > signal[center_idx] or
                np.nanmin(signal[center_idx:right_idx+1]) > signal[center_idx]):
                continue
        else:  # Q,S negative peaks
            if (np.nanmax(signal[left_idx:center_idx+1]) < signal[center_idx] or
                np.nanmax(signal[center_idx:right_idx+1]) < signal[center_idx]):
                continue

        rdsm = rise_samples / duration_samples
        sharpness = calc_sharpness_derivative(
            signal, left_idx, right_idx, fs=sampling_rate, cfg=cfg
        )

        to_ms = lambda s: (s / sampling_rate) * 1000.0
        shape_features_list.append(
            [to_ms(duration_samples), float(right_idx), float(left_idx),
             to_ms(rise_samples), to_ms(decay_samples), float(rdsm), float(sharpness)]
        )
        valid_labels.append(label)
        bounds_by_wave[label] = (left_idx, right_idx)

    # pack arrays
    shape_features_array = (
        np.asarray(shape_features_list, dtype=float) if shape_features_list else np.empty((0, 7))
    )

    # per-component dict
    per_component: Dict[str, Dict[str, float]] = {}
    for row_idx, label in enumerate(valid_labels):
        right_idx = float(shape_features_array[row_idx, 1])
        left_idx  = float(shape_features_array[row_idx, 2])
        per_component[label] = {
            "duration_ms": float(shape_features_array[row_idx, 0]),
            "ri_idx": right_idx,
            "le_idx": left_idx,
            "rise_ms": float(shape_features_array[row_idx, 3]),
            "decay_ms": float(shape_features_array[row_idx, 4]),
            "rdsm": float(shape_features_array[row_idx, 5]),
            "sharpness": float(shape_features_array[row_idx, 6]),
            "voltage_integral_uv_ms": float(
                np.nan if label not in bounds_by_wave else
                compute_voltage_integrals(signal, {label: bounds_by_wave[label]}, sampling_rate)[f"{label}_voltage_integral"]
            ),
        }

    # pairwise diffs
    pairs = cfg.shape_interdeflection_pairs
    diffs = compute_interdeflection_differences(
        peak_voltages_by_wave, pairs, mode=cfg.shape_diff_mode
    )
    diffs_named = {f"{k}_voltage_diff_{cfg.shape_diff_mode}": v for k, v in diffs.items()}

    shape_features: Dict[str, Any] = {
        "valid_components": valid_labels,
        "per_component": per_component,
        "pairwise_differences": diffs_named
    }
    return shape_features
