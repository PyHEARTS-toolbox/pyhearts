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




def _find_derivative_zero_crossing(
    sig: np.ndarray,
    start_idx: int,
    direction: int,  # -1 for left, +1 for right
    max_steps: int,
    min_derivative_threshold: float = 1e-6,
) -> int:
    """
    Find where the derivative approaches zero (wave onset/offset).
    
    This is more robust than threshold-based detection for low-amplitude waves.
    """
    n = len(sig)
    current = start_idx
    
    # Compute derivative sign at starting point
    if direction == -1:
        # Going left: looking for where derivative changes from negative to ~zero
        for _ in range(max_steps):
            if current <= 1:
                break
            deriv = sig[current] - sig[current - 1]
            if abs(deriv) < min_derivative_threshold:
                break
            current -= 1
    else:
        # Going right: looking for where derivative changes from positive to ~zero
        for _ in range(max_steps):
            if current >= n - 2:
                break
            deriv = sig[current + 1] - sig[current]
            if abs(deriv) < min_derivative_threshold:
                break
            current += 1
    
    return current


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
    1. Primary: threshold-based crossing at cfg.threshold_fraction of peak height
    2. Fallback extension: derivative-based refinement for low-amplitude P/T waves
    
    The derivative-based extension helps address systematic underestimation
    of PR and QT intervals by capturing more of the wave onset/offset.

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

    thr = cfg.threshold_fraction * height
    # prefer std-guided search; cap by physiologic window if available
    if np.isfinite(std):
        max_offset = int(round(cfg.shape_search_scale * std))
    else:
        max_offset = 0

    if comp_label and comp_label in cfg.shape_max_window_ms:
        physiol = int(round(cfg.shape_max_window_ms[comp_label] * sampling_rate / 1000.0))
        max_offset = physiol if max_offset == 0 else min(max_offset, physiol)

    max_left = max(0, center_idx - max_offset)
    max_right = min(len(sig) - 1, center_idx + max_offset)

    left_idx = center_idx
    right_idx = center_idx

    # Primary: threshold-based crossing
    if height >= 0:
        while left_idx > max_left and sig[left_idx] > thr:
            left_idx -= 1
        while right_idx < max_right and sig[right_idx] > thr:
            right_idx += 1
    else:
        while left_idx > max_left and sig[left_idx] < thr:
            left_idx -= 1
        while right_idx < max_right and sig[right_idx] < thr:
            right_idx += 1
    
    # Derivative-based extension for P and T waves (addresses interval biases)
    # Only apply to P and T waves which showed systematic underestimation
    if comp_label in ("P", "T"):
        # Compute baseline noise level for derivative threshold
        segment = sig[max_left:max_right + 1]
        if len(segment) > 5:
            baseline_deriv = np.median(np.abs(np.diff(segment))) * 0.3
        else:
            baseline_deriv = abs(height) * 0.02
        
        # Extension allowance: up to 20% more samples beyond threshold crossing
        extension_budget = max(2, int(0.20 * (center_idx - left_idx)))
        
        # Extend left boundary if derivative is still significant
        extended_left = left_idx
        for _ in range(extension_budget):
            if extended_left <= max_left + 1:
                break
            deriv = abs(sig[extended_left] - sig[extended_left - 1])
            if deriv < baseline_deriv:
                break
            extended_left -= 1
        
        # Extend right boundary similarly
        extension_budget = max(2, int(0.20 * (right_idx - center_idx)))
        extended_right = right_idx
        for _ in range(extension_budget):
            if extended_right >= max_right - 1:
                break
            deriv = abs(sig[extended_right + 1] - sig[extended_right])
            if deriv < baseline_deriv:
                break
            extended_right += 1
        
        left_idx = extended_left
        right_idx = extended_right

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
