from __future__ import annotations

from functools import partial
from typing import Dict, Mapping, Optional, Tuple, Literal, Union, List

import numpy as np
from pyhearts.config import ProcessCycleConfig

# Type aliases (Python 3.9 compatible)
Peak = Tuple[Optional[int], Optional[float]]
Peaks = Mapping[str, Peak]
ValidatedPeaks = Dict[str, Peak]


def validate_p_wave_morphology(
    signal: np.ndarray,
    p_center_idx: int,
    p_height: float,
    sampling_rate: float,
    r_center_idx: Optional[int] = None,
    verbose: bool = False,
    cycle_idx: Optional[int] = None,
) -> Tuple[bool, Dict[str, float]]:
    """
    Validate P wave morphology to distinguish P waves from Q peaks.
    
    Morphology-based validation using:
    1. Duration: P waves are 60-200ms wide, Q peaks are 20-40ms
    2. Shape: P waves have gradual rise/fall (low max derivative), Q peaks are sharp (high derivative)
    3. Onset/Offset detection: P waves should have detectable onset and offset points
    
    Parameters
    ----------
    signal : np.ndarray
        ECG signal segment containing the P wave candidate.
    p_center_idx : int
        Index of the P wave peak candidate.
    p_height : float
        Amplitude of the P wave candidate.
    sampling_rate : float
        Sampling rate in Hz.
    r_center_idx : int, optional
        R peak index (for distance validation).
    verbose : bool, default False
        If True, print diagnostic messages.
    cycle_idx : int, optional
        Cycle index for logging.
    
    Returns
    -------
    Tuple[bool, Dict[str, float]]
        (is_valid, morphology_features)
        - is_valid: True if morphology suggests P wave, False if likely Q peak
        - morphology_features: dict with duration_ms, max_derivative, sharpness_ratio
    """
    if p_center_idx < 0 or p_center_idx >= len(signal):
        return False, {}
    
    # Define search window around P peak
    # P waves are typically 60-200ms wide, so search ±150ms
    search_window_ms = 150.0
    search_window_samples = int(round(search_window_ms * sampling_rate / 1000.0))
    
    search_start = max(0, p_center_idx - search_window_samples)
    search_end = min(len(signal), p_center_idx + search_window_samples + 1)
    
    if search_end - search_start < 10:  # Need at least 10 samples
        if verbose:
            print(f"[Cycle {cycle_idx}]: P morphology validation failed - window too small")
        return False, {}
    
    # Extract segment around P peak
    segment = signal[search_start:search_end]
    p_local_idx = p_center_idx - search_start
    
    # Compute derivative to assess sharpness
    # Q peaks are sharp (high derivative), P waves are gradual (low derivative)
    derivative = np.gradient(segment)
    abs_derivative = np.abs(derivative)
    
    # Find maximum derivative magnitude (sharpness indicator)
    max_derivative = float(np.max(abs_derivative))
    
    # Normalize by P wave amplitude to get sharpness ratio
    # Q peaks have high sharpness (derivative/amplitude), P waves have low sharpness
    if abs(p_height) > 0:
        sharpness_ratio = max_derivative / abs(p_height)
    else:
        sharpness_ratio = float('inf')
    
    # Detect onset and offset using derivative-based method
    # More robust: use derivative to find where signal starts rising/falling
    # Compute local baseline around P peak (median of ±50ms window)
    baseline_window_ms = 50.0
    baseline_window_samples = int(round(baseline_window_ms * sampling_rate / 1000.0))
    baseline_start = max(0, p_local_idx - baseline_window_samples)
    baseline_end = min(len(segment), p_local_idx + baseline_window_samples + 1)
    local_baseline = np.median(segment[baseline_start:baseline_end])
    
    # Use 10% of peak-to-baseline difference as threshold (more lenient)
    peak_to_baseline = abs(p_height - local_baseline)
    threshold = peak_to_baseline * 0.10
    
    # Find onset (left side): search backwards from peak until signal returns to baseline
    onset_idx = p_local_idx
    for i in range(p_local_idx - 1, -1, -1):
        signal_value = segment[i]
        distance_from_baseline = abs(signal_value - local_baseline)
        if distance_from_baseline > threshold:
            # Still above threshold, continue
            continue
        else:
            # Returned to baseline, this is the onset
            onset_idx = i
            break
    
    # Find offset (right side): search forwards from peak until signal returns to baseline
    offset_idx = p_local_idx
    for i in range(p_local_idx + 1, len(segment)):
        signal_value = segment[i]
        distance_from_baseline = abs(signal_value - local_baseline)
        if distance_from_baseline > threshold:
            # Still above threshold, continue
            continue
        else:
            # Returned to baseline, this is the offset
            offset_idx = i
            break
    
    # Calculate duration
    duration_samples = offset_idx - onset_idx
    duration_ms = (duration_samples / sampling_rate) * 1000.0
    
    # Morphology validation criteria:
    # 1. Duration: P waves are 30-200ms, Q peaks are 10-30ms
    # Use lenient minimum (30ms) - some P waves can be short
    min_p_duration_ms = 30.0  # Very lenient - P waves can be as short as 30ms
    max_p_duration_ms = 200.0
    max_q_duration_ms = 30.0  # Q peaks are narrower
    
    # 2. Sharpness: P waves have gradual rise/fall (low sharpness), Q peaks are sharp (high sharpness)
    # Sharpness ratio = max_derivative / amplitude
    # P waves: typically < 0.5, Q peaks: typically > 1.0
    max_p_sharpness = 1.5  # Very lenient - allow higher sharpness for P waves
    min_q_sharpness = 1.0  # Q peaks are sharper
    
    # 3. Onset/Offset detection: P waves should have detectable boundaries
    # If onset/offset are too close to peak, it's likely a Q peak
    min_boundary_distance_ms = 10.0  # Very lenient - allow 10ms minimum
    min_boundary_distance_samples = int(round(min_boundary_distance_ms * sampling_rate / 1000.0))
    
    onset_distance = p_local_idx - onset_idx
    offset_distance = offset_idx - p_local_idx
    
    # Validation checks: Use combined criteria - reject only if clearly Q peak
    is_valid = True
    rejection_reason = None
    
    # Primary check: If duration is very short AND sharpness is high, definitely Q peak
    # This is the strongest indicator of Q peak vs P wave
    if duration_ms < max_q_duration_ms and sharpness_ratio > min_q_sharpness:
        is_valid = False
        rejection_reason = f"Q peak characteristics (duration={duration_ms:.1f}ms < {max_q_duration_ms}ms AND sharpness={sharpness_ratio:.2f} > {min_q_sharpness})"
    
    # Secondary check: Very sharp AND very short duration (likely Q peak)
    # Only reject if both conditions are met (more conservative)
    elif duration_ms < 20.0 and sharpness_ratio > 1.2:
        is_valid = False
        rejection_reason = f"very short and sharp (duration={duration_ms:.1f}ms < 20ms AND sharpness={sharpness_ratio:.2f} > 1.2, likely Q peak)"
    
    # Warning for borderline cases (but don't reject)
    if is_valid:
        if duration_ms < min_p_duration_ms:
            if verbose:
                print(f"[Cycle {cycle_idx}]: P wave warning - duration borderline ({duration_ms:.1f}ms < {min_p_duration_ms}ms), but accepting due to low sharpness")
        if duration_ms > max_p_duration_ms:
            if verbose:
                print(f"[Cycle {cycle_idx}]: P wave warning - duration very wide ({duration_ms:.1f}ms > {max_p_duration_ms}ms)")
    
    morphology_features = {
        "duration_ms": duration_ms,
        "max_derivative": max_derivative,
        "sharpness_ratio": sharpness_ratio,
        "onset_distance_samples": onset_distance,
        "offset_distance_samples": offset_distance,
    }
    
    if verbose:
        if is_valid:
            print(f"[Cycle {cycle_idx}]: P wave morphology validated: duration={duration_ms:.1f}ms, sharpness={sharpness_ratio:.2f}")
        else:
            print(f"[Cycle {cycle_idx}]: P wave morphology rejected: {rejection_reason}")
    
    return is_valid, morphology_features


def validate_peaks(
    peaks: Peaks,
    r_center_idx: int | None,
    r_height: float | None,
    sampling_rate: float | None,
    verbose: bool,
    cycle_idx: int | None,
    *,
    cfg: ProcessCycleConfig | None = None,
    q_center_idx_for_validation: int | None = None,  # Q center for P wave validation (may come from simplified detection)
) -> ValidatedPeaks:
    """
    Validate ECG peaks for polarity and minimum amplitude relative to R.

    Parameters
    ----------
    peaks : mapping from component -> (center_idx, height)
    r_center_idx : index of R center for the same cycle
    r_height : amplitude of R peak in same cycle
    sampling_rate : Hz (reserved for future proximity checks)
    verbose : enable per-peak logs
    cycle_idx : cycle number for logs
    cfg : ProcessCycleConfig carrying amp_min_ratio; falls back to defaults if None
    q_center_idx_for_validation : int | None, optional
        Q center index for P wave validation (from simplified detection if full Q detection skipped)

    Returns
    -------
    dict[str, Peak]
        Validated peaks with invalid entries set to (None, None).
    """
    expected: Mapping[str, Literal["peak", "trough"]] = {
        "P": "peak",
        "Q": "trough",
        "S": "trough",
        "T": "peak",
    }

    # Use Q from peaks if available, otherwise use validation Q
    q_center_idx_from_peaks = peaks.get("Q", (None, None))[0]
    q_center_idx = q_center_idx_from_peaks if q_center_idx_from_peaks is not None else q_center_idx_for_validation

    validate_one = partial(
        log_peak_result,
        r_center_idx=r_center_idx,
        r_height=r_height,
        sampling_rate=sampling_rate,
        verbose=verbose,
        cycle_idx=cycle_idx,
        cfg=cfg,
        q_center_idx=q_center_idx,  # Pass Q center for P validation
    )

    return {
        comp: validate_one(
            comp=comp,
            center_idx=center,
            height=height,
            expected_polarity=expected[comp],
        )
        for comp, (center, height) in peaks.items()
    }


def log_peak_result(
    comp: str,
    center_idx: int | None,
    height: float | None,
    expected_polarity: Literal["peak", "trough"] = "peak",
    r_center_idx: int | None = None,
    r_height: float | None = None,
    sampling_rate: float | None = None,
    verbose: bool = False,
    cycle_idx: int | None = None,
    cfg: ProcessCycleConfig | None = None,
    q_center_idx: int | None = None,  # Added for P wave validation
) -> Peak:
    """
    Apply polarity and relative-amplitude checks for a single component.
    
    For P waves, also validates distance from QRS complex to avoid misclassifying
    Q peaks as P waves (morphology-based validation).
    """
    # Missing or invalid amplitude
    if center_idx is None or height is None or not np.isfinite(height):
        if verbose:
            print(f"[Cycle {cycle_idx}]: {comp} peak not found.")
        return None, None

    # P wave validation: reject P waves too close to QRS
    # This prevents misclassifying inverted Q peaks as P waves when P is absent or too small
    # Can be disabled via config to match high-sensitivity detection style (no distance checks)
    if comp == "P" and r_center_idx is not None and sampling_rate is not None:
        # Check if distance validation is enabled (default True for backward compatibility)
        enable_distance_validation = getattr(cfg, "p_enable_distance_validation", True) if cfg is not None else True
        
        if enable_distance_validation:
            p_r_distance_samples = r_center_idx - center_idx
            p_r_distance_ms = (p_r_distance_samples / sampling_rate) * 1000.0
            
            # P waves should be at least 100ms before R peak (to avoid QRS complex and Q peaks)
            # Typical P-R intervals are 120-200ms, but can be as short as 100-120ms in some cases.
            # Established delineation methods show P-R intervals as low as 124ms, so we use 100ms to avoid rejecting valid P waves.
            # If P is too close to R (< 100ms), it's likely a Q peak or QRS artifact, not a P wave.
            min_p_r_distance_ms = 100.0
            if p_r_distance_ms < min_p_r_distance_ms:
                if verbose:
                    print(f"[Cycle {cycle_idx}]: P wave rejected - too close to R peak "
                          f"({p_r_distance_ms:.1f}ms < {min_p_r_distance_ms}ms). Likely Q peak or QRS artifact, not P wave.")
                return None, None
            
            # If Q peak is detected, P should be well before Q (at least 50ms)
            # Use Q position to distinguish P from Q - if P is too close to Q,
            # it's likely a Q peak being misclassified as P (especially for inverted QRS)
            if q_center_idx is not None:
                p_q_distance_samples = q_center_idx - center_idx
                p_q_distance_ms = (p_q_distance_samples / sampling_rate) * 1000.0
                
                # Stricter threshold: P should be at least 50ms before Q
                # If P is within 50ms of Q, it's likely part of the QRS complex, not a true P wave
                min_p_q_distance_ms = 50.0
                if p_q_distance_ms < min_p_q_distance_ms:
                    if verbose:
                        print(f"[Cycle {cycle_idx}]: P wave rejected - too close to Q peak "
                              f"({p_q_distance_ms:.1f}ms < {min_p_q_distance_ms}ms). Likely Q peak misclassified as P.")
                    return None, None
                
                # Additional check: if P-Q distance is very short (< 100ms), be suspicious
                # True P waves are typically 100-200ms before Q, not 50-100ms
                if p_q_distance_ms < 100.0:
                    if verbose:
                        print(f"[Cycle {cycle_idx}]: P wave warning - P-Q distance is short "
                              f"({p_q_distance_ms:.1f}ms < 100ms). May be Q peak.")
                    # Don't reject, but log warning for now
        
        # Morphology-based validation: distinguish P waves from Q peaks using shape/duration
        # This works even when Q detection fails (e.g., at low sampling rates)
        # We need the full signal segment for morphology analysis
        # Note: This requires access to the signal, which we'll need to pass in
        # For now, we'll skip morphology validation if signal is not available
        # (This will be called from process_cycle where signal is available)

    # Polarity checks
    # Step 5: Allow negative P-waves and T-waves for inverted leads or detrended signals
    # For T-waves: Accept both positive and negative (detrending can change apparent polarity)
    # The morphology detection in detect_t_wave_derivative_based handles true morphology
    if expected_polarity == "peak" and height < 0 and comp in ("P", "T"):
        # For P-waves, allow negative if it's the only detection (inverted leads)
        # For T-waves, always accept negative - detrended signals can make positive T waves appear negative,
        # and truly inverted T waves should also be accepted
        if comp == "P":
            if verbose:
                print(f"[Cycle {cycle_idx}]: P-wave is negative (inverted lead), accepting anyway.")
            # Accept negative P-wave (inverted lead)
        else:
            # T-waves: Always accept negative (can be due to detrending or truly inverted)
            # The morphology detection algorithm determines this, so trust its judgment
            if verbose:
                print(f"[Cycle {cycle_idx}]: T-wave is negative (inverted or detrended signal), accepting anyway.")
            # Accept negative T-wave
    # Also allow positive T-waves even if they might be inverted in raw signal
    # (detrending can make inverted T waves appear positive)
    if expected_polarity == "peak" and height >= 0 and comp == "T":
        # Accept positive T-waves - they may be inverted in raw signal but appear positive after detrending
        # The morphology detection algorithm handles this
        pass  # Accept by default
    if expected_polarity == "trough" and height >= 0 and comp in ("Q", "S"):
        if verbose:
            print(f"[Cycle {cycle_idx}]: {comp} polarity invalid (expected negative).")
        return None, None

    # Relative amplitude check against R
    if r_height is not None and np.isfinite(r_height):
        r_abs = float(abs(r_height))
        ratios = (
            cfg.amp_min_ratio  # type: ignore[attr-defined]
            if (cfg is not None and hasattr(cfg, "amp_min_ratio"))
            else {"P": 0.03, "T": 0.03, "Q": 0.005, "S": 0.005}
        )
        min_ratio = ratios.get(comp)

        if min_ratio is not None:
            # Use config value for amplitude ratio (allows high-sensitivity detection style higher thresholds)
            # For P and T waves, use the config value directly (no hardcoded override)
            # Absolute minimum check to avoid accepting noise
            abs_min = 0.001  # Very small absolute minimum (1 microvolt) to avoid accepting pure noise
            
            # Calculate threshold: use config ratio, but ensure absolute minimum
            threshold = max(min_ratio * r_abs, abs_min)
            
            if abs(float(height)) < threshold:
                if verbose:
                    print(
                        f"[Cycle {cycle_idx}]: {comp} too small "
                        f"({abs(float(height)):.3f} < {threshold:.3f}, ratio={min_ratio:.4f})."
                    )
                return None, None

    return center_idx, float(height)


def validate_peak_temporal_order(
    peak_data: Dict[str, Dict[str, Union[int, float, None]]],
    verbose: bool = False,
    cycle_idx: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate that detected peaks are in the correct physiological temporal order:
    P < Q < R < S < T (in time, along X-axis).
    
    Parameters
    ----------
    peak_data : dict
        Dictionary containing peak information with keys like "P", "Q", "R", "S", "T".
        Each value should be a dict with at least "center_idx" key.
    verbose : bool
        Enable verbose logging.
    cycle_idx : int, optional
        Cycle index for logging.
    
    Returns
    -------
    Tuple[bool, List[str]]
        - bool: True if ordering is valid, False otherwise.
        - List[str]: List of validation error messages (empty if valid).
    """
    errors = []
    
    # Expected order: P -> Q -> R -> S -> T
    expected_order = ["P", "Q", "R", "S", "T"]
    
    # Extract center indices for detected peaks
    peak_indices = {}
    for comp in expected_order:
        if comp in peak_data:
            center_idx = peak_data[comp].get("center_idx")
            if center_idx is not None and not np.isnan(center_idx):
                peak_indices[comp] = int(center_idx)
    
    # If we have fewer than 2 peaks, ordering check is not applicable
    if len(peak_indices) < 2:
        if verbose:
            print(f"[Cycle {cycle_idx}]: Only {len(peak_indices)} peak(s) detected - skipping temporal order check.")
        return True, []
    
    # Check ordering: each peak should come before the next in expected order
    detected_components = [comp for comp in expected_order if comp in peak_indices]
    
    for i in range(len(detected_components) - 1):
        curr_comp = detected_components[i]
        next_comp = detected_components[i + 1]
        curr_idx = peak_indices[curr_comp]
        next_idx = peak_indices[next_comp]
        
        if curr_idx >= next_idx:
            error_msg = (
                f"Temporal order violation: {curr_comp} (idx={curr_idx}) "
                f"comes after or at {next_comp} (idx={next_idx})"
            )
            errors.append(error_msg)
            if verbose:
                print(f"[Cycle {cycle_idx}]: {error_msg}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_intervals_physiological(
    interval_results: Dict[str, float],
    sampling_rate: float,
    verbose: bool = False,
    cycle_idx: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate that calculated intervals are within physiologically reasonable limits.
    
    Parameters
    ----------
    interval_results : dict
        Dictionary mapping interval names (e.g., 'PR_interval_ms') to values in milliseconds.
    sampling_rate : float
        Sampling rate in Hz.
    verbose : bool
        Enable verbose logging.
    cycle_idx : int, optional
        Cycle index for logging.
    
    Returns
    -------
    Tuple[bool, List[str]]
        - bool: True if all intervals are valid, False otherwise.
        - List[str]: List of validation error messages (empty if valid).
    """
    errors = []
    
    # Physiological limits (in milliseconds) - from intervals.py
    PHYSIOLOGICAL_LIMITS_MS = {
        'PR_interval': (50, 500),
        'PR_segment': (10, 400),
        'QRS_interval': (20, 300),
        'ST_segment': (20, 500),
        'ST_interval': (5, 700),
        'QT_interval': (200, 750),
        'RR_interval': (300, 1800),  # ~200-33 bpm
        'PP_interval': (300, 1800),  # ~200-33 bpm
    }
    
    for interval_name, value in interval_results.items():
        # Skip NaN values (missing intervals are handled elsewhere)
        if np.isnan(value):
            continue
        
        # Extract base name (remove _ms suffix)
        base_name = interval_name.replace('_ms', '')
        
        if base_name in PHYSIOLOGICAL_LIMITS_MS:
            min_ms, max_ms = PHYSIOLOGICAL_LIMITS_MS[base_name]
            
            if not (min_ms <= value <= max_ms):
                error_msg = (
                    f"{interval_name} = {value:.2f} ms is outside physiological range "
                    f"[{min_ms}, {max_ms}] ms"
                )
                errors.append(error_msg)
                if verbose:
                    print(f"[Cycle {cycle_idx}]: {error_msg}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_cycle_physiology(
    peak_data: Dict[str, Dict[str, Union[int, float, None]]],
    interval_results: Dict[str, float],
    sampling_rate: float,
    verbose: bool = False,
    cycle_idx: Optional[int] = None,
) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Comprehensive physiological validation for a single cycle.
    
    Checks:
    1. Peak temporal ordering (P < Q < R < S < T)
    2. Interval physiological limits
    
    Parameters
    ----------
    peak_data : dict
        Dictionary containing peak information.
    interval_results : dict
        Dictionary of calculated intervals.
    sampling_rate : float
        Sampling rate in Hz.
    verbose : bool
        Enable verbose logging.
    cycle_idx : int, optional
        Cycle index for logging.
    
    Returns
    -------
    Tuple[bool, Dict[str, List[str]]]
        - bool: True if cycle passes all validations, False otherwise.
        - Dict[str, List[str]]: Dictionary with 'peak_ordering' and 'intervals' keys,
          each containing a list of error messages (empty if valid).
    """
    all_errors = {
        'peak_ordering': [],
        'intervals': [],
    }
    
    # Validate peak temporal ordering
    order_valid, order_errors = validate_peak_temporal_order(
        peak_data, verbose=verbose, cycle_idx=cycle_idx
    )
    all_errors['peak_ordering'] = order_errors
    
    # Validate intervals
    interval_valid, interval_errors = validate_intervals_physiological(
        interval_results, sampling_rate, verbose=verbose, cycle_idx=cycle_idx
    )
    all_errors['intervals'] = interval_errors
    
    is_valid = order_valid and interval_valid
    
    if verbose and not is_valid:
        print(f"[Cycle {cycle_idx}]: Physiological validation FAILED:")
        if order_errors:
            print(f"  Peak ordering errors: {len(order_errors)}")
        if interval_errors:
            print(f"  Interval errors: {len(interval_errors)}")
    
    return is_valid, all_errors
