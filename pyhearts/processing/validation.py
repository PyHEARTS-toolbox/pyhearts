from __future__ import annotations

from functools import partial
from typing import Dict, Mapping, Optional, Tuple, Literal, Union, List

import numpy as np
from pyhearts.config import ProcessCycleConfig

# Type aliases (Python 3.9 compatible)
Peak = Tuple[Optional[int], Optional[float]]
Peaks = Mapping[str, Peak]
ValidatedPeaks = Dict[str, Peak]


def validate_peaks(
    peaks: Peaks,
    r_center_idx: int | None,
    r_height: float | None,
    sampling_rate: float | None,
    verbose: bool,
    cycle_idx: int | None,
    *,
    cfg: ProcessCycleConfig | None = None,
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

    validate_one = partial(
        log_peak_result,
        r_center_idx=r_center_idx,
        r_height=r_height,
        sampling_rate=sampling_rate,
        verbose=verbose,
        cycle_idx=cycle_idx,
        cfg=cfg,
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
) -> Peak:
    """
    Apply polarity and relative-amplitude checks for a single component.
    """
    # Missing or invalid amplitude
    if center_idx is None or height is None or not np.isfinite(height):
        if verbose:
            print(f"[Cycle {cycle_idx}]: {comp} peak not found.")
        return None, None

    # Polarity checks
    # Step 5: Allow negative P-waves and T-waves for inverted leads
    # Negative T-waves are physiologically valid (common in certain leads and conditions)
    if expected_polarity == "peak" and height < 0 and comp in ("P", "T"):
        # For both P-waves and T-waves, allow negative (inverted leads)
        if verbose:
            print(f"[Cycle {cycle_idx}]: {comp}-wave is negative (inverted lead), accepting anyway.")
        # Accept negative P/T-wave (inverted lead) - continue to amplitude check
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

        if min_ratio is not None and abs(float(height)) < min_ratio * r_abs:
            if verbose:
                need = min_ratio * r_abs
                print(
                    f"[Cycle {cycle_idx}]: {comp} too small "
                    f"({abs(float(height)):.3f} < {need:.3f})."
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
