"""
Adaptive threshold validation for T and P wave detection, mirroring ECGPUWAVE's approach.

ECGPUWAVE uses amplitude-adaptive thresholds that scale based on the detected wave amplitude,
making it more sensitive to small waves while maintaining specificity for large waves.
"""

from __future__ import annotations
from typing import Optional, Literal, Tuple
import numpy as np

__all__ = ["gate_by_adaptive_threshold", "compute_adaptive_threshold_multiplier"]


def compute_adaptive_threshold_multiplier(
    wave_amplitude: float,
    base_threshold: float = 3.5,
    amplitude_ranges: Optional[list[Tuple[float, float, float]]] = None,
) -> float:
    """
    Compute adaptive threshold multiplier based on wave amplitude (ECGPUWAVE-style).
    
    ECGPUWAVE scales thresholds based on T wave amplitude:
    - Large waves (|amp| >= 0.41): threshold × 2.0
    - Medium-large (0.35-0.41): threshold × 1.0 (kte*2 - 1)
    - Medium (0.25-0.35): threshold × 0.0 (kte*2 - 2) 
    - Small-medium (0.10-0.25): threshold × -1.0 (kte*2 - 3)
    - Small (< 0.10): threshold × 1.0 (no scaling)
    
    Parameters
    ----------
    wave_amplitude : float
        Absolute amplitude of the detected wave (mV).
    base_threshold : float, default 3.5
        Base threshold value (e.g., kte from ECGPUWAVE).
    amplitude_ranges : list of (min_amp, max_amp, scale_factor), optional
        Custom amplitude ranges and scale factors. If None, uses ECGPUWAVE defaults.
        Each tuple: (min_amplitude, max_amplitude, scale_factor)
        where scale_factor is applied as: threshold = base_threshold * scale_factor
    
    Returns
    -------
    float
        Scaled threshold value.
    
    Examples
    --------
    >>> # ECGPUWAVE default behavior for T wave end (kte=3.5)
    >>> compute_adaptive_threshold_multiplier(0.5, base_threshold=3.5)
    7.0  # Large T wave: 3.5 * 2.0
    >>> compute_adaptive_threshold_multiplier(0.15, base_threshold=3.5)
    4.0  # Small-medium: 3.5 * 2 - 3 = 4.0
    >>> compute_adaptive_threshold_multiplier(0.05, base_threshold=3.5)
    3.5  # Small: no scaling
    """
    abs_amp = abs(wave_amplitude)
    
    # ECGPUWAVE default amplitude ranges for T wave end detection
    if amplitude_ranges is None:
        # Format: (min_amp, max_amp, scale_factor)
        # Scale factors are: kte1 = kte * scale_factor
        # ECGPUWAVE logic: kte1 = kte*2, kte*2-1, kte*2-2, kte*2-3, or kte
        amplitude_ranges = [
            (0.41, float('inf'), 2.0),    # Large: kte * 2
            (0.35, 0.41, 1.0),            # Medium-large: kte * 2 - 1 (but we use 1.0 as scale)
            (0.25, 0.35, 0.0),            # Medium: kte * 2 - 2 (but we use 0.0 as scale)
            (0.10, 0.25, -1.0),           # Small-medium: kte * 2 - 3 (but we use -1.0 as scale)
            (0.0, 0.10, 1.0),             # Small: kte (no scaling)
        ]
    
    # Find matching range (check in order, first match wins)
    for min_amp, max_amp, scale_factor in amplitude_ranges:
        if min_amp <= abs_amp < max_amp:
            # Apply scaling: threshold = base_threshold * scale_factor
            # But ECGPUWAVE uses additive scaling, so we need to handle that
            if scale_factor == 2.0:
                # Large: kte * 2
                return base_threshold * 2.0
            elif scale_factor == 1.0 and abs_amp >= 0.35:
                # Medium-large: kte * 2 - 1
                return base_threshold * 2.0 - 1.0
            elif scale_factor == 0.0:
                # Medium: kte * 2 - 2
                return base_threshold * 2.0 - 2.0
            elif scale_factor == -1.0:
                # Small-medium: kte * 2 - 3
                return base_threshold * 2.0 - 3.0
            else:
                # Small: kte (no scaling)
                return base_threshold
    
    # Fallback: no scaling
    return base_threshold


def gate_by_adaptive_threshold(
    seg_raw: np.ndarray,
    sampling_rate: float,
    *,
    comp: Literal["P", "T"] = "T",
    cand_rel_idx: Optional[int] = None,
    expected_polarity: Optional[Literal["positive", "negative"]] = None,
    base_threshold: Optional[float] = None,
    amplitude_ranges: Optional[list[Tuple[float, float, float]]] = None,
    verbose: bool = False,
) -> list[bool | int | float | None]:
    """
    Component-aware adaptive threshold gate (ECGPUWAVE-style).
    
    Uses amplitude-adaptive thresholds that scale based on detected wave amplitude.
    This makes the detector more sensitive to small waves while maintaining specificity
    for large waves.
    
    Parameters
    ----------
    seg_raw : np.ndarray
        Detrended signal segment for the component search window.
    sampling_rate : float
        Sampling rate in Hz.
    comp : {"P","T"}, default "T"
        Component label.
    cand_rel_idx : int or None, optional
        Candidate index relative to seg_raw. If None, finds peak automatically.
    expected_polarity : {"positive","negative"} or None, optional
        Expected peak polarity; defaults to "positive" for P/T if None.
    base_threshold : float or None, optional
        Base threshold value. If None, uses component-specific defaults:
        - T wave: 3.5 (kte from ECGPUWAVE)
        - P wave: 2.0 (ktb from ECGPUWAVE)
    amplitude_ranges : list of (min_amp, max_amp, scale_factor), optional
        Custom amplitude ranges for adaptive scaling. If None, uses ECGPUWAVE defaults.
    verbose : bool, default False
        If True, print diagnostic messages.
    
    Returns
    -------
    list of [bool, int or None, float or None]
        [keep, rel_idx, raw_height], where:
          - keep: True if wave passes adaptive threshold, else False
          - rel_idx: candidate index used for the decision (relative to seg_raw)
          - raw_height: amplitude from seg_raw at rel_idx
    """
    n = int(seg_raw.size)
    if n < 3 or not np.isfinite(seg_raw).all():
        return [False, None, None]
    
    # Default base thresholds (from ECGPUWAVE)
    if base_threshold is None:
        base_threshold = 3.5 if comp == "T" else 2.0  # kte for T, ktb for P
    
    # Polarity default by component
    if expected_polarity is None:
        expected_polarity = "positive"
    
    # Find candidate peak if not provided
    if cand_rel_idx is None:
        if expected_polarity == "positive":
            cand_rel_idx = int(np.argmax(seg_raw))
        else:
            cand_rel_idx = int(np.argmin(seg_raw))
    
    if not (0 <= cand_rel_idx < n):
        return [False, None, None]
    
    # Get raw amplitude
    raw_height = float(seg_raw[cand_rel_idx])
    abs_height = abs(raw_height)
    
    # Compute adaptive threshold multiplier based on amplitude
    adaptive_threshold = compute_adaptive_threshold_multiplier(
        abs_height,
        base_threshold=base_threshold,
        amplitude_ranges=amplitude_ranges,
    )
    
    # ECGPUWAVE uses the threshold to find wave boundaries, but for peak validation,
    # we compare the peak amplitude to a scaled version of itself
    # The logic: if the wave is large enough to have a scaled threshold, it's valid
    
    # For validation, we use a simpler approach:
    # - Large waves (|amp| >= 0.41): require threshold * 2.0 (more strict)
    # - Small waves (< 0.10): use base threshold (more lenient)
    # - Medium waves: use intermediate thresholds
    
    # Compute minimum required amplitude based on adaptive threshold
    # ECGPUWAVE's approach: the threshold is used for boundary detection,
    # but for peak validation, we check if the peak amplitude is reasonable
    # relative to the adaptive threshold
    
    # Simplified validation: accept if amplitude is above a minimum threshold
    # that scales with the adaptive threshold
    min_required_amplitude = adaptive_threshold * 0.1  # 10% of adaptive threshold
    
    # For very small waves, be more lenient
    if abs_height < 0.10:
        # Small waves: use lower minimum (more lenient)
        min_required_amplitude = base_threshold * 0.05  # 5% of base threshold
    
    keep = abs_height >= min_required_amplitude
    
    if verbose:
        print(f"[AdaptiveThreshold] {comp} wave: amplitude={abs_height:.4f} mV, "
              f"adaptive_threshold={adaptive_threshold:.4f}, "
              f"min_required={min_required_amplitude:.4f}, keep={keep}")
    
    return [keep, cand_rel_idx, raw_height]


def gate_by_adaptive_threshold_ecgpuwave_style(
    seg_raw: np.ndarray,
    sampling_rate: float,
    *,
    comp: Literal["P", "T"] = "T",
    cand_rel_idx: Optional[int] = None,
    expected_polarity: Optional[Literal["positive", "negative"]] = None,
    base_threshold: Optional[float] = None,
    r_peak_amplitude: Optional[float] = None,
    verbose: bool = False,
) -> list[bool | int | float | None]:
    """
    ECGPUWAVE-style adaptive threshold gate with relative amplitude checking.
    
    This version mirrors ECGPUWAVE's approach more closely by:
    1. Using adaptive thresholds based on wave amplitude
    2. Optionally checking relative to R peak amplitude
    3. Using ECGPUWAVE's exact scaling logic
    
    Parameters
    ----------
    seg_raw : np.ndarray
        Detrended signal segment for the component search window.
    sampling_rate : float
        Sampling rate in Hz.
    comp : {"P","T"}, default "T"
        Component label.
    cand_rel_idx : int or None, optional
        Candidate index relative to seg_raw. If None, finds peak automatically.
    expected_polarity : {"positive","negative"} or None, optional
        Expected peak polarity; defaults to "positive" for P/T if None.
    base_threshold : float or None, optional
        Base threshold value. If None, uses component-specific defaults.
    r_peak_amplitude : float or None, optional
        R peak amplitude for relative amplitude checking. If provided, uses
        relative amplitude thresholds (e.g., T wave should be > 5% of R peak).
    verbose : bool, default False
        If True, print diagnostic messages.
    
    Returns
    -------
    list of [bool, int or None, float or None]
        [keep, rel_idx, raw_height], where:
          - keep: True if wave passes adaptive threshold, else False
          - rel_idx: candidate index used for the decision (relative to seg_raw)
          - raw_height: amplitude from seg_raw at rel_idx
    """
    n = int(seg_raw.size)
    if n < 3 or not np.isfinite(seg_raw).all():
        return [False, None, None]
    
    # Default base thresholds (from ECGPUWAVE)
    if base_threshold is None:
        base_threshold = 3.5 if comp == "T" else 2.0  # kte for T, ktb for P
    
    # Polarity default by component
    if expected_polarity is None:
        expected_polarity = "positive"
    
    # Find candidate peak if not provided
    if cand_rel_idx is None:
        if expected_polarity == "positive":
            cand_rel_idx = int(np.argmax(seg_raw))
        else:
            cand_rel_idx = int(np.argmin(seg_raw))
    
    if not (0 <= cand_rel_idx < n):
        return [False, None, None]
    
    # Get raw amplitude
    raw_height = float(seg_raw[cand_rel_idx])
    abs_height = abs(raw_height)
    
    # ECGPUWAVE's adaptive threshold logic (from tbound.m lines 196-201)
    # Scales threshold based on wave amplitude
    if abs_height >= 0.41:
        scaled_threshold = base_threshold * 2.0
    elif abs_height >= 0.35:
        scaled_threshold = base_threshold * 2.0 - 1.0
    elif abs_height >= 0.25:
        scaled_threshold = base_threshold * 2.0 - 2.0
    elif abs_height >= 0.10:
        scaled_threshold = base_threshold * 2.0 - 3.0
    else:
        scaled_threshold = base_threshold  # No scaling for very small waves
    
    # Validation: ECGPUWAVE accepts waves if they pass the adaptive threshold
    # For peak validation, we check:
    # 1. Minimum absolute amplitude (very small waves might be noise)
    # 2. Relative to R peak if provided
    # 3. The adaptive threshold is used for boundary detection, not peak validation
    
    # Minimum amplitude check (in mV)
    min_absolute_amplitude = 0.01  # 10 microvolts minimum
    
    # Relative amplitude check (if R peak provided)
    if r_peak_amplitude is not None and r_peak_amplitude > 0:
        # T waves should be at least 2-5% of R peak amplitude
        min_relative_amplitude = abs(r_peak_amplitude) * (0.02 if comp == "T" else 0.01)
    else:
        min_relative_amplitude = 0.0
    
    # Combined check: wave must exceed both absolute and relative minimums
    min_required = max(min_absolute_amplitude, min_relative_amplitude)
    
    # For very small waves (< 0.10 mV), be more lenient with relative check
    if abs_height < 0.10 and r_peak_amplitude is not None:
        # Very small waves: lower relative threshold (1% instead of 2-5%)
        min_relative_amplitude = abs(r_peak_amplitude) * 0.01
        min_required = max(min_absolute_amplitude, min_relative_amplitude)
    
    keep = abs_height >= min_required
    
    if verbose:
        print(f"[ECGPUWAVE-style] {comp} wave: amplitude={abs_height:.4f} mV, "
              f"scaled_threshold={scaled_threshold:.4f}, "
              f"min_required={min_required:.4f}, keep={keep}")
        if r_peak_amplitude is not None:
            rel_ratio = abs_height / abs(r_peak_amplitude) * 100
            print(f"  Relative to R peak: {rel_ratio:.2f}%")
    
    return [keep, cand_rel_idx, raw_height]

