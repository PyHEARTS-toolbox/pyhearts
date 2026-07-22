from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def find_peak_derivative_based(
    signal: np.ndarray,
    start_idx: int,
    end_idx: int,
    polarity: str = "positive",
    verbose: bool = False,
    label: Optional[str] = None,
    cycle_idx: Optional[int] = None
) -> Tuple[Optional[int], Optional[float]]:
    """
    Find peak using derivative-based method.
    
    Finds peaks by locating zero-crossings in the first derivative:
    - Positive peaks: derivative goes from positive to negative (rising to falling)
    - Negative peaks: derivative goes from negative to positive (falling to rising)
    
    This method is more robust to baseline drift than simple argmax/argmin.
    
    Parameters
    ----------
    signal : np.ndarray
        1D ECG signal array.
    start_idx : int
        Start index of search window (inclusive).
    end_idx : int
        End index of search window (exclusive).
    polarity : {'positive', 'negative'}
        Whether to search for positive or negative peaks.
    verbose : bool, optional
        If True, print diagnostic messages.
    label : str, optional
        Name of the peak for logging.
    cycle_idx : int, optional
        Current cycle index for logging.
    
    Returns
    -------
    idx_absolute : int or None
        Absolute index of the detected peak.
    amplitude : float or None
        Amplitude of the detected peak.
    """
    if polarity not in {"positive", "negative"}:
        raise ValueError("polarity must be 'positive' or 'negative'")
    
    start_idx = nearest_sample_index(start_idx, len(signal))
    end_idx = nearest_sample_index(end_idx, len(signal))

    # Validate search window
    if (
        start_idx >= end_idx
        or start_idx < 0
        or end_idx > len(signal)
        or end_idx - start_idx < 3  # Need at least 3 samples for derivative
    ):
        if verbose and label:
            print(f"[Cycle {cycle_idx}]: Invalid segment for {label} peak derivative detection (start={start_idx}, end={end_idx})")
        return None, None
    
    segment = signal[start_idx:end_idx]
    
    # Compute first derivative
    deriv = np.diff(segment)
    
    if len(deriv) < 2:
        return None, None
    
    if polarity == "positive":
        # For positive peaks: find where derivative goes from positive to negative
        # (signal transitions from rising to falling)
        sign_changes = np.diff(np.sign(deriv))
        zero_crossings = np.where(sign_changes < 0)[0]  # negative transition
    else:
        # For negative peaks: find where derivative goes from negative to positive
        # (signal transitions from falling to rising)
        sign_changes = np.diff(np.sign(deriv))
        zero_crossings = np.where(sign_changes > 0)[0]  # positive transition
    
    # If no zero-crossings found, try more lenient detection:
    # Look for local maxima/minima in the derivative (not just zero-crossings)
    # This helps detect small or noisy P waves
    if len(zero_crossings) == 0:
        # Fallback: find local extrema in derivative
        # For positive peaks: find where derivative is maximum (most positive before going negative)
        # For negative peaks: find where derivative is minimum (most negative before going positive)
        if polarity == "positive":
            # Find local maxima in derivative (where signal is rising fastest)
            # These indicate potential positive peaks
            local_maxima = []
            for i in range(1, len(deriv) - 1):
                if deriv[i] > deriv[i-1] and deriv[i] > deriv[i+1] and deriv[i] > 0:
                    local_maxima.append(i)
            if len(local_maxima) > 0:
                # Use the largest local maximum
                max_deriv_idx = local_maxima[np.argmax([deriv[i] for i in local_maxima])]
                zero_crossings = np.array([max_deriv_idx])
        else:
            # Find local minima in derivative (where signal is falling fastest)
            # These indicate potential negative peaks
            local_minima = []
            for i in range(1, len(deriv) - 1):
                if deriv[i] < deriv[i-1] and deriv[i] < deriv[i+1] and deriv[i] < 0:
                    local_minima.append(i)
            if len(local_minima) > 0:
                # Use the smallest local minimum
                min_deriv_idx = local_minima[np.argmin([deriv[i] for i in local_minima])]
                zero_crossings = np.array([min_deriv_idx])
    
    if len(zero_crossings) == 0:
        if verbose and label:
            print(f"[Cycle {cycle_idx}]: No {polarity} peaks found via derivative method for {label}")
        return None, None
    
    # Find peak at each zero-crossing (peak is at zero_crossing + 1 in original signal)
    # Choose the most prominent peak (largest absolute amplitude)
    best_peak_idx = None
    best_peak_amp = None
    
    for zc in zero_crossings:
        peak_idx_rel = zc + 1  # Peak is one sample after zero-crossing
        if peak_idx_rel < len(segment):
            peak_idx_abs = start_idx + peak_idx_rel
            peak_amp = signal[peak_idx_abs]
            
            if best_peak_amp is None or abs(peak_amp) > abs(best_peak_amp):
                best_peak_idx = peak_idx_abs
                best_peak_amp = peak_amp
    
    if verbose and label and best_peak_idx is not None:
        print(f"[Cycle {cycle_idx}]: Found {label} peak via derivative at index {best_peak_idx} with amplitude {best_peak_amp:.6f}")
    
    return best_peak_idx, best_peak_amp


def refine_peak_parabolic(
    signal: np.ndarray,
    peak_idx: int
) -> float:
    """
    Refine peak position using parabolic interpolation for sub-sample accuracy.
    
    Fits a parabola through three points around the peak to estimate
    the true peak position between samples.
    
    Parameters
    ----------
    signal : np.ndarray
        1D signal array.
    peak_idx : int
        Initial peak index (must be valid and not at edges).
    
    Returns
    -------
    refined_idx : float
        Refined peak index (may be fractional for sub-sample accuracy).
    """
    if peak_idx <= 0 or peak_idx >= len(signal) - 1:
        return float(peak_idx)
    
    # Get three points around peak
    y1, y2, y3 = signal[peak_idx-1], signal[peak_idx], signal[peak_idx+1]
    
    # Parabolic interpolation: y = ax^2 + bx + c
    # Through points (-1, y1), (0, y2), (1, y3)
    # Vertex (peak) at x = -b/(2a)
    
    # Solve for coefficients
    # y1 = a(-1)^2 + b(-1) + c = a - b + c
    # y2 = a(0)^2 + b(0) + c = c
    # y3 = a(1)^2 + b(1) + c = a + b + c
    # 
    # So: c = y2
    #     a - b = y1 - y2
    #     a + b = y3 - y2
    #     => 2a = (y1 - y2) + (y3 - y2) = y1 + y3 - 2y2
    #     => a = (y1 + y3 - 2y2) / 2
    #     => b = (y3 - y2) - a = (y3 - y2) - (y1 + y3 - 2y2)/2 = (y3 - y1)/2
    
    a = (y1 + y3 - 2*y2) / 2.0
    if abs(a) < 1e-10:  # Avoid division by zero (flat signal)
        return float(peak_idx)
    
    # Vertex offset from center sample (x = 0 corresponds to peak_idx)
    offset = (y1 - y3) / (2.0 * a)
    
    # Clamp offset to reasonable range (should be within [-0.5, 0.5] for valid parabola)
    offset = np.clip(offset, -0.5, 0.5)
    
    return peak_idx + offset


def nearest_sample_index(rel_idx: float, n: int) -> int:
    """Round a cycle-relative index to the nearest valid sample index."""
    if n <= 0:
        return 0
    return int(np.clip(np.round(float(rel_idx)), 0, n - 1))


def sample_at_fractional_index(signal: np.ndarray, rel_idx: float) -> float:
    """Linear interpolation of *signal* at a fractional cycle-relative index."""
    if signal.size == 0:
        return np.nan
    rel = float(np.clip(rel_idx, 0.0, float(signal.size - 1)))
    i0 = int(np.floor(rel))
    i1 = min(i0 + 1, signal.size - 1)
    frac = rel - i0
    return float(signal[i0] * (1.0 - frac) + signal[i1] * frac)


def refine_peak_index_subsample(
    signal: np.ndarray,
    peak_idx: int | float,
    *,
    enabled: bool = True,
) -> float:
    """Return fractional cycle-relative peak index (parabolic refinement when enabled)."""
    if peak_idx is None or (isinstance(peak_idx, float) and np.isnan(peak_idx)):
        return np.nan
    rel = float(peak_idx)
    if not enabled or signal.size < 3:
        return rel
    anchor = int(np.clip(np.round(rel), 1, signal.size - 2))
    return refine_peak_parabolic(signal, anchor)


def cycle_rel_to_global_sample(
    rel_idx: float,
    xs_samples: np.ndarray,
    signal: np.ndarray | None = None,
    *,
    refine_subsample: bool = False,
) -> float:
    """
    Map a cycle-relative peak index to a global sample index.

    When ``refine_subsample`` is True, applies parabolic refinement on *signal*
    before linear interpolation across ``xs_samples``.
    """
    if rel_idx is None or (isinstance(rel_idx, float) and np.isnan(rel_idx)):
        return np.nan
    rel = refine_peak_index_subsample(
        signal if signal is not None else np.zeros(1),
        rel_idx,
        enabled=refine_subsample and signal is not None and signal.size >= 3,
    )
    n = len(xs_samples)
    if n == 0:
        return np.nan
    if n == 1:
        return float(xs_samples[0])
    rel = float(np.clip(rel, 0.0, float(n - 1)))
    i0 = int(np.floor(rel))
    i1 = min(i0 + 1, n - 1)
    frac = rel - i0
    return float(xs_samples[i0]) * (1.0 - frac) + float(xs_samples[i1]) * frac


def find_peaks(
    signal: np.ndarray,
    xs: np.ndarray,
    start_idx: int,
    end_idx: int,
    mode: str,
    verbose: bool = True,
    label: Optional[str] = None,
    cycle_idx: Optional[int] = None,
    use_derivative: bool = False,
    refine_subsample: bool = False,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Find a local min or max peak in a segment of the signal.

    Parameters
    ----------
    signal : np.ndarray
        1D ECG (or other) signal array.
    xs : np.ndarray
        Corresponding x-axis values (e.g., time or samples).
    start_idx : int
        Start index of search window (inclusive).
    end_idx : int
        End index of search window (exclusive).
    mode : {'min', 'max'}
        Whether to search for a local minimum or maximum.
    verbose : bool, optional
        If True, print diagnostic messages.
    label : str, optional
        Name of the peak for logging.
    cycle_idx : int, optional
        Current cycle index for logging.

    Returns
    -------
    idx_absolute : float or None
        Absolute index of the detected peak (fractional when ``refine_subsample``).
    amplitude : float or None
        Amplitude of the detected peak.
    center : float or None
        Corresponding x-axis value of the detected peak.
    """
    if mode not in {"min", "max"}:
        raise ValueError("mode must be 'min' or 'max'")

    # Validate search window - handle None values
    if start_idx is None or end_idx is None:
        return None, None, None

    start_idx = nearest_sample_index(start_idx, len(signal))
    end_idx = nearest_sample_index(end_idx, len(signal))
    
    # Validate search window
    if (
        start_idx >= end_idx
        or start_idx < 0
        or end_idx > len(signal)
        or end_idx - start_idx == 0
    ):
        if verbose and label:
            print(f"[Cycle {cycle_idx}]: Invalid segment for {label} peak (start={start_idx}, end={end_idx})")
        return None, None, None

    # Use derivative-based detection if requested
    if use_derivative:
        polarity = "positive" if mode == "max" else "negative"
        idx_absolute, amplitude = find_peak_derivative_based(
            signal, start_idx, end_idx, polarity, verbose, label, cycle_idx
        )
        if idx_absolute is not None:
            if refine_subsample:
                idx_f = refine_peak_index_subsample(
                    signal, idx_absolute, enabled=True
                )
                amplitude = sample_at_fractional_index(signal, idx_f)
                idx_absolute = nearest_sample_index(idx_f, len(signal))
            else:
                idx_absolute = int(idx_absolute)
                amplitude = signal[idx_absolute]
            center = float(xs[idx_absolute]) if idx_absolute < len(xs) else float(xs[-1])
            return idx_absolute, amplitude, center
        else:
            return None, None, None
    
    # Original method: simple argmax/argmin
    idx_relative = (
        np.argmin(signal[start_idx:end_idx])
        if mode == "min"
        else np.argmax(signal[start_idx:end_idx])
    )
    idx_absolute = start_idx + idx_relative
    if refine_subsample:
        idx_f = refine_peak_index_subsample(signal, idx_absolute, enabled=True)
        amplitude = sample_at_fractional_index(signal, idx_f)
        idx_absolute = nearest_sample_index(idx_f, len(signal))
    else:
        idx_absolute = int(idx_absolute)
        amplitude = signal[idx_absolute]
    center = float(xs[idx_absolute])

    if verbose and label:
        print(f"[Cycle {cycle_idx}]: Found {label} peak at index {idx_absolute} with amplitude {amplitude:.6f}")

    return idx_absolute, amplitude, center


def global_index_to_cycle_relative(global_idx: int, xs_samples: np.ndarray) -> int:
    """Map a full-signal sample index to a cycle-relative index."""
    matches = np.where(xs_samples == global_idx)[0]
    if len(matches) > 0:
        return int(matches[0])
    return int(np.argmin(np.abs(xs_samples - global_idx)))


def refine_r_peak_near_anchor(
    signal: np.ndarray,
    anchor_idx: int,
    sampling_rate: float,
    half_window_ms: float = 20.0,
    refine_mode: str = "derivative",
) -> int:
    """
    Refine an R-peak index near the anchor.

    ``derivative``: derivative zero-crossing (classic detector mapping).
    ``extremum``: local max/min on the signal (TRP-style, ±window on raw/epoch).
    """
    n = len(signal)
    if n < 3:
        return int(np.clip(anchor_idx, 0, max(0, n - 1)))

    anchor_idx = int(np.clip(anchor_idx, 0, n - 1))
    half_w = max(1, int(round(half_window_ms * sampling_rate / 1000.0)))
    start = max(0, anchor_idx - half_w)
    end = min(n, anchor_idx + half_w + 1)
    segment = signal[start:end]
    if segment.size == 0:
        return anchor_idx

    if refine_mode == "extremum":
        if float(signal[anchor_idx]) >= 0.0:
            return start + int(np.argmax(segment))
        return start + int(np.argmin(segment))

    derivative = np.gradient(signal.astype(float))
    end = min(len(derivative), anchor_idx + half_w + 1)
    local_deriv = derivative[start:end]
    if len(local_deriv) < 2:
        return anchor_idx

    sign_changes = np.diff(np.sign(local_deriv))
    zero_crossings = np.where(np.abs(sign_changes) > 0)[0]
    if len(zero_crossings) > 0:
        anchor_rel = anchor_idx - start
        zc_idx = start + zero_crossings[
            np.argmin(np.abs(zero_crossings - anchor_rel))
        ]
        signal_peak_idx = zc_idx + 1
        if 0 <= signal_peak_idx < n:
            return signal_peak_idx

    segment = signal[start:end]
    if segment.size == 0:
        return anchor_idx
    return start + int(np.argmax(np.abs(segment)))

