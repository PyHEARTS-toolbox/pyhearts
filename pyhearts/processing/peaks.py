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
    Find peak using derivative-based method (ECGPUWAVE-style).
    
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


def find_peaks(
    signal: np.ndarray,
    xs: np.ndarray,
    start_idx: int,
    end_idx: int,
    mode: str,
    verbose: bool = True,
    label: Optional[str] = None,
    cycle_idx: Optional[int] = None,
    use_derivative: bool = False
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
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
    idx_absolute : int or None
        Absolute index of the detected peak.
    amplitude : float or None
        Amplitude of the detected peak.
    center : float or None
        Corresponding x-axis value of the detected peak.
    """
    if mode not in {"min", "max"}:
        raise ValueError("mode must be 'min' or 'max'")

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

    # Use derivative-based detection if requested (ECGPUWAVE-style)
    if use_derivative:
        polarity = "positive" if mode == "max" else "negative"
        idx_absolute, amplitude = find_peak_derivative_based(
            signal, start_idx, end_idx, polarity, verbose, label, cycle_idx
        )
        if idx_absolute is not None:
            # Optionally refine with parabolic interpolation
            refined_idx = refine_peak_parabolic(signal, idx_absolute)
            idx_absolute = int(np.round(refined_idx))
            amplitude = signal[idx_absolute]
            center = xs[idx_absolute] if idx_absolute < len(xs) else xs[-1]
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
    amplitude = signal[idx_absolute]
    center = xs[idx_absolute]

    if verbose and label:
        print(f"[Cycle {cycle_idx}]: Found {label} peak at index {idx_absolute} with amplitude {amplitude:.6f}")

    return idx_absolute, amplitude, center

