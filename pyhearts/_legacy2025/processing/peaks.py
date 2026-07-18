from typing import Optional, Tuple
import numpy as np


def find_peaks(
    signal: np.ndarray,
    xs: np.ndarray,
    start_idx: int,
    end_idx: int,
    mode: str,
    verbose: bool = True,
    label: Optional[str] = None,
    cycle_idx: Optional[int] = None
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

    # Find relative index within the segment
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

