"""Crop ECG signals for fast iteration (lite mode, benchmarks)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def crop_signal(
    signal: np.ndarray,
    fs: float,
    *,
    fraction: Optional[float] = None,
    duration_s: Optional[float] = None,
    from_end: bool = False,
) -> Tuple[np.ndarray, int, int]:
    """
    Crop *signal* to a prefix (default) or suffix segment.

    Parameters
    ----------
    signal : np.ndarray
        1-D ECG samples.
    fs : float
        Sampling rate (Hz).
    fraction : float, optional
        Fraction of the recording length (0–1).
    duration_s : float, optional
        Duration in seconds; overrides *fraction* when set.
    from_end : bool
        If False, keep the first segment; if True, keep the last segment.

    Returns
    -------
    cropped : np.ndarray
    start_sample : int
        Inclusive start index in the original signal.
    end_sample : int
        Exclusive end index in the original signal.
    """
    n = len(signal)
    if n == 0:
        return signal, 0, 0

    if duration_s is not None:
        seg_len = min(n, max(1, int(round(float(duration_s) * fs))))
    elif fraction is not None:
        seg_len = min(n, max(1, int(round(n * float(fraction)))))
    else:
        return signal, 0, n

    if from_end:
        start = max(0, n - seg_len)
        end = n
    else:
        start = 0
        end = seg_len

    return signal[start:end], start, end
