from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np


def clip_guess_to_bounds(
    guess: Sequence[float] | np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    *,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Clip ``curve_fit`` initial guess into ``(lower, upper)`` and repair flat bounds.

    SciPy ``trf`` rejects any ``p0`` element outside the open interval implied by
    the bounds. Tight ``bound_factor`` values (and int-truncated σ limits) can put
    a valid seed on or outside an edge; this is a defect repair, not tuning.
    """
    lo = np.asarray(bounds[0], dtype=float).copy()
    hi = np.asarray(bounds[1], dtype=float).copy()
    p0 = np.asarray(guess, dtype=float).copy()
    if lo.shape != hi.shape or p0.shape != lo.shape:
        raise ValueError(
            f"guess/bounds shape mismatch: guess={p0.shape}, lo={lo.shape}, hi={hi.shape}"
        )

    # Ensure a usable open interval on every parameter.
    flat = ~np.isfinite(lo) | ~np.isfinite(hi) | (hi <= lo)
    if np.any(flat):
        mid = np.where(np.isfinite(p0), p0, 0.0)
        width = np.maximum(np.abs(mid) * 0.05, 1e-3)
        lo = np.where(flat, mid - width, lo)
        hi = np.where(flat, mid + width, hi)

    span = hi - lo
    # Keep epsilon inside the interval even for very narrow spans.
    eps = np.minimum(epsilon, 0.25 * span)
    eps = np.where(eps > 0, eps, epsilon)
    return np.clip(p0, lo + eps, hi - eps)


def calc_bounds(
    center: int,
    height: float,
    std: int,
    bound_factor: float,
    flip_height: bool = False
) -> Tuple[List[Union[int, float]], List[Union[int, float]]]:
    """
    Calculate lower and upper bounds for a Gaussian component's features: 
    center (mu), height (amplitude), and standard deviation (sigma).

    Ensures lower < upper for all bounds. Height bounds respect flip_height logic
    and allow for negative peaks.

    Parameters
    ----------
    center : int
        Center position of the Gaussian.
    height : float
        Amplitude of the Gaussian peak.
    std : int
        Standard deviation of the Gaussian.
    bound_factor : float, optional
        Proportion to scale bounds around each feature (default is 0.2).
    flip_height : bool, optional
        If True, reverses direction of height bounds (for negative peaks).

    Returns
    -------
    Tuple[List[Union[int, float]], List[Union[int, float]]]
        Lower bounds and upper bounds, each as a list of [center, height, std].
    """
    delta = int(bound_factor * std)

    # --- Center bounds ---
    center_lower = center - delta
    center_upper = center + delta
    if center_upper - center_lower < 2:
        offset = ((2 - (center_upper - center_lower)) // 2) + 1
        center_lower -= offset
        center_upper += offset

    # --- Height bounds ---
    factor_low = 1 - bound_factor
    factor_high = 1 + bound_factor

    if flip_height ^ (height < 0):  # XOR for flipping
        lower_height = height * factor_high
        upper_height = height * factor_low
    else:
        lower_height = height * factor_low
        upper_height = height * factor_high

    # Ensure lower_height < upper_height
    if lower_height == upper_height:
        epsilon = 1e-6 if height >= 0 else -1e-6
        upper_height += epsilon

    elif lower_height > upper_height:
        lower_height, upper_height = upper_height, lower_height

    # --- Std bounds ---
    lower_std = max(int(std * factor_low), 1)
    upper_std = int(std * factor_high)

    # Enforce wider bounds for very small std values
    if std < 2:
        lower_std = 1
        upper_std = 5  # ensures upper - lower = 4

    # General case: Ensure lower_std < upper_std and a minimum width
    elif lower_std >= upper_std or (upper_std - lower_std < 1):
        if lower_std == 1:
            upper_std = max(upper_std, 2)
        else:
            lower_std = max(lower_std - 1, 1)
            upper_std = lower_std + 1

    return [center_lower, lower_height, lower_std], [center_upper, upper_height, upper_std]

