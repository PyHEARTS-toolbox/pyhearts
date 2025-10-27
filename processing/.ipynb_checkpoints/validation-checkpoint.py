from functools import partial
from typing import Mapping, Optional, Tuple, Literal

import numpy as np
from pyhearts.config import ProcessCycleConfig

# Type aliases
Peak = tuple[int | None, float | None]
Peaks = Mapping[str, Peak]
ValidatedPeaks = dict[str, Peak]


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
    if expected_polarity == "peak" and height < 0 and comp in ("P", "T"):
        if verbose:
            print(f"[Cycle {cycle_idx}]: {comp} polarity invalid (expected positive).")
        return None, None
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
