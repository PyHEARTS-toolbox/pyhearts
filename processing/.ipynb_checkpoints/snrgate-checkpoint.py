# --- helpers/snrgate.py ---
from __future__ import annotations

from typing import Optional, Literal, Dict, TYPE_CHECKING

import numpy as np

try:
    from scipy.signal import savgol_filter
    _HAVE_SG = True
except Exception:
    _HAVE_SG = False

if TYPE_CHECKING:  # avoid runtime import cycles
    from pyhearts.config import ProcessCycleConfig
else:
    ProcessCycleConfig = object  # type: ignore[assignment]

__all__ = ["gate_by_local_mad", "half_fwhm_samples"]


def _rolling_mad(x: np.ndarray, win: int) -> np.ndarray:
    """
    Compute a rolling MAD proxy: mean abs deviation * 1.4826 (Gaussian σ equiv).

    Parameters
    ----------
    x : np.ndarray
        Input signal segment.
    win : int
        Window length (will be clamped to >=3 and forced odd).

    Returns
    -------
    np.ndarray
        Rolling MAD series, same length as x.
    """
    win = max(3, int(win))
    if win % 2 == 0:
        win -= 1
    k = np.ones(win, dtype=float) / win
    m = np.convolve(x, k, mode="same")
    return 1.4826 * np.convolve(np.abs(x - m), k, mode="same")


def half_fwhm_samples(sig: np.ndarray, peak_idx: int) -> int:
    """
    Estimate ~half the FWHM (in samples) around a candidate index using |signal|.

    Parameters
    ----------
    sig : np.ndarray
        Signal array.
    peak_idx : int
        Candidate peak index (0-based).

    Returns
    -------
    int
        Half the FWHM in samples (>=1).
    """
    n = int(sig.size)
    if not (0 <= peak_idx < n):
        return 1
    amp = float(abs(sig[peak_idx]))
    if not np.isfinite(amp) or amp <= 0:
        return 1
    half_amp = 0.5 * amp

    L = peak_idx
    while L > 0 and abs(sig[L]) >= half_amp:
        L -= 1
    R = peak_idx
    while R < n - 1 and abs(sig[R]) >= half_amp:
        R += 1
    return max(1, (R - L) // 2)


def _cfg_lookup(
    table: Optional[Dict[str, float | int | bool]],
    comp: str,
    default_val: float | int | bool,
) -> float | int | bool:
    """Return component-specific value from cfg dict with a safe fallback."""
    if isinstance(table, dict) and comp in table:
        return table[comp]
    return default_val


def gate_by_local_mad(
    seg_raw: np.ndarray,
    sampling_rate: float,
    *,
    comp: Literal["P", "T"] = "T",
    cand_rel_idx: Optional[int] = None,
    expected_polarity: Optional[Literal["positive", "negative"]] = None,
    cfg: Optional[ProcessCycleConfig] = None,
    baseline_mode: Literal["rolling", "median"] = "rolling",
) -> list[bool | int | float | None]:
    """
    Component-aware SNR gate using a local MAD estimate.

    Parameters
    ----------
    seg_raw : np.ndarray
        Detrended signal segment for the component search window (raw—unsmoothed).
    sampling_rate : float
        Sampling rate in Hz.
    comp : {"P","T"}, default "T"
        Component label; controls defaults from cfg.
    cand_rel_idx : int or None, optional
        Candidate index relative to seg_raw. If None and smoothing is enabled, the
        candidate is localized on the smoothed segment.
    expected_polarity : {"positive","negative"} or None, optional
        Expected peak polarity; defaults to "positive" for P/T if None.
    cfg : ProcessCycleConfig or None, optional
        Configuration providing:
          - snr_mad_multiplier: dict[str,float]  (e.g., {"P":2.0, "T":2.0})
          - snr_exclusion_ms:   dict[str,int]    (e.g., {"P":0,   "T":20})
              * 0 → exclude by half-FWHM; >0 → exclude that many ms
          - snr_apply_savgol:   dict[str,bool]   (e.g., {"P":False, "T":True})
          - savgol_window_pts:  int              (odd, e.g., 7)
          - savgol_polyorder:   int              (e.g., 3)
    baseline_mode : {"rolling","median"}, default "rolling"
        Method to estimate local MAD.

    Returns
    -------
    list of [bool, int or None, float or None]
        [keep, rel_idx, raw_height], where:
          - keep: True if |height| ≥ k * MAD, else False
          - rel_idx: candidate index used for the decision (relative to seg_raw)
          - raw_height: amplitude from seg_raw at rel_idx (not smoothed)
    """
    n = int(seg_raw.size)
    if n < 3 or not np.isfinite(seg_raw).all():
        return [False, None, None]

    # Resolve cfg-driven knobs with safe defaults
    k = float(_cfg_lookup(getattr(cfg, "snr_mad_multiplier", None), comp, 2.0))
    excl_ms = int(_cfg_lookup(getattr(cfg, "snr_exclusion_ms", None), comp, 20 if comp == "T" else 0))
    use_sg = bool(_cfg_lookup(getattr(cfg, "snr_apply_savgol", None), comp, comp == "T"))
    sg_win = int(getattr(cfg, "savgol_window_pts", 7) or 7)
    sg_poly = int(getattr(cfg, "savgol_polyorder", 3) or 3)

    # Polarity default by component
    if expected_polarity is None:
        expected_polarity = "positive"

    # Candidate localization (optionally on smoothed)
    rel_idx = cand_rel_idx
    if rel_idx is None and use_sg and _HAVE_SG and n >= sg_win:
        if sg_win % 2 == 0:
            sg_win -= 1
        sg_win = min(sg_win, n if n % 2 == 1 else n - 1)
        if sg_win < 3:
            sg_win = 3
        try:
            seg_for_loc = savgol_filter(seg_raw, sg_win, sg_poly, mode="interp")
        except Exception:
            seg_for_loc = seg_raw
        rel_idx = int(np.argmax(seg_for_loc)) if expected_polarity == "positive" else int(np.argmin(seg_for_loc))

    if rel_idx is None:
        rel_idx = int(np.argmax(seg_raw)) if expected_polarity == "positive" else int(np.argmin(seg_raw))

    # Raw amplitude for decision (never smoothed)
    height = float(seg_raw[rel_idx])

    # Local MAD estimate excluding the candidate vicinity
    if baseline_mode == "rolling":
        win = max(3, int(round(np.sqrt(n))))
        mad_series = _rolling_mad(seg_raw, win)
        half = half_fwhm_samples(seg_raw, rel_idx) if excl_ms <= 0 else max(3, int(round(excl_ms * sampling_rate / 1000.0)))
        L = max(0, rel_idx - half)
        R = min(n, rel_idx + half + 1)
        noise_vals = np.concatenate([mad_series[:L], mad_series[R:]]) if (R > L and (L > 0 or R < n)) else mad_series
        mad = float(np.median(noise_vals)) if noise_vals.size else float(np.median(mad_series))
    else:  # "median"
        half = half_fwhm_samples(seg_raw, rel_idx) if excl_ms <= 0 else max(3, int(round(excl_ms * sampling_rate / 1000.0)))
        L = max(0, rel_idx - half)
        R = min(n, rel_idx + half + 1)
        noise_seg = np.concatenate([seg_raw[:L], seg_raw[R:]]) if (R > L and (L > 0 or R < n)) else seg_raw
        med = float(np.median(noise_seg)) if noise_seg.size else 0.0
        mad = 1.4826 * float(np.median(np.abs(noise_seg - med))) if noise_seg.size else 0.0

    if mad <= 1e-9:
        return [True, rel_idx, height]  # degenerate/flat noise → accept

    keep = (abs(height) >= k * mad)
    return [keep, rel_idx, height]



