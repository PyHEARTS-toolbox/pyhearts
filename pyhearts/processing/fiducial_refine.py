"""
Local P/T timing refinement operators (Sprint 2).

Used by record delineation and record fiducial smoothing after coarse record-T/template guesses.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.delineation_signal import smooth_search_window
from pyhearts.processing.derivative_t_detection import (
    compute_filtered_derivative,
    detect_t_wave_derivative_based,
)
from pyhearts.processing.peaks import find_peak_derivative_based


def adaptive_refine_half_window_ms(
    cfg: ProcessCycleConfig,
    wave: str,
    r_global: float,
    guess_global: float,
    sampling_rate: float,
) -> float:
    """
    Refine search half-width: base ``record_delineation_refine_ms``, optionally scaled by RT.
    """
    base = float(cfg.record_delineation_refine_ms)
    if not cfg.record_delineation_refine_adaptive:
        return base
    rt_ms = abs(float(guess_global) - float(r_global)) / sampling_rate * 1000.0
    scaled = base + cfg.record_delineation_refine_rt_frac * rt_ms
    return float(
        np.clip(
            scaled,
            base,
            cfg.record_delineation_refine_ms_max,
        )
    )


def _operator_for_wave(cfg: ProcessCycleConfig, wave: str) -> str:
    wave = wave.upper()
    if wave == "P":
        return cfg.record_refine_p_operator
    if wave == "T":
        return cfg.record_refine_t_operator
    raise ValueError(f"wave must be P or T, got {wave!r}")


def clinical_operator_for_wave(cfg: ProcessCycleConfig, wave: str) -> str:
    """Operator for Sprint 3 clinical verify (falls back to record refine operators)."""
    wave = wave.upper()
    if wave == "P":
        return cfg.clinical_verify_p_operator or cfg.record_refine_p_operator
    if wave == "T":
        return cfg.clinical_verify_t_operator or cfg.record_refine_t_operator
    raise ValueError(f"wave must be P or T, got {wave!r}")


def _apex_from_polarity(
    segment: np.ndarray,
    polarity: str,
    operator: str,
) -> int:
    if operator == "argmin":
        return int(np.argmin(segment))
    if operator == "argmax":
        return int(np.argmax(segment))
    peak_abs, _ = find_peak_derivative_based(
        segment,
        0,
        len(segment),
        polarity,
        verbose=False,
        label=None,
    )
    if peak_abs is not None:
        return int(peak_abs)
    if polarity == "positive":
        return int(np.argmax(segment))
    return int(np.argmin(segment))


def _refine_t_derivative_zc(
    segment: np.ndarray,
    anchor_rel: int,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> Optional[int]:
    """T peak via LP derivative zero-crossing (ECGPUWAVE-style) in segment coordinates."""
    if len(segment) < 5:
        return None
    deriv = compute_filtered_derivative(
        segment,
        sampling_rate,
        lowpass_cutoff=cfg.record_refine_t_lowpass_hz,
    )
    t_peak, _, _, _, _ = detect_t_wave_derivative_based(
        segment,
        deriv,
        0,
        len(segment),
        sampling_rate=sampling_rate,
        verbose=False,
    )
    if t_peak is not None and 0 <= t_peak < len(segment):
        return int(t_peak)
    return _apex_from_polarity(segment, "negative", "derivative_apex")


def refine_in_segment(
    segment: np.ndarray,
    anchor_rel: int,
    *,
    wave: str,
    polarity: str,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    half_window_ms: float,
    operator: Optional[str] = None,
) -> int:
    """
    Refine fiducial index within *segment* (cycle-relative coordinates).

    ``anchor_rel`` is the coarse guess index inside ``segment``.
    """
    if len(segment) < 3:
        return int(np.clip(anchor_rel, 0, max(0, len(segment) - 1)))

    half = int(round(half_window_ms * sampling_rate / 1000.0))
    lo = max(0, int(round(anchor_rel)) - half)
    hi = min(len(segment), int(round(anchor_rel)) + half + 1)
    if hi - lo < 3:
        return int(np.clip(anchor_rel, 0, len(segment) - 1))

    seg_smooth, lo, hi = smooth_search_window(segment, lo, hi, sampling_rate, cfg)
    operator = operator or _operator_for_wave(cfg, wave)

    if wave.upper() == "T" and operator == "derivative_zc":
        peak_rel = _refine_t_derivative_zc(seg_smooth, anchor_rel - lo, sampling_rate, cfg)
    else:
        peak_rel = _apex_from_polarity(seg_smooth, polarity, operator)

    return int(np.clip(lo + peak_rel, 0, len(segment) - 1))


def refine_global_on_epoch_signal(
    global_idx: float,
    r_global: float,
    one_cycle,
    sig: np.ndarray,
    *,
    wave: str,
    polarity: str,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    half_window_ms: float,
) -> float:
    """Refine a global sample index using detrended epoch ``sig`` (smoothing pass)."""
    if "index" in one_cycle.columns:
        xs = one_cycle["index"].values.astype(int)
    else:
        xs = one_cycle["signal_x"].values.astype(int)
    if len(xs) == 0 or len(sig) == 0:
        return float(global_idx)

    g = float(global_idx)
    xf = xs.astype(float)
    if g <= xf[0]:
        center_rel = 0.0
    elif g >= xf[-1]:
        center_rel = float(len(sig) - 1)
    else:
        i1 = int(np.searchsorted(xf, g))
        i0 = i1 - 1
        frac = (g - xf[i0]) / (xf[i1] - xf[i0]) if xf[i1] != xf[i0] else 0.0
        center_rel = float(i0) + frac

    refined_rel = refine_in_segment(
        sig,
        int(round(center_rel)),
        wave=wave,
        polarity=polarity,
        sampling_rate=sampling_rate,
        cfg=cfg,
        half_window_ms=half_window_ms,
    )
    i0 = int(np.floor(refined_rel))
    i1 = min(i0 + 1, len(xs) - 1)
    frac = refined_rel - i0
    return float(xs[i0]) * (1.0 - frac) + float(xs[i1]) * frac
