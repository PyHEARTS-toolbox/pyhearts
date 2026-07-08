"""
Record-level R→P / R→T delay regularization (step 5).

After per-beat detection, clip beat-to-beat delays toward a robust record median
(MAD-gated) to tighten interval scatter without moving R or inlier beats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.fiducial_provenance import set_wave_source
from pyhearts.processing.fiducial_refine import refine_global_on_epoch_signal
from pyhearts.processing.peaks import (
    refine_peak_index_subsample,
    sample_at_fractional_index,
)


def _is_finite(val) -> bool:
    return val is not None and not (isinstance(val, float) and np.isnan(val))


@dataclass(frozen=True)
class RecordDelayPrior:
    """Robust record-level delay (samples) from R to P or T."""

    median_delay_samples: float
    mad_delay_samples: float
    n_beats: int
    valid: bool


def estimate_record_delay_prior(
    r_global: np.ndarray,
    wave_global: np.ndarray,
    *,
    min_beats: int,
    default_delay_ms: float,
    sampling_rate: float,
) -> RecordDelayPrior:
    """Median/MAD of (wave - R) sample delays for finite pairs."""
    delays: List[float] = []
    for r, w in zip(r_global, wave_global):
        if _is_finite(r) and _is_finite(w):
            delays.append(float(w) - float(r))

    default_samples = default_delay_ms * sampling_rate / 1000.0
    if len(delays) < min_beats:
        return RecordDelayPrior(
            median_delay_samples=default_samples,
            mad_delay_samples=max(8.0, 0.08 * default_samples),
            n_beats=len(delays),
            valid=False,
        )

    arr = np.asarray(delays, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad < 3.0:
        mad = max(8.0, 0.08 * max(abs(med), 1.0))
    return RecordDelayPrior(
        median_delay_samples=med,
        mad_delay_samples=mad,
        n_beats=len(delays),
        valid=True,
    )


def _clip_delay_to_fence(
    delay: float,
    prior: RecordDelayPrior,
    max_deviation_mad: float,
) -> float:
    """Clip delay to [median - k*MAD, median + k*MAD]."""
    fence = prior.mad_delay_samples * max_deviation_mad
    lo = prior.median_delay_samples - fence
    hi = prior.median_delay_samples + fence
    return float(np.clip(delay, lo, hi))


def smooth_wave_global_indices(
    r_global: np.ndarray,
    wave_global: np.ndarray,
    prior: RecordDelayPrior,
    *,
    max_deviation_mad: float,
    strength: float,
) -> Tuple[np.ndarray, int]:
    """
    MAD-gated delay regularization on global peak indices.

    Out-of-fence beats are moved toward ``R + clipped_delay`` by *strength*
    (0 = no change, 1 = full move to fenced target). In-fence beats unchanged.
    """
    out = np.asarray(wave_global, dtype=float).copy()
    n_adjusted = 0
    strength = float(np.clip(strength, 0.0, 1.0))
    if not prior.valid or strength <= 0.0:
        return out, 0

    fence = prior.mad_delay_samples * max_deviation_mad
    for i, (r, w) in enumerate(zip(r_global, out)):
        if not (_is_finite(r) and _is_finite(w)):
            continue
        delay = float(w) - float(r)
        if abs(delay - prior.median_delay_samples) <= fence:
            continue
        clipped = _clip_delay_to_fence(delay, prior, max_deviation_mad)
        target = float(r) + clipped
        out[i] = float(w) + strength * (target - float(w))
        n_adjusted += 1
    return out, n_adjusted


def refine_global_peak_on_epoch(
    global_idx: float,
    r_global: float,
    one_cycle: pd.DataFrame,
    *,
    wave: str,
    sampling_rate: float,
    half_window_ms: float,
    cfg: ProcessCycleConfig,
) -> float:
    """Localize P/T on detrended epoch after MAD snap (Sprint 2 operators)."""
    sig = one_cycle["signal_y"].values.astype(float)
    if len(sig) < 3:
        return global_idx

    if "index" in one_cycle.columns:
        xs = one_cycle["index"].values.astype(int)
    else:
        xs = one_cycle["signal_x"].values.astype(int)
    matches = np.where(xs == int(round(r_global)))[0]
    r_rel = int(matches[0]) if len(matches) else 0
    r_volt = float(sig[r_rel]) if r_rel < len(sig) else 0.0

    if wave == "P":
        polarity = "positive" if r_volt >= 0 else "negative"
    else:
        polarity = "negative" if r_volt >= 0 else "positive"

    return refine_global_on_epoch_signal(
        global_idx,
        r_global,
        one_cycle,
        sig,
        wave=wave,
        polarity=polarity,
        sampling_rate=sampling_rate,
        cfg=cfg,
        half_window_ms=half_window_ms,
    )


def _sync_cycle_relative_peak(
    output_dict: Dict,
    cycle_idx: int,
    wave: str,
    global_idx: float,
    one_cycle: pd.DataFrame,
    sig: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> None:
    """Update cycle-relative center index from adjusted global index."""
    if "index" in one_cycle.columns:
        xs = one_cycle["index"].values.astype(int)
    else:
        xs = one_cycle["signal_x"].values.astype(int)
    if len(xs) == 0:
        return

    r_g = output_dict.get("R_global_center_idx", [np.nan])
    r_val = r_g[cycle_idx] if cycle_idx < len(r_g) else np.nan
    if not _is_finite(r_val):
        return
    refine = cfg.use_subsample_peak_refinement
    g = float(global_idx)
    xf = xs.astype(float)
    if g <= xf[0]:
        best_rel = 0.0
    elif g >= xf[-1]:
        best_rel = float(len(xs) - 1)
    else:
        i1 = int(np.searchsorted(xf, g))
        i0 = i1 - 1
        frac = (g - xf[i0]) / (xf[i1] - xf[i0]) if xf[i1] != xf[i0] else 0.0
        best_rel = float(i0) + frac
    if refine and sig.size >= 3:
        anchor = int(np.clip(np.round(best_rel), 1, sig.size - 2))
        from pyhearts.processing.peaks import refine_peak_parabolic

        best_rel = refine_peak_parabolic(sig, anchor)

    center_key = f"{wave}_center_idx"
    volt_key = f"{wave}_center_voltage"
    ms_key = f"{wave}_center_ms"
    global_key = f"{wave}_global_center_idx"

    output_dict[global_key][cycle_idx] = float(global_idx)
    output_dict[center_key][cycle_idx] = best_rel
    if refine:
        output_dict[volt_key][cycle_idx] = sample_at_fractional_index(sig, best_rel)
    else:
        i_n = int(np.clip(np.round(best_rel), 0, len(sig) - 1))
        output_dict[volt_key][cycle_idx] = float(sig[i_n])
    output_dict[ms_key][cycle_idx] = (best_rel / sampling_rate) * 1000.0


def apply_record_fiducial_smoothing(
    output_dict: Dict,
    epochs_df: pd.DataFrame,
    cycle_labels: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    *,
    record_t_morphology: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Post-pass: regularize P and T global timing using record-level delay priors.

    R peaks are never modified. Runs after per-beat detection (and T recovery).
    """
    if not cfg.record_fiducial_smoothing:
        return {"p_adjusted": 0, "t_adjusted": 0, "skipped": 1}

    r_global = np.asarray(output_dict.get("R_global_center_idx", []), dtype=float)
    p_global = np.asarray(output_dict.get("P_global_center_idx", []), dtype=float)
    t_global = np.asarray(output_dict.get("T_global_center_idx", []), dtype=float)

    stats: Dict[str, int] = {
        "p_adjusted": 0,
        "t_adjusted": 0,
        "p_prior_valid": 0,
        "t_prior_valid": 0,
    }

    p_prior = estimate_record_delay_prior(
        r_global,
        p_global,
        min_beats=cfg.record_smooth_min_beats,
        default_delay_ms=cfg.record_smooth_default_rp_ms,
        sampling_rate=sampling_rate,
    )
    t_prior = estimate_record_delay_prior(
        r_global,
        t_global,
        min_beats=cfg.record_smooth_min_beats,
        default_delay_ms=cfg.record_smooth_default_rt_ms,
        sampling_rate=sampling_rate,
    )
    stats["p_prior_valid"] = int(p_prior.valid)
    stats["t_prior_valid"] = int(t_prior.valid)
    stats["p_prior_n"] = p_prior.n_beats
    stats["t_prior_n"] = t_prior.n_beats

    if cfg.record_smooth_p and p_prior.valid:
        p_new, n_p = smooth_wave_global_indices(
            r_global,
            p_global,
            p_prior,
            max_deviation_mad=cfg.record_smooth_max_deviation_mad,
            strength=cfg.record_smooth_strength,
        )
        stats["p_adjusted"] = n_p
    else:
        p_new = p_global

    if cfg.record_smooth_t and t_prior.valid:
        t_new, n_t = smooth_wave_global_indices(
            r_global,
            t_global,
            t_prior,
            max_deviation_mad=cfg.record_smooth_max_deviation_mad,
            strength=cfg.record_smooth_strength,
        )
        stats["t_adjusted"] = n_t
    else:
        t_new = t_global

    if stats["p_adjusted"] == 0 and stats["t_adjusted"] == 0:
        return stats

    output_dict["P_global_center_idx"] = list(p_new)
    output_dict["T_global_center_idx"] = list(t_new)

    for cycle_idx, cycle_label in enumerate(cycle_labels):
        if cycle_idx >= len(r_global):
            break
        one_cycle = epochs_df.loc[epochs_df["cycle"] == cycle_label].sort_values("index")
        if one_cycle.empty:
            continue
        sig = one_cycle["signal_y"].values.astype(float)

        r_g = r_global[cycle_idx] if cycle_idx < len(r_global) else np.nan

        if stats["p_adjusted"] > 0 and _is_finite(p_new[cycle_idx]) and _is_finite(r_g):
            p_g = p_new[cycle_idx]
            if cfg.record_smooth_refine_on_epoch_ms > 0:
                p_g = refine_global_peak_on_epoch(
                    p_g,
                    float(r_g),
                    one_cycle,
                    wave="P",
                    sampling_rate=sampling_rate,
                    half_window_ms=cfg.record_smooth_refine_on_epoch_ms,
                    cfg=cfg,
                )
                p_new[cycle_idx] = p_g
            _sync_cycle_relative_peak(
                output_dict, cycle_idx, "P", p_g, one_cycle, sig, sampling_rate, cfg
            )
            set_wave_source(
                output_dict, cycle_idx, "P", "smoothed", confidence="high"
            )

        if stats["t_adjusted"] > 0 and _is_finite(t_new[cycle_idx]) and _is_finite(r_g):
            t_g = t_new[cycle_idx]
            if cfg.record_smooth_refine_on_epoch_ms > 0:
                t_g = refine_global_peak_on_epoch(
                    t_g,
                    float(r_g),
                    one_cycle,
                    wave="T",
                    sampling_rate=sampling_rate,
                    half_window_ms=cfg.record_smooth_refine_on_epoch_ms,
                    cfg=cfg,
                )
                t_new[cycle_idx] = t_g
            _sync_cycle_relative_peak(
                output_dict, cycle_idx, "T", t_g, one_cycle, sig, sampling_rate, cfg
            )
            set_wave_source(
                output_dict, cycle_idx, "T", "smoothed", confidence="high"
            )

    if verbose:
        print(f"[record smoothing] stats={stats}")
    return stats
