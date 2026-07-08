"""
Sprint 3: second-pass P/T timing on clinical (or epoch) trace after record delineation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.derivative_t_detection import compute_t_search_window
from pyhearts.processing.fiducial_provenance import set_wave_source
from pyhearts.processing.fiducial_refine import (
    adaptive_refine_half_window_ms,
    clinical_operator_for_wave,
    refine_in_segment,
)
from pyhearts.processing.record_delineation import _sync_peak
from pyhearts.processing.t_plausibility import (
    check_t_peak_dominance,
    validate_p_pr_interval,
)


def _finite(val) -> bool:
    return val is not None and not (isinstance(val, float) and np.isnan(val))


_T_CLINICAL_FILL_SOURCES = (
    "record_template_fallback",
    "record_wavelet_fallback",
    "record_fill_missing",
    "record_fill_missing_template",
)


def _record_rt_prior_ms(
    t_list: List,
    r_list: List,
    fs: float,
) -> tuple[Optional[float], Optional[float]]:
    """Median RT (ms) and MAD for conditional clinical T verification."""
    rts: List[float] = []
    for t_val, r_val in zip(t_list, r_list):
        if _finite(t_val) and _finite(r_val):
            rts.append((float(t_val) - float(r_val)) / fs * 1000.0)
    if len(rts) < 3:
        return None, None
    arr = np.asarray(rts, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad < 1.0:
        mad = max(15.0, 0.05 * abs(med))
    return med, mad


def _should_clinical_verify_t(
    t_val: float,
    r_val: float,
    t_source: Optional[str],
    cfg: ProcessCycleConfig,
    fs: float,
    median_rt_ms: Optional[float],
    mad_rt_ms: Optional[float],
) -> bool:
    """Clinical T refine only when missing was filled or RT disputes record prior."""
    if not cfg.clinical_verify_t_conditional:
        return _finite(t_val)
    if not _finite(t_val):
        return False
    src = str(t_source or "")
    if any(tag in src for tag in _T_CLINICAL_FILL_SOURCES):
        return True
    if median_rt_ms is None or mad_rt_ms is None:
        return True
    rt_ms = (float(t_val) - float(r_val)) / fs * 1000.0
    fence = max(
        float(cfg.clinical_verify_t_dispute_min_ms),
        float(cfg.clinical_verify_t_dispute_mad_mult) * float(mad_rt_ms),
    )
    return abs(rt_ms - median_rt_ms) > fence


def _ms_to_samples(ms: float, fs: float) -> int:
    return int(round(ms * fs / 1000.0))


def _clinical_signal_for_cycle(
    one_cycle: pd.DataFrame,
    clinical_ecg: Optional[np.ndarray],
    cfg: ProcessCycleConfig,
) -> np.ndarray:
    if cfg.clinical_verify_signal == "epoch" or clinical_ecg is None:
        return one_cycle["signal_y"].values.astype(float)
    if "index" not in one_cycle.columns:
        return one_cycle["signal_y"].values.astype(float)
    xs = one_cycle["index"].values.astype(int)
    lo, hi = int(xs[0]), int(xs[-1])
    if 0 <= lo < clinical_ecg.size and hi < clinical_ecg.size:
        return clinical_ecg[lo : hi + 1].astype(float, copy=False)
    return one_cycle["signal_y"].values.astype(float)


def _polarity_from_r(sig: np.ndarray, r_rel: int, wave: str) -> str:
    r_rel = int(np.clip(r_rel, 0, len(sig) - 1))
    r_volt = float(sig[r_rel])
    if wave == "P":
        return "positive" if r_volt >= 0 else "negative"
    return "negative" if r_volt >= 0 else "positive"


def _p_search_bounds(
    r_rel: int,
    n_samples: int,
    cfg: ProcessCycleConfig,
    fs: float,
) -> tuple[int, int]:
    before = _ms_to_samples(cfg.record_delineation_p_search_before_r_ms, fs)
    end_before = _ms_to_samples(cfg.record_delineation_p_search_end_before_r_ms, fs)
    lo = max(0, r_rel - before)
    hi = max(lo + 2, r_rel - end_before)
    hi = min(hi, n_samples)
    return lo, hi


def _t_search_bounds(
    r_rel: int,
    n_samples: int,
    cycle_len: int,
    cfg: ProcessCycleConfig,
    fs: float,
    *,
    s_rel: Optional[int] = None,
) -> tuple[int, int]:
    return compute_t_search_window(
        r_rel,
        cycle_len,
        fs,
        s_center_idx=s_rel,
        start_offset_ms=cfg.record_delineation_t_search_after_r_ms,
        max_offset_ms=cfg.record_delineation_t_search_end_ms,
    )


def _clinical_half_window_ms(
    cfg: ProcessCycleConfig,
    wave: str,
    r_g: float,
    guess_g: float,
    fs: float,
) -> float:
    wave = wave.upper()
    if wave == "P":
        return float(cfg.clinical_verify_p_window_ms)
    if cfg.clinical_verify_t_window_ms > 0:
        return float(cfg.clinical_verify_t_window_ms)
    if cfg.clinical_verify_t_adaptive:
        return adaptive_refine_half_window_ms(cfg, wave, r_g, guess_g, fs)
    return float(cfg.clinical_verify_p_window_ms)


def _plausible_p(
    p_rel: int,
    r_rel: int,
    sig: np.ndarray,
    fs: float,
    cfg: ProcessCycleConfig,
) -> bool:
    if not validate_p_pr_interval(p_rel, r_rel, fs, cfg):
        return False
    r_amp = abs(float(sig[r_rel])) if r_rel < len(sig) else 0.0
    p_amp = abs(float(sig[p_rel])) if p_rel < len(sig) else 0.0
    min_ratio = cfg.amp_min_ratio.get("P", 0.03)
    if r_amp > 1e-9 and p_amp < min_ratio * r_amp:
        return False
    return True


def _plausible_t(
    t_rel: int,
    r_rel: int,
    sig: np.ndarray,
    t_lo: int,
    t_hi: int,
    fs: float,
    cfg: ProcessCycleConfig,
) -> bool:
    rt_ms = (t_rel - r_rel) / fs * 1000.0
    lo, hi = cfg.t_rt_bounds_ms
    if rt_ms < lo or rt_ms > hi:
        return False
    r_amp = abs(float(sig[r_rel])) if r_rel < len(sig) else 0.0
    t_amp = abs(float(sig[t_rel])) if t_rel < len(sig) else 0.0
    min_ratio = cfg.amp_min_ratio.get("T", 0.02)
    if r_amp > 1e-9 and t_amp < min_ratio * r_amp:
        return False
    morphology = 1 if float(sig[t_rel]) < 0 else 0
    if not check_t_peak_dominance(
        sig, t_rel, t_lo, t_hi, morphology, fs, cfg
    ):
        return False
    return True


def _refine_wave_clinical(
    sig: np.ndarray,
    xs: np.ndarray,
    anchor_global: float,
    r_global: float,
    *,
    wave: str,
    cfg: ProcessCycleConfig,
    fs: float,
) -> Optional[int]:
    """Return refined global sample index or None."""
    if len(sig) < 3:
        return None
    xf = xs.astype(float)
    g = float(anchor_global)
    if g <= xf[0]:
        center_rel = 0
    elif g >= xf[-1]:
        center_rel = len(sig) - 1
    else:
        i1 = int(np.searchsorted(xf, g))
        i0 = i1 - 1
        frac = (g - xf[i0]) / (xf[i1] - xf[i0]) if xf[i1] != xf[i0] else 0.0
        center_rel = int(round(float(i0) + frac))

    r_matches = np.where(np.abs(xf - float(r_global)) < 0.5)[0]
    r_rel = int(r_matches[0]) if len(r_matches) else int(
        np.argmin(np.abs(xf - float(r_global)))
    )

    if wave == "P":
        lo, hi = _p_search_bounds(r_rel, len(sig), cfg, fs)
    else:
        s_rel = None
        lo, hi = _t_search_bounds(
            r_rel, len(sig), len(sig), cfg, fs, s_rel=s_rel
        )

    lo = max(0, min(lo, len(sig) - 2))
    hi = max(lo + 2, min(hi, len(sig)))
    center_rel = int(np.clip(center_rel, lo, hi - 1))

    half_ms = _clinical_half_window_ms(cfg, wave, r_global, anchor_global, fs)
    polarity = _polarity_from_r(sig, r_rel, wave)
    op = clinical_operator_for_wave(cfg, wave)

    refined_rel = refine_in_segment(
        sig,
        center_rel,
        wave=wave,
        polarity=polarity,
        sampling_rate=fs,
        cfg=cfg,
        half_window_ms=half_ms,
        operator=op,
    )

    if wave == "P" and not _plausible_p(refined_rel, r_rel, sig, fs, cfg):
        return None
    if wave == "T" and not _plausible_t(refined_rel, r_rel, sig, lo, hi, fs, cfg):
        return None

    return int(xs[refined_rel]) if refined_rel < len(xs) else int(xs[-1])


def apply_clinical_fiducial_verification(
    output_dict: Dict,
    epochs_df: pd.DataFrame,
    cycle_labels: np.ndarray,
    clinical_ecg: np.ndarray,
    r_peaks: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    *,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Wide-window refine of P/T on clinical trace after record delineation.

    Runs before RT plausibility gate and record MAD smoothing.
    """
    stats = {
        "skipped": 0,
        "p_checked": 0,
        "t_checked": 0,
        "t_skipped_conditional": 0,
        "p_accepted": 0,
        "t_accepted": 0,
        "p_rejected": 0,
        "t_rejected": 0,
        "features_refreshed": 0,
    }
    modified: Set[int] = set()

    if not cfg.record_clinical_verify:
        stats["skipped"] = 1
        return stats

    r_list = output_dict.get("R_global_center_idx", [])
    p_list = output_dict.get("P_global_center_idx", [])
    t_list = output_dict.get("T_global_center_idx", [])
    fs = float(sampling_rate)
    median_rt_ms, mad_rt_ms = _record_rt_prior_ms(t_list, r_list, fs)

    for cycle_idx, cycle_label in enumerate(cycle_labels):
        if cycle_idx >= len(r_list):
            break
        one_cycle = epochs_df.loc[epochs_df["cycle"] == cycle_label].sort_values("index")
        if one_cycle.empty:
            continue
        r_g = r_list[cycle_idx]
        if not _finite(r_g):
            continue

        if "index" in one_cycle.columns:
            xs = one_cycle["index"].values.astype(int)
        else:
            xs = one_cycle["signal_x"].values.astype(int)
        sig = _clinical_signal_for_cycle(one_cycle, clinical_ecg, cfg)

        prev_src_p = (
            output_dict.get("p_source", [None])[cycle_idx]
            if cycle_idx < len(output_dict.get("p_source", []))
            else None
        )
        prev_src_t = (
            output_dict.get("t_source", [None])[cycle_idx]
            if cycle_idx < len(output_dict.get("t_source", []))
            else None
        )

        if cycle_idx < len(p_list) and _finite(p_list[cycle_idx]):
            stats["p_checked"] += 1
            old_p = float(p_list[cycle_idx])
            new_p = _refine_wave_clinical(
                sig, xs, old_p, float(r_g), wave="P", cfg=cfg, fs=fs
            )
            if new_p is not None:
                _sync_peak(
                    output_dict, cycle_idx, "P", float(new_p), one_cycle, fs, cfg
                )
                base = prev_src_p if prev_src_p else "record"
                set_wave_source(
                    output_dict, cycle_idx, "P", f"{base}+clinical", confidence="high"
                )
                stats["p_accepted"] += 1
                if abs(new_p - old_p) > 0.5:
                    modified.add(cycle_idx)
            else:
                set_wave_source(
                    output_dict,
                    cycle_idx,
                    "P",
                    f"{prev_src_p or 'record'}+clinical_rejected",
                    confidence="low",
                )
                stats["p_rejected"] += 1

        if cycle_idx < len(t_list) and _finite(t_list[cycle_idx]):
            if _should_clinical_verify_t(
                float(t_list[cycle_idx]),
                float(r_g),
                prev_src_t,
                cfg,
                fs,
                median_rt_ms,
                mad_rt_ms,
            ):
                stats["t_checked"] += 1
                old_t = float(t_list[cycle_idx])
                new_t = _refine_wave_clinical(
                    sig, xs, old_t, float(r_g), wave="T", cfg=cfg, fs=fs
                )
                if new_t is not None:
                    _sync_peak(
                        output_dict, cycle_idx, "T", float(new_t), one_cycle, fs, cfg
                    )
                    base = prev_src_t if prev_src_t else "record"
                    set_wave_source(
                        output_dict, cycle_idx, "T", f"{base}+clinical", confidence="high"
                    )
                    stats["t_accepted"] += 1
                    if abs(new_t - old_t) > 0.5:
                        modified.add(cycle_idx)
                else:
                    set_wave_source(
                        output_dict,
                        cycle_idx,
                        "T",
                        f"{prev_src_t or 'record'}+clinical_rejected",
                        confidence="low",
                    )
                    stats["t_rejected"] += 1
            else:
                stats["t_skipped_conditional"] += 1

    if modified and cfg.clinical_verify_refresh_features:
        from pyhearts.processing.cycle_feature_refresh import (
            refresh_cycles_after_timing_update,
        )

        refresh_stats = refresh_cycles_after_timing_update(
            output_dict,
            epochs_df,
            cycle_labels,
            sampling_rate,
            cfg,
            modified,
            verbose=verbose,
        )
        stats["features_refreshed"] = refresh_stats.get("cycles_refreshed", 0)

    if verbose:
        print(f"[clinical verify] {stats}")
    return stats
