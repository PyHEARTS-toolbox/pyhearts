"""
Rahul-first: seed per-beat P/T timing from record STPQ before ``process_cycle``.

Miss policy (``record_fiducial_miss_policy``)
---------------------------------------------
When STPQ returns ``None`` or an implausible apex for P or T:

1. **Never** fall back to per-cycle derivative P/T detection (Rahul-first timing lock).
2. **Never** skip the beat — ``process_cycle`` still runs for R, Q/S, and shape on
   whatever waves are available.
3. **``template_fallback_then_nan``** (default): if ``record_delineation_template_fallback``,
   use RR-scaled template offset; else leave that wave NaN.
   Source ``record_template_fallback`` / ``record_wavelet_fallback``, confidence ``medium``.
4. **``nan_only``**: skip template fallback; wave stays NaN, source ``record_stpq_miss``,
   confidence ``none``.
5. Implausible STPQ guesses are rejected before seeding (same fallback/NaN path).
6. Beats with NaN P and/or T still enter shape extraction; interval features for missing
   waves remain NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.delineation_signal import prepare_record_delineation_signal
from pyhearts.processing.fiducial_provenance import init_fiducial_provenance, set_wave_source
from pyhearts.processing.record_delineation import (
    MedianBeatTemplate,
    _local_rr_samples,
    _pt_expected_offset,
    _resolve_p_guess,
    _resolve_t_guess,
    _sync_peak,
    _template_offset_guess,
    build_record_beat_template,
    delineate_record_template,
)


@dataclass(frozen=True)
class LockedWaveFiducial:
    """Pre-seeded apex for ``process_cycle`` (global sample indices)."""

    peak_idx: int
    peak_amplitude: float
    source: str
    confidence: str
    onset_idx: Optional[int] = None
    offset_idx: Optional[int] = None


def _finite(val) -> bool:
    return val is not None and not (isinstance(val, float) and np.isnan(val))


def _ms_to_samples(ms: float, fs: float) -> int:
    return max(1, int(round(ms * fs / 1000.0)))


def _confidence_for_source(source: str) -> str:
    if source in ("record_stpq", "record_template"):
        return "high"
    if source in ("record_template_fallback", "record_wavelet_fallback"):
        return "medium"
    if source == "record_stpq_miss":
        return "none"
    return "low"


def validate_record_pt_plausibility(
    *,
    p_guess: Optional[float],
    t_guess: Optional[float],
    r_g: float,
    r_next: Optional[float],
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Reject implausible STPQ/template guesses before seeding.

    Returns (p, t) with invalid entries set to ``None``.
    """
    from pyhearts.processing.record_stpq_detection import _validate_p_guess

    min_after_r = _ms_to_samples(40.0, sampling_rate)
    max_rt = _ms_to_samples(cfg.record_stpq_t_max_rt_ms, sampling_rate)

    p_out = _validate_p_guess(
        p_guess,
        t_guess,
        int(round(r_g)),
        int(round(r_next)) if r_next is not None else None,
        sampling_rate,
        cfg,
    )
    t_out = t_guess

    if t_out is not None:
        if t_out <= float(r_g) + min_after_r:
            t_out = None
        elif r_next is not None and t_out >= float(r_next) - min_after_r:
            t_out = None
        elif cfg.record_stpq_t_max_rt_ms > 0 and t_out > float(r_g) + max_rt:
            t_out = None

    if (
        p_out is not None
        and t_out is not None
        and cfg.record_stpq_p_r_anchor
        and getattr(cfg, "record_stpq_p_r_anchor_mode", "current_r") == "next_r"
        and p_out >= t_out
    ):
        p_out = None

    return p_out, t_out


def _resolve_with_miss_policy(
    *,
    wave: str,
    stpq_guess: Optional[float],
    stpq_source: str,
    r_g: float,
    tmpl: MedianBeatTemplate,
    scale: float,
    cfg: ProcessCycleConfig,
    expected_offset: Optional[float],
    stats: Dict[str, int],
    ecg_delim: Optional[np.ndarray] = None,
    r_det: Optional[int] = None,
    r_next: Optional[int] = None,
    sampling_rate: Optional[float] = None,
) -> Tuple[Optional[float], str]:
    """Apply miss policy after STPQ miss or plausibility rejection."""
    off_attr = "p_offset_samples" if wave == "P" else "t_offset_samples"
    off = _pt_expected_offset(getattr(tmpl, off_attr), expected_offset, cfg)

    if stpq_guess is not None and stpq_source:
        return stpq_guess, stpq_source

    miss_key = f"{wave.lower()}_stpq_miss"
    stats[miss_key] = stats.get(miss_key, 0) + 1

    if cfg.record_fiducial_miss_policy == "nan_only":
        return None, "record_stpq_miss"

    if cfg.record_delineation_template_fallback:
        guess: Optional[float] = None
        if wave == "P" and cfg.record_stpq_p_r_anchor and ecg_delim is not None and r_det is not None:
            from pyhearts.processing.record_stpq_detection import record_fallback_p_search

            wavelet_off = (
                float(expected_offset) * float(scale)
                if expected_offset is not None and wave == "P"
                else None
            )
            p_fb = record_fallback_p_search(
                ecg_delim,
                int(r_det),
                r_next,
                tmpl,
                float(sampling_rate or 250.0),
                cfg,
                wavelet_pr_offset_samples=wavelet_off,
            )
            guess = p_fb
        elif wave == "P" and tmpl.template_anchor != "s_to_q" and off is not None:
            guess = _template_offset_guess(r_g, off, scale)
        elif wave == "T" and off is not None:
            guess = _template_offset_guess(r_g, off, scale)
        if guess is not None:
            fb_key = f"{wave.lower()}_template_fallback"
            stats[fb_key] = stats.get(fb_key, 0) + 1
            src = (
                "record_wavelet_fallback"
                if cfg.record_wavelet_pt_prior and expected_offset is not None
                else "record_template_fallback"
            )
            return guess, src

    return None, "record_stpq_miss"


def _sample_amplitude(ecg: np.ndarray, idx: float) -> float:
    i = int(np.clip(round(idx), 0, len(ecg) - 1))
    return float(ecg[i])


def seed_record_fiducials_before_cycles(
    output_dict: Dict,
    ecg_signal: np.ndarray,
    r_peaks: np.ndarray,
    epochs_df: pd.DataFrame,
    cycle_labels: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    *,
    clinical_ecg: Optional[np.ndarray] = None,
    expected_max_energy: float = 0.0,
    verbose: bool = False,
) -> Tuple[Dict[int, Dict[str, LockedWaveFiducial]], Dict[str, int]]:
    """
    Run record STPQ (+ template fallback per miss policy) and seed ``output_dict``
    and ``precomputed_peaks`` before the per-cycle loop.

    Returns ``(precomputed_peaks, stats)``.
    """
    stats: Dict[str, int] = {
        "template_valid": 0,
        "p_seeded": 0,
        "t_seeded": 0,
        "p_nan": 0,
        "t_nan": 0,
    }
    precomputed: Dict[int, Dict[str, LockedWaveFiducial]] = {}

    init_fiducial_provenance(output_dict, len(cycle_labels))

    ecg_delim = prepare_record_delineation_signal(ecg_signal, sampling_rate, cfg)
    template_ecg = (
        np.asarray(clinical_ecg, dtype=float) if clinical_ecg is not None else ecg_signal
    )

    raw = build_record_beat_template(template_ecg, r_peaks, sampling_rate, cfg)
    tmpl = delineate_record_template(raw, sampling_rate, cfg)
    if not tmpl.valid:
        if verbose:
            print("[record seed] template invalid — no P/T seeded")
        return precomputed, stats

    stats["template_valid"] = 1
    stats["t_morphology"] = str(getattr(tmpl, "t_morphology", "normal") or "normal")

    wavelet_priors = None
    if cfg.record_wavelet_pt_prior:
        from pyhearts.processing.record_wavelet_delineation import (
            compute_record_wavelet_pt_priors,
        )

        wavelet_priors = compute_record_wavelet_pt_priors(
            ecg_delim,
            r_peaks,
            cycle_labels,
            tmpl,
            sampling_rate,
            cfg,
            expected_max_energy,
        )

    rr_scale = cfg.record_delineation_rr_scale_pt
    lo_rr_scale, hi_rr_scale = cfg.record_delineation_rr_scale_bounds
    amp_ecg = np.asarray(ecg_signal, dtype=float)

    for cycle_idx, cycle_label in enumerate(cycle_labels):
        one_cycle = epochs_df.loc[epochs_df["cycle"] == cycle_label].sort_values("index")
        if one_cycle.empty:
            continue

        epoch_i = int(cycle_label)
        if epoch_i < 0 or epoch_i >= len(r_peaks):
            continue

        r_det = int(r_peaks[epoch_i])
        r_next = int(r_peaks[epoch_i + 1]) if epoch_i + 1 < len(r_peaks) else None
        r_g = float(r_det)

        local_rr = _local_rr_samples(cycle_idx, r_peaks, cycle_labels, tmpl.median_rr_samples)
        scale = 1.0
        if rr_scale and tmpl.median_rr_samples > 0:
            scale = float(np.clip(local_rr / tmpl.median_rr_samples, lo_rr_scale, hi_rr_scale))

        p_expected = (
            wavelet_priors.expected_p_offset(cycle_idx) if wavelet_priors else None
        )
        t_expected = (
            wavelet_priors.expected_t_offset(cycle_idx) if wavelet_priors else None
        )

        p_raw, p_src_raw = _resolve_p_guess(
            ecg_delim=ecg_delim,
            r_det=r_det,
            r_next=r_next,
            r_g=r_g,
            tmpl=tmpl,
            sampling_rate=sampling_rate,
            cfg=cfg,
            scale=scale,
            stats=stats,
            p_expected_offset=p_expected,
            t_expected_offset=t_expected,
        )
        t_raw, t_src_raw = _resolve_t_guess(
            ecg_delim=ecg_delim,
            r_det=r_det,
            r_next=r_next,
            r_g=r_g,
            tmpl=tmpl,
            sampling_rate=sampling_rate,
            cfg=cfg,
            scale=scale,
            stats=stats,
            t_expected_offset=t_expected,
            p_expected_offset=p_expected,
        )

        p_guess, t_guess = validate_record_pt_plausibility(
            p_guess=p_raw,
            t_guess=t_raw,
            r_g=r_g,
            r_next=r_next,
            sampling_rate=sampling_rate,
            cfg=cfg,
        )

        if p_raw is not None and p_guess is None:
            p_src_raw = ""
        if t_raw is not None and t_guess is None:
            t_src_raw = ""

        p_final, p_src = _resolve_with_miss_policy(
            wave="P",
            stpq_guess=p_guess,
            stpq_source=p_src_raw if p_guess is not None else "",
            r_g=r_g,
            tmpl=tmpl,
            scale=scale,
            cfg=cfg,
            expected_offset=p_expected,
            stats=stats,
            ecg_delim=ecg_delim,
            r_det=r_det,
            r_next=r_next,
            sampling_rate=sampling_rate,
        )
        t_final, t_src = _resolve_with_miss_policy(
            wave="T",
            stpq_guess=t_guess,
            stpq_source=t_src_raw if t_guess is not None else "",
            r_g=r_g,
            tmpl=tmpl,
            scale=scale,
            cfg=cfg,
            expected_offset=t_expected,
            stats=stats,
            ecg_delim=ecg_delim,
            r_det=r_det,
            r_next=r_next,
            sampling_rate=sampling_rate,
        )

        p_final, t_final = validate_record_pt_plausibility(
            p_guess=p_final,
            t_guess=t_final,
            r_g=r_g,
            r_next=r_next,
            sampling_rate=sampling_rate,
            cfg=cfg,
        )

        beat_peaks: Dict[str, LockedWaveFiducial] = {}

        if p_final is not None:
            _sync_peak(output_dict, cycle_idx, "P", p_final, one_cycle, sampling_rate, cfg)
            conf = _confidence_for_source(p_src)
            set_wave_source(output_dict, cycle_idx, "P", p_src, confidence=conf)
            beat_peaks["P"] = LockedWaveFiducial(
                peak_idx=int(round(p_final)),
                peak_amplitude=_sample_amplitude(amp_ecg, p_final),
                source=p_src,
                confidence=conf,
            )
            stats["p_seeded"] += 1
        else:
            stats["p_nan"] += 1
            set_wave_source(output_dict, cycle_idx, "P", "record_stpq_miss", confidence="none")

        if t_final is not None:
            _sync_peak(output_dict, cycle_idx, "T", t_final, one_cycle, sampling_rate, cfg)
            conf = _confidence_for_source(t_src)
            set_wave_source(output_dict, cycle_idx, "T", t_src, confidence=conf)
            beat_peaks["T"] = LockedWaveFiducial(
                peak_idx=int(round(t_final)),
                peak_amplitude=_sample_amplitude(amp_ecg, t_final),
                source=t_src,
                confidence=conf,
            )
            stats["t_seeded"] += 1
        else:
            stats["t_nan"] += 1
            set_wave_source(output_dict, cycle_idx, "T", "record_stpq_miss", confidence="none")

        if beat_peaks:
            precomputed[cycle_idx] = beat_peaks

    if verbose:
        print(f"[record seed before cycles] {stats}")

    return precomputed, stats
