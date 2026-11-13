"""
Record-level median-beat templates and template delineation.

Builds a median beat template around detected R peaks (same window policy as
epoch_ecg) and locates P/T landmarks on it. The record-level T stage in
``pyhearts.core.analyzer`` and ``record_t_detection`` use these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.delineation_signal import (
    prepare_record_delineation_signal,
    savgol_search_segment,
    smooth_search_window,
)
from pyhearts.processing.peaks import sample_at_fractional_index
from pyhearts.processing.record_t_detection import (
    _apex_with_threshold,
    _early_peak_landmark_frac,
    _is_early_peak_landmark,
    _t_search_prefer_negative,
    defer_record_t_overwrite,
    record_t_use_biphasic_fallback,
)


def _finite(val) -> bool:
    return val is not None and not (isinstance(val, float) and np.isnan(val))


def _ms_to_samples(ms: float, fs: float) -> int:
    return int(round(ms * fs / 1000.0))


@dataclass(frozen=True)
class MedianBeatTemplate:
    """Median beat and P/T offsets (samples) relative to R at template center."""

    template: np.ndarray
    pre_r_samples: int
    r_center_idx: int
    p_offset_samples: Optional[float]
    t_offset_samples: Optional[float]
    p_polarity: str  # "positive" | "negative"
    t_polarity: str
    median_rr_samples: float
    n_beats: int
    valid: bool
    template_anchor: str = "r_centered"
    t_landmark_idx: Optional[int] = None
    p_landmark_idx: Optional[int] = None
    th_t_up: Optional[float] = None
    th_t_down: Optional[float] = None
    th_p_up: Optional[float] = None
    th_p_down: Optional[float] = None
    t_morphology: str = "normal"
    p_pr_center_ms: Optional[float] = None  # record-estimated PR before R for template-guided P
    t_landmark_source: str = "unknown"
    t_biphasic_pos_landmark_idx: Optional[float] = None
    t_biphasic_neg_landmark_idx: Optional[float] = None
    t_post_apex_dz_preference: bool = False


def _resample_segment(segment: np.ndarray, target_len: int) -> np.ndarray:
    if segment.size == target_len:
        return segment.astype(float, copy=False)
    if segment.size < 2 or target_len < 2:
        return np.resize(segment, target_len)
    x_old = np.linspace(0.0, 1.0, segment.size)
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, segment.astype(float))


def _find_s_after_r(
    ecg: np.ndarray,
    r_idx: int,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> Optional[int]:
    """Local minimum after R within the configured Q/S search window (S wave)."""
    from pyhearts.processing.qrs_extrema import find_s_wave_after_r

    return find_s_wave_after_r(
        ecg, r_idx, sampling_rate, search_window_ms=cfg.record_s_search_after_r_ms, inverted=False
    )


def _find_q_before_r(
    ecg: np.ndarray,
    r_idx: int,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> Optional[int]:
    """Local minimum before R within the configured Q/S search window (Q wave)."""
    from pyhearts.processing.qrs_extrema import find_q_wave_before_r

    return find_q_wave_before_r(
        ecg, r_idx, sampling_rate, search_window_ms=cfg.record_q_search_before_r_ms, inverted=False
    )


def _t_region_amplitude_reference(
    sig: np.ndarray,
    baseline: float,
    cfg: ProcessCycleConfig,
) -> float:
    """
    Reference amplitude for T landmark threshold scaling.

    Uses max positive excursion in ``record_template_t_amplitude_norm_sq_frac`` so
    S-wave deflection does not dominate the min-peak fraction check.
    """
    n = sig.size
    lo_f, hi_f = cfg.record_template_t_amplitude_norm_sq_frac
    lo = int(round(lo_f * (n - 1)))
    hi = int(round(hi_f * (n - 1)))
    lo = int(np.clip(lo, 0, n - 2))
    hi = int(np.clip(hi, lo + 1, n - 1))
    pos_excursion = sig[lo : hi + 1] - baseline
    if pos_excursion.size:
        peak = float(np.max(pos_excursion))
        if peak > 0.0:
            return peak
    return float(np.max(np.abs(sig - baseline)))


def _t_region_abs_amplitude_reference(
    sig: np.ndarray,
    baseline: float,
    cfg: ProcessCycleConfig,
) -> float:
    """Max |excursion| in ``record_template_t_amplitude_norm_sq_frac`` (inverted T scaling)."""
    n = sig.size
    lo_f, hi_f = cfg.record_template_t_amplitude_norm_sq_frac
    lo = int(round(lo_f * (n - 1)))
    hi = int(round(hi_f * (n - 1)))
    lo = int(np.clip(lo, 0, n - 2))
    hi = int(np.clip(hi, lo + 1, n - 1))
    excursion = np.abs(sig[lo : hi + 1] - baseline)
    if excursion.size:
        return float(np.max(excursion))
    return float(np.max(np.abs(sig - baseline)))


def _excursion_prominence(excursion: np.ndarray, peak_rel: int, half_width: int) -> float:
    """Peak height above the higher of the left/right envelope minima."""
    if excursion.size == 0:
        return 0.0
    peak_rel = int(np.clip(peak_rel, 0, excursion.size - 1))
    lo = max(0, peak_rel - half_width)
    hi = min(excursion.size - 1, peak_rel + half_width)
    peak_h = float(excursion[peak_rel])
    left_min = float(np.min(excursion[lo : peak_rel + 1]))
    right_min = float(np.min(excursion[peak_rel : hi + 1]))
    return peak_h - max(left_min, right_min)


def _landmarks_t_isoelectric_fallback(template: np.ndarray) -> int:
    """Legacy Tⱼ: sample in first half of S→Q template nearest median baseline."""
    baseline = float(np.median(template))
    n = template.size
    if n < 6:
        return 0
    t_region_end = max(2, n // 2)
    return int(np.argmin(np.abs(template[:t_region_end] - baseline)))


def _has_descending_right_limb(
    sig: np.ndarray,
    peak_idx: int,
    sampling_rate: float,
    *,
    ms: float = 20.0,
) -> bool:
    """
    True when a right limb exists within ``ms`` after the peak.

    For positive apexes the signal falls; for negative troughs it rises.
    """
    off = _ms_to_samples(ms, sampling_rate)
    j = int(peak_idx) + off
    if j >= len(sig):
        return False
    peak_val = float(sig[int(peak_idx)])
    later = float(sig[j])
    if peak_val >= 0.0:
        return later < peak_val
    return later > peak_val


def _has_flat_plateau_right_limb(
    sig: np.ndarray,
    peak_idx: int,
    baseline: float,
    sampling_rate: float,
    *,
    ms: float = 20.0,
    min_retention_frac: float = 0.85,
) -> bool:
    """
    True when |excursion| stays near peak for ``ms`` after apex (flat-topped plateau).

    Distinguishes sele0104-class flat inverted plateaus from sele0203-class troughs
    that recover quickly toward baseline (normal post-trough rise, not flat top).
    """
    off = _ms_to_samples(ms, sampling_rate)
    end = int(peak_idx) + off
    if end >= len(sig):
        return False
    peak_abs = abs(float(sig[int(peak_idx)]) - baseline)
    if peak_abs <= 0.0:
        return False
    window = sig[int(peak_idx) : end + 1]
    min_abs = float(np.min(np.abs(window - baseline)))
    return min_abs >= min_retention_frac * peak_abs


def _rising_edge_t_landmark(abs_excursion: np.ndarray, lo: int) -> Optional[int]:
    """
    First sample in the landmark window whose |excursion| exceeds 50% of the window max.

    Used when prominence and right-limb descent both fail (genuinely flat plateau).
    """
    if abs_excursion.size == 0:
        return None
    peak = float(np.max(abs_excursion))
    if peak <= 0.0:
        return None
    thr = 0.5 * peak
    for i, v in enumerate(abs_excursion):
        if float(v) >= thr:
            return lo + int(i)
    return None


def _resolve_t_landmark_after_peak_gates(
    sig: np.ndarray,
    tmpl: np.ndarray,
    lo: int,
    peak_idx: int,
    peak_val: float,
    peak_prom: float,
    min_amp: float,
    min_prom_req: float,
    abs_excursion: np.ndarray,
    rising_abs_excursion: np.ndarray,
    lo_rise: int,
    sampling_rate: float,
    baseline: float,
) -> Tuple[int, str]:
    """
    Choose Tⱼ after amplitude/prominence gates on the candidate apex.

    Order: full pass → plateau exception (amp ok, prom fail, flat top or positive
    descent) → rising-edge onset → isoelectric fallback.
    """
    if peak_val >= min_amp and peak_prom >= min_prom_req:
        return int(peak_idx), "early_peak"
    if peak_val >= min_amp and peak_prom < min_prom_req:
        peak_sig = float(sig[int(peak_idx)])
        plateau = False
        if peak_sig >= 0.0:
            plateau = _has_descending_right_limb(sig, peak_idx, sampling_rate)
        else:
            plateau = _has_flat_plateau_right_limb(
                sig, peak_idx, baseline, sampling_rate
            )
        if plateau:
            return int(peak_idx), "plateau_apex"
        rising = _rising_edge_t_landmark(rising_abs_excursion, lo_rise)
        if rising is not None:
            return rising, "rising_edge"
    return _landmarks_t_isoelectric_fallback(tmpl), "isoelectric"


def _landmarks_closest_to_baseline(
    template: np.ndarray,
    cfg: ProcessCycleConfig,
    sampling_rate: float,
) -> Tuple[int, int, str]:
    """
    Tⱼ and Pⱼ on the S→Q median template.

    Tⱼ: dominant |excursion| peak in ``record_template_t_landmark_sq_frac`` when it
    meets amplitude + prominence thresholds; else plateau exception (flat-topped apex),
    rising-edge onset, or isoelectric-nearest (first half).
    Pⱼ: unchanged — nearest isoelectric in the second half.
    """
    tmpl = np.asarray(template, dtype=float)
    n = tmpl.size
    if n < 6:
        mid = n // 2
        return 0, max(0, n - 1), "isoelectric"

    baseline = float(np.median(tmpl))
    sig = (
        savgol_search_segment(tmpl, float(sampling_rate), cfg)
        if cfg.p_t_search_savgol
        else tmpl
    )

    lo_f, hi_f = cfg.record_template_t_landmark_sq_frac
    lo = int(round(lo_f * (n - 1)))
    hi = int(round(hi_f * (n - 1)))
    lo = int(np.clip(lo, 0, n - 2))
    hi = int(np.clip(hi, lo + 1, n - 1))
    lo_rise_f = float(getattr(cfg, "record_template_t_rising_edge_lo_frac", lo_f))
    lo_rise = int(round(lo_rise_f * (n - 1)))
    lo_rise = int(np.clip(lo_rise, 0, max(0, lo)))

    abs_ref = _t_region_abs_amplitude_reference(sig, baseline, cfg)
    min_peak = float(cfg.record_template_t_landmark_min_peak_frac)
    min_prom = float(cfg.record_template_t_landmark_min_prominence_frac)
    half_w = max(3, (hi - lo) // 4)

    window = sig[lo : hi + 1]
    abs_excursion = np.abs(window - baseline)
    rising_abs_excursion = np.abs(sig[lo_rise : hi + 1] - baseline)
    abs_rel = int(np.argmax(abs_excursion))
    abs_val = float(abs_excursion[abs_rel])
    abs_prom = _excursion_prominence(abs_excursion, abs_rel, half_w)
    min_amp = min_peak * abs_ref if abs_ref > 0.0 else float("inf")
    min_prom_req = min_prom * abs_ref if abs_ref > 0.0 else float("inf")

    if cfg.record_template_t_landmark_inverted_peak and abs_ref > 0.0:
        t_rel, t_source = _resolve_t_landmark_after_peak_gates(
            sig,
            tmpl,
            lo,
            lo + abs_rel,
            abs_val,
            abs_prom,
            min_amp,
            min_prom_req,
            abs_excursion,
            rising_abs_excursion,
            lo_rise,
            sampling_rate,
            baseline,
        )
    else:
        full_peak = _t_region_amplitude_reference(sig, baseline, cfg)
        pos_excursion = window - baseline
        pos_rel = int(np.argmax(pos_excursion))
        pos_val = float(pos_excursion[pos_rel])
        pos_prom = _excursion_prominence(pos_excursion, pos_rel, half_w)
        min_pos_amp = min_peak * full_peak if full_peak > 0.0 else float("inf")
        min_pos_prom = min_prom * full_peak if full_peak > 0.0 else float("inf")
        if full_peak > 0.0:
            t_rel, t_source = _resolve_t_landmark_after_peak_gates(
                sig,
                tmpl,
                lo,
                lo + pos_rel,
                pos_val,
                pos_prom,
                min_pos_amp,
                min_pos_prom,
                np.abs(pos_excursion),
                rising_abs_excursion,
                lo_rise,
                sampling_rate,
                baseline,
            )
        else:
            t_rel = _landmarks_t_isoelectric_fallback(tmpl)
            t_source = "isoelectric"

    p_start = min(n - 2, n // 2)
    p_rel = p_start + int(np.argmin(np.abs(tmpl[p_start:] - baseline)))
    return t_rel, p_rel, t_source


_MORPH_LANDMARK_OVERRIDE_MIN_DELTA_FRAC = 0.15


def _morphology_t_segment(
    template: np.ndarray,
    t_j: int,
    p_j: int,
    cfg: ProcessCycleConfig,
) -> np.ndarray:
    """T-region segment for morphology classification and T thresholds."""
    morph_frac = getattr(cfg, "record_template_t_morphology_sq_frac", None)
    n = template.size
    if morph_frac is not None and n >= 2:
        lo_frac, hi_frac = morph_frac
        i0 = int(round(float(lo_frac) * (n - 1)))
        i1 = int(round(float(hi_frac) * (n - 1)))
        i0 = max(0, min(i0, n - 2))
        i1 = max(i0 + 1, min(i1, n - 1))
        return template[i0 : i1 + 1]
    mid_tp = int((t_j + p_j) / 2)
    if mid_tp > t_j:
        return template[t_j:mid_tp]
    return template[t_j : t_j + 1]


def _compute_template_thresholds(
    template: np.ndarray,
    t_j: int,
    p_j: int,
    mean_r_amplitude: float,
    cfg: ProcessCycleConfig,
) -> Tuple[float, float, float, float, str]:
    """
    Template P/T amplitude thresholds (conditions I–III).

    Returns (th_t_up, th_t_down, th_p_up, th_p_down, morphology_tag).
    """
    p_third = int(t_j + (p_j - t_j) / 3) if p_j > t_j else t_j + 1

    t_seg = _morphology_t_segment(template, t_j, p_j, cfg)
    p_seg = template[p_third:p_j] if p_j > p_third else template[max(0, p_j - 1) : p_j + 1]

    if t_seg.size:
        t_peak_val = float(t_seg[int(np.argmax(np.abs(t_seg)))])
    else:
        t_peak_val = 0.0

    baseline = float(np.median(template))
    t_amp = abs(t_peak_val - baseline) if t_seg.size else 0.0
    ratio = cfg.record_template_t_r_amplitude_ratio
    fixed_p = cfg.record_template_fixed_p_mv

    inverted_t = t_peak_val < baseline if t_seg.size else False
    large_t = t_amp >= ratio * max(mean_r_amplitude, 1e-6)

    if inverted_t:
        th_t_down = float(np.min(t_seg)) / 2.0 if t_seg.size else -0.05
        th_t_up = -th_t_down
        morph = "inverted_t"
    else:
        th_t_up = float(np.max(t_seg)) / 2.0 if t_seg.size else 0.05
        th_t_down = -th_t_up
        morph = "large_t" if large_t else "normal"

    if large_t and not inverted_t:
        th_p_up = fixed_p
        th_p_down = -fixed_p
    else:
        th_p_up = float(np.max(p_seg)) / 2.0 if p_seg.size else fixed_p * 0.5
        th_p_down = -th_p_up

    return th_t_up, th_t_down, th_p_up, th_p_down, morph


def finalize_record_t_median_template(
    template: np.ndarray,
    cfg: ProcessCycleConfig,
    sampling_rate: float,
    *,
    pre_r_samples: int,
    median_rr_samples: float,
    n_beats: int,
    mean_r_amplitude: float = 1.0,
    beat_anchors: Optional[List[Tuple[int, int, int]]] = None,
    ecg_work: Optional[np.ndarray] = None,
) -> MedianBeatTemplate:
    """Attach record-T landmarks/thresholds to a median S→Q template waveform."""
    template = np.asarray(template, dtype=float)
    t_j, p_j, t_landmark_source = _landmarks_closest_to_baseline(
        template, cfg, sampling_rate
    )
    from pyhearts.processing.record_template_biphasic import (
        apply_biphasic_positive_negative_landmark,
        biphasic_pm_classification_enabled,
        biphasic_pm_lobe_search_enabled,
        classify_biphasic_positive_negative,
    )

    morph_bi = None
    t_biphasic_pos = None
    t_biphasic_neg = None
    if biphasic_pm_classification_enabled(cfg):
        bi_tag, pos_i, neg_i = classify_biphasic_positive_negative(
            template, cfg, sampling_rate
        )
        if bi_tag == "biphasic_positive_negative" and pos_i is not None and neg_i is not None:
            t_biphasic_pos = float(pos_i)
            t_biphasic_neg = float(neg_i)
            morph_bi = bi_tag
            if biphasic_pm_lobe_search_enabled(cfg):
                t_j, t_landmark_source, morph_bi, t_biphasic_pos, t_biphasic_neg = (
                    apply_biphasic_positive_negative_landmark(
                        t_j, t_landmark_source, int(pos_i), int(neg_i)
                    )
                )
                t_biphasic_pos = float(t_biphasic_pos)
                t_biphasic_neg = float(t_biphasic_neg)

    th_t_up, th_t_down, th_p_up, th_p_down, morph = _compute_template_thresholds(
        template, t_j, p_j, mean_r_amplitude, cfg
    )
    if morph_bi is not None:
        morph = morph_bi

    from pyhearts.processing.record_post_apex_dz_morphology import (
        classify_post_apex_dz_preference_template,
    )

    probe_tmpl = type(
        "_Probe",
        (),
        {
            "valid": True,
            "template": template,
            "t_landmark_idx": t_j,
            "p_landmark_idx": p_j,
            "t_landmark_source": t_landmark_source,
            "t_morphology": morph,
            "t_polarity": "positive",
        },
    )()
    post_apex_dz = classify_post_apex_dz_preference_template(
        probe_tmpl,
        cfg,
        sampling_rate,
        beat_anchors=beat_anchors,
        ecg_work=ecg_work,
    )

    return MedianBeatTemplate(
        template=template,
        pre_r_samples=pre_r_samples,
        r_center_idx=0,
        p_offset_samples=None,
        t_offset_samples=None,
        p_polarity="positive",
        t_polarity="negative",
        median_rr_samples=median_rr_samples,
        n_beats=n_beats,
        valid=True,
        template_anchor="s_to_q",
        t_landmark_idx=t_j,
        p_landmark_idx=p_j,
        th_t_up=th_t_up,
        th_t_down=th_t_down,
        th_p_up=th_p_up,
        th_p_down=th_p_down,
        t_morphology=morph,
        t_landmark_source=t_landmark_source,
        t_biphasic_pos_landmark_idx=t_biphasic_pos,
        t_biphasic_neg_landmark_idx=t_biphasic_neg,
        t_post_apex_dz_preference=post_apex_dz,
    )


def build_record_t_beat_template(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> MedianBeatTemplate:
    """
    Ensemble template from S(i) → Q(i+1) segments (first ``record_template_max_duration_s``).
    """
    ecg_work = prepare_record_delineation_signal(ecg, sampling_rate, cfg)
    ecg = np.asarray(ecg, dtype=float)
    r_peaks = np.asarray(r_peaks, dtype=int)
    max_sample = int(cfg.record_template_max_duration_s * sampling_rate)
    qs_half = _ms_to_samples(cfg.record_qs_search_window_ms, sampling_rate)

    segments: List[np.ndarray] = []
    record_t_anchors: List[Tuple[int, int, int]] = []
    r_for_amp: List[int] = []

    for i in range(len(r_peaks) - 1):
        r_i = int(r_peaks[i])
        if r_i >= max_sample:
            break
        r_for_amp.append(r_i)
        s_i = _find_s_after_r(ecg_work, r_i, sampling_rate, cfg)
        if s_i is None:
            s_i = min(len(ecg_work) - 1, r_i + max(1, qs_half // 2))
        q_next = _find_q_before_r(ecg_work, int(r_peaks[i + 1]), sampling_rate, cfg)
        if q_next is None:
            q_next = max(s_i + 3, int(r_peaks[i + 1]) - max(1, qs_half // 2))
        if q_next <= s_i + 3:
            continue
        seg = ecg_work[s_i:q_next]
        if seg.size >= 8:
            segments.append(seg)
            record_t_anchors.append((int(s_i), int(q_next), int(r_i)))

    pre_r = _epoch_half_width(r_peaks, sampling_rate, cfg)
    if len(segments) < cfg.record_delineation_min_beats:
        return MedianBeatTemplate(
            template=np.array([]),
            pre_r_samples=pre_r,
            r_center_idx=0,
            p_offset_samples=None,
            t_offset_samples=None,
            p_polarity="positive",
            t_polarity="negative",
            median_rr_samples=float(_ms_to_samples(800.0, sampling_rate)),
            n_beats=len(segments),
            valid=False,
            template_anchor="s_to_q",
        )

    target_len = int(np.median([s.size for s in segments]))
    target_len = max(8, target_len)
    resampled = np.vstack([_resample_segment(s, target_len) for s in segments])
    if cfg.record_template_aggregate == "mean":
        template = np.mean(resampled, axis=0)
    else:
        template = np.median(resampled, axis=0)

    rr = np.diff(r_peaks.astype(float))
    if rr.size:
        min_rr = _ms_to_samples(cfg.rr_bounds_ms[0], sampling_rate)
        max_rr = _ms_to_samples(cfg.rr_bounds_ms[1], sampling_rate)
        rr = np.clip(rr, min_rr, max_rr)
        median_rr = float(np.median(rr))
    else:
        median_rr = float(2 * pre_r)

    r_amps = [abs(ecg[r]) for r in r_for_amp if 0 <= r < ecg.size]
    mean_r = float(np.median(r_amps)) if r_amps else 1.0
    return finalize_record_t_median_template(
        template,
        cfg,
        sampling_rate,
        pre_r_samples=pre_r,
        median_rr_samples=median_rr,
        n_beats=len(segments),
        mean_r_amplitude=mean_r,
        beat_anchors=record_t_anchors,
        ecg_work=ecg_work,
    )


def build_record_beat_template(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> MedianBeatTemplate:
    """Dispatch R-centered median beat vs S→Q record-T template."""
    if cfg.record_template_anchor == "s_to_q":
        return build_record_t_beat_template(ecg, r_peaks, sampling_rate, cfg)
    return build_median_beat_template(ecg, r_peaks, sampling_rate, cfg)


def _epoch_half_width(r_peaks: np.ndarray, sampling_rate: float, cfg: ProcessCycleConfig) -> int:
    rr = np.diff(r_peaks.astype(int))
    if rr.size == 0:
        rr = np.array([int(round(0.8 * sampling_rate))])
    min_rr = max(1, _ms_to_samples(cfg.rr_bounds_ms[0], sampling_rate))
    max_rr = max(min_rr, _ms_to_samples(cfg.rr_bounds_ms[1], sampling_rate))
    rr = np.clip(rr, min_rr, max_rr)
    pre_r = cfg.pre_r_window
    if pre_r is None:
        pre_r = int(round(float(np.median(rr)) / 2.0))
    return max(2, int(pre_r))


def _extract_detrended_beat(
    ecg: np.ndarray,
    r_peak: int,
    pre_r: int,
    *,
    linear_detrend: bool = True,
) -> Optional[np.ndarray]:
    start = int(r_peak) - pre_r
    end = int(r_peak) + pre_r
    if start < 0 or end > ecg.size or end - start != 2 * pre_r:
        return None
    window = ecg[start:end].astype(float, copy=True)
    if not linear_detrend:
        return window
    n = window.size
    x = np.arange(n, dtype=float)
    x_mean = (n - 1) / 2.0
    y_mean = float(np.mean(window))
    denom = float(np.sum((x - x_mean) ** 2))
    slope = float(np.dot(x - x_mean, window - y_mean) / denom) if denom > 0 else 0.0
    trend = slope * (x - x_mean) + y_mean
    return window - trend


def build_median_beat_template(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> MedianBeatTemplate:
    """Stack detrended beats aligned at R; return median template and metadata."""
    ecg = np.asarray(ecg, dtype=float)
    ecg_stack = prepare_record_delineation_signal(ecg, sampling_rate, cfg)
    linear_detrend = cfg.delineation_baseline_method != "median_record"
    r_peaks = np.asarray(r_peaks, dtype=int)
    pre_r = _epoch_half_width(r_peaks, sampling_rate, cfg)
    beats: List[np.ndarray] = []

    for r_peak in r_peaks:
        beat = _extract_detrended_beat(
            ecg_stack, int(r_peak), pre_r, linear_detrend=linear_detrend
        )
        if beat is not None:
            beats.append(beat)

    if len(beats) < cfg.record_delineation_min_beats:
        return MedianBeatTemplate(
            template=np.array([]),
            pre_r_samples=pre_r,
            r_center_idx=pre_r,
            p_offset_samples=None,
            t_offset_samples=None,
            p_polarity="positive",
            t_polarity="negative",
            median_rr_samples=float(_ms_to_samples(800.0, sampling_rate)),
            n_beats=len(beats),
            valid=False,
        )

    stack = np.vstack(beats)
    template = np.median(stack, axis=0)
    rr = np.diff(r_peaks.astype(float))
    if rr.size:
        min_rr = _ms_to_samples(cfg.rr_bounds_ms[0], sampling_rate)
        max_rr = _ms_to_samples(cfg.rr_bounds_ms[1], sampling_rate)
        rr = np.clip(rr, min_rr, max_rr)
        median_rr = float(np.median(rr))
    else:
        median_rr = float(2 * pre_r)

    return MedianBeatTemplate(
        template=template,
        pre_r_samples=pre_r,
        r_center_idx=pre_r,
        p_offset_samples=None,
        t_offset_samples=None,
        p_polarity="positive",
        t_polarity="negative",
        median_rr_samples=median_rr,
        n_beats=len(beats),
        valid=True,
    )


def _apex_in_segment(
    segment: np.ndarray,
    *,
    prefer: str,
) -> Optional[int]:
    if segment.size < 3:
        return None
    if prefer == "max":
        return int(np.argmax(segment))
    return int(np.argmin(segment))


def _bandpass_segment(
    segment: np.ndarray,
    sampling_rate: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> np.ndarray:
    if segment.size < order * 3:
        return segment
    nyq = sampling_rate / 2.0
    lo = max(low_hz / nyq, 1e-5)
    hi = min(high_hz / nyq, 0.99)
    if lo >= hi:
        return segment
    b, a = butter(order, [lo, hi], btype="band")
    return filtfilt(b, a, segment.astype(float))


def delineate_record_t_template(
    template: MedianBeatTemplate,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    *,
    manual_ann_ext: Optional[str] = None,
) -> MedianBeatTemplate:
    """Locate P/T on record-T template with beat-scaled windows and optional template thresholds."""
    if not template.valid or template.template.size < 10:
        return template
    if template.t_landmark_idx is None or template.p_landmark_idx is None:
        return template

    fs = float(sampling_rate)
    sig = savgol_search_segment(template.template, fs, cfg)
    t_j = int(template.t_landmark_idx)
    p_j = int(template.p_landmark_idx)
    from pyhearts.processing.record_t_detection import (
        _record_t_search_end_tpl_idx,
        _t_search_prefer_negative,
        record_t_use_biphasic_fallback,
    )

    # Standard wide template T window — beat-level early_peak narrowing is applied later.
    t_search_end = int(
        round(
            _record_t_search_end_tpl_idx(
                float(t_j),
                float(p_j),
                cfg,
                early_peak=False,
            )
        )
    )
    t_search_end = max(t_j + 1, t_search_end)
    p_inner = int(t_j + (p_j - t_j) / 3) if p_j > t_j else t_j + 1
    L = len(sig)

    t_seg, t_lo, t_hi = smooth_search_window(sig, t_j, t_search_end, fs, cfg)
    th_t_up = template.th_t_up or 0.05
    th_t_down = template.th_t_down or -0.05
    if cfg.p_t_threshold_mode == "template":
        t_rel, t_pol = _apex_with_threshold(
            t_seg,
            prefer="min" if _t_search_prefer_negative(template) else "max",
            threshold_up=th_t_up,
            threshold_down=th_t_down,
            check_biphasic=record_t_use_biphasic_fallback(template),
        )
    else:
        t_rel = _apex_in_segment(t_seg, prefer="min")
        t_pol = "negative"
        if t_rel is None:
            t_rel = _apex_in_segment(t_seg, prefer="max")
            t_pol = "positive"
    t_peak_idx = (t_lo + t_rel) if t_rel is not None else None

    # P window: from (L-1-p_j) before Q end to one-third T–P span before Q
    p_far = max(0, (L - 1) - p_j)
    p_near = max(0, int((p_j - t_j) / 3))
    p_start = min(L - 3, max(0, L - 1 - p_far))
    p_end = min(L, max(p_start + 3, L - 1 - p_near))
    p_seg, p_lo, p_hi = smooth_search_window(sig, p_start, p_end, fs, cfg)
    th_p_up = template.th_p_up or 0.05
    th_p_down = template.th_p_down or -0.05
    if cfg.p_t_threshold_mode == "template":
        p_rel, p_pol = _apex_with_threshold(
            p_seg,
            prefer="max",
            threshold_up=th_p_up,
            threshold_down=th_p_down,
            check_biphasic=False,
        )
    else:
        p_rel = _apex_in_segment(p_seg, prefer="max")
        p_pol = "positive" if p_rel is not None and p_seg[p_rel] >= 0 else "negative"
    p_peak_idx = (p_lo + p_rel) if p_rel is not None else None

    preserve_negative_polarity = defer_record_t_overwrite(
        template, manual_ann_ext=manual_ann_ext
    ) or (
        _is_early_peak_landmark(template)
        and str(getattr(template, "t_morphology", "normal") or "normal") == "normal"
        and _early_peak_landmark_frac(template) <= 0.21
    )

    return MedianBeatTemplate(
        template=template.template,
        pre_r_samples=template.pre_r_samples,
        r_center_idx=template.r_center_idx,
        p_offset_samples=float(p_peak_idx) if p_peak_idx is not None else None,
        t_offset_samples=float(t_peak_idx) if t_peak_idx is not None else None,
        p_polarity=p_pol if p_peak_idx is not None else "positive",
        t_polarity=(
            template.t_polarity
            if preserve_negative_polarity
            else (t_pol if t_peak_idx is not None else template.t_polarity)
        ),
        median_rr_samples=template.median_rr_samples,
        n_beats=template.n_beats,
        valid=True,
        template_anchor="s_to_q",
        t_landmark_idx=template.t_landmark_idx,
        p_landmark_idx=template.p_landmark_idx,
        th_t_up=template.th_t_up,
        th_t_down=template.th_t_down,
        th_p_up=template.th_p_up,
        th_p_down=template.th_p_down,
        t_morphology=template.t_morphology,
        t_landmark_source=template.t_landmark_source,
        t_biphasic_pos_landmark_idx=template.t_biphasic_pos_landmark_idx,
        t_biphasic_neg_landmark_idx=template.t_biphasic_neg_landmark_idx,
        t_post_apex_dz_preference=template.t_post_apex_dz_preference,
    )


def delineate_record_template(
    template: MedianBeatTemplate,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    *,
    manual_ann_ext: Optional[str] = None,
) -> MedianBeatTemplate:
    if template.template_anchor == "s_to_q":
        return delineate_record_t_template(
            template, sampling_rate, cfg, manual_ann_ext=manual_ann_ext
        )
    return delineate_median_beat_template(template, sampling_rate, cfg)


def delineate_median_beat_template(
    template: MedianBeatTemplate,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> MedianBeatTemplate:
    """Find P/T apex on median beat; fill offset fields."""
    if not template.valid or template.template.size < 10:
        return template

    sig = savgol_search_segment(template.template, float(sampling_rate), cfg)
    fs = float(sampling_rate)
    r_idx = template.r_center_idx

    p_lo = max(0, r_idx - _ms_to_samples(cfg.record_delineation_p_search_before_r_ms, fs))
    p_hi = max(p_lo + 3, r_idx - _ms_to_samples(cfg.record_delineation_p_search_end_before_r_ms, fs))
    p_seg = sig[p_lo:p_hi]
    if cfg.pwave_use_bandpass and p_seg.size >= 8:
        p_seg = _bandpass_segment(
            p_seg,
            fs,
            cfg.pwave_bandpass_low_hz,
            cfg.pwave_bandpass_high_hz,
            cfg.pwave_bandpass_order,
        )
    p_rel = _apex_in_segment(p_seg, prefer="max")
    if p_rel is None:
        p_rel = _apex_in_segment(p_seg, prefer="min")
        p_pol = "negative"
    else:
        p_pol = "positive" if p_seg[p_rel] >= 0 else "negative"
        if p_pol == "positive" and np.ptp(p_seg) > 0:
            alt = _apex_in_segment(p_seg, prefer="min")
            if alt is not None and abs(p_seg[alt]) > abs(p_seg[p_rel]):
                p_rel, p_pol = alt, "negative"

    p_offset = float(p_lo + p_rel - r_idx) if p_rel is not None else None

    t_lo = min(len(sig) - 3, r_idx + _ms_to_samples(cfg.record_delineation_t_search_after_r_ms, fs))
    t_hi = min(len(sig), r_idx + _ms_to_samples(cfg.record_delineation_t_search_end_ms, fs))
    if t_hi <= t_lo + 3:
        t_hi = min(len(sig), t_lo + max(3, _ms_to_samples(120.0, fs)))
    t_seg = sig[t_lo:t_hi]
    t_rel = _apex_in_segment(t_seg, prefer="min")
    if t_rel is None:
        t_rel = _apex_in_segment(t_seg, prefer="max")
        t_pol = "positive"
    else:
        t_pol = "negative" if t_seg[t_rel] <= 0 else "positive"
        if t_pol == "negative" and np.ptp(t_seg) > 0:
            alt = _apex_in_segment(t_seg, prefer="max")
            if alt is not None and abs(t_seg[alt]) > abs(t_seg[t_rel]):
                t_rel, t_pol = alt, "positive"

    t_offset = float(t_lo + t_rel - r_idx) if t_rel is not None else None

    return MedianBeatTemplate(
        template=template.template,
        pre_r_samples=template.pre_r_samples,
        r_center_idx=template.r_center_idx,
        p_offset_samples=p_offset,
        t_offset_samples=t_offset,
        p_polarity=p_pol if p_rel is not None else "positive",
        t_polarity=t_pol if t_rel is not None else "negative",
        median_rr_samples=template.median_rr_samples,
        n_beats=template.n_beats,
        valid=True,
    )


def _local_rr_samples(
    cycle_idx: int,
    r_peaks: np.ndarray,
    cycle_labels: np.ndarray,
    median_rr: float,
) -> float:
    epoch_i = int(cycle_labels[cycle_idx])
    if epoch_i < 0 or epoch_i >= len(r_peaks):
        return median_rr
    if epoch_i > 0:
        return float(r_peaks[epoch_i] - r_peaks[epoch_i - 1])
    if epoch_i + 1 < len(r_peaks):
        return float(r_peaks[epoch_i + 1] - r_peaks[epoch_i])
    return median_rr


def _record_t_anchors_valid(
    r_det: int,
    r_next: Optional[int],
    s_i: int,
    q_next: int,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> bool:
    """Beat-specific Q/S plausibility (S after R, Q before next R, ordered S < Q)."""
    fs = float(sampling_rate)
    min_after_r = max(1, _ms_to_samples(15.0, fs))
    max_s_after_r = _ms_to_samples(cfg.record_s_search_after_r_ms, fs)
    if not (r_det + min_after_r <= s_i <= r_det + max_s_after_r):
        return False
    if q_next <= s_i + max(3, _ms_to_samples(40.0, fs)):
        return False
    if r_next is not None:
        max_q_before_r = _ms_to_samples(cfg.record_q_search_before_r_ms, fs)
        if not (s_i < q_next <= int(r_next) + _ms_to_samples(30.0, fs)):
            return False
        if q_next < int(r_next) - max_q_before_r:
            return False
    return True


def _pt_expected_offset(
    tmpl_offset: Optional[float],
    wavelet_offset: Optional[float],
    cfg: ProcessCycleConfig,
) -> Optional[float]:
    """Template-offset units: wavelet prior when enabled, else median template."""
    if cfg.record_wavelet_pt_prior and wavelet_offset is not None:
        return wavelet_offset
    return tmpl_offset


def _record_t_s_q_anchor_indices(
    ecg_delim: np.ndarray,
    r_det: int,
    r_next: Optional[int],
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> Tuple[Optional[int], Optional[int]]:
    """Beat-local S and next-beat Q sample indices for s_to_q record-T (diagnostics)."""
    if r_next is None:
        return None, None
    from pyhearts.processing.qrs_extrema import (
        find_q_wave_before_r,
        find_s_wave_after_r,
        q_search_before_r_ms,
        s_search_after_r_ms,
    )

    s_i = find_s_wave_after_r(
        ecg_delim, r_det, sampling_rate, search_window_ms=s_search_after_r_ms(cfg), inverted=False
    )
    q_next = find_q_wave_before_r(
        ecg_delim, int(r_next), sampling_rate, search_window_ms=q_search_before_r_ms(cfg), inverted=False
    )
    if s_i is None or q_next is None:
        return None, None
    return int(s_i), int(q_next)


def _record_t_pt_guesses_for_beat(
    ecg_delim: np.ndarray,
    r_det: int,
    r_next: Optional[int],
    tmpl: MedianBeatTemplate,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    scale: float,
    *,
    p_expected_offset: Optional[float] = None,
    t_expected_offset: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Dispatch R-centered record-T vs S→Q beat-wise record search."""
    if (
        tmpl.template_anchor == "s_to_q"
        and cfg.p_t_threshold_mode == "template"
        and cfg.record_delineation_t_search
    ):
        from pyhearts.processing.record_t_detection import record_t_pt_guesses

        wavelet_off = (
            float(p_expected_offset) * float(scale)
            if p_expected_offset is not None
            else None
        )
        return record_t_pt_guesses(
            ecg_delim,
            r_det,
            r_next,
            tmpl,
            sampling_rate,
            cfg,
            scale,
            wavelet_pr_offset_samples=wavelet_off,
        )
    return _record_t_pt_guesses(
        ecg_delim,
        r_det,
        r_next,
        tmpl,
        sampling_rate,
        cfg,
        scale,
        p_expected_offset=p_expected_offset,
        t_expected_offset=t_expected_offset,
    )


def _record_t_pt_guesses(
    ecg_delim: np.ndarray,
    r_det: int,
    r_next: Optional[int],
    tmpl: MedianBeatTemplate,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    scale: float,
    *,
    p_expected_offset: Optional[float] = None,
    t_expected_offset: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Map record-T template indices to global P/T guesses with beat-specific Q/S validation.
    """
    t_guess: Optional[float] = None
    p_guess: Optional[float] = None
    fs = float(sampling_rate)

    s_i = _find_s_after_r(ecg_delim, r_det, fs, cfg)
    if s_i is None:
        return None, None

    q_next: Optional[int] = None
    if r_next is not None:
        q_next = _find_q_before_r(ecg_delim, int(r_next), fs, cfg)
        if q_next is None:
            return None, None
        if not _record_t_anchors_valid(r_det, r_next, s_i, q_next, fs, cfg):
            return None, None

    t_off_base = _pt_expected_offset(
        tmpl.t_offset_samples, t_expected_offset, cfg
    )
    p_off_base = _pt_expected_offset(
        tmpl.p_offset_samples, p_expected_offset, cfg
    )
    if (
        t_off_base is not None
        and tmpl.t_landmark_idx is not None
        and tmpl.p_landmark_idx is not None
    ):
        t_j = float(tmpl.t_landmark_idx)
        p_j = float(tmpl.p_landmark_idx)
        mid_tp = t_j + (p_j - t_j) / 2.0
        t_off = float(t_off_base) * scale
        t_lo = float(s_i) + min(t_off, mid_tp * scale)
        t_hi = float(s_i) + mid_tp * scale
        if r_next is not None:
            t_hi = min(t_hi, float(r_next) - _ms_to_samples(80.0, fs))
        if t_hi > t_lo + 3:
            t_guess = (t_lo + t_hi) / 2.0
        else:
            t_guess = float(s_i) + t_off
        if r_next is not None and t_guess >= float(r_next) - _ms_to_samples(60.0, fs):
            return None, None
        if t_guess <= float(s_i) + _ms_to_samples(60.0, fs):
            return None, None

    if p_off_base is not None and q_next is not None and tmpl.template.size > 0:
        L = tmpl.template.size
        p_j = float(tmpl.p_landmark_idx or 0)
        t_j = float(tmpl.t_landmark_idx or 0)
        dist_q_to_p = (L - 1) - float(p_off_base) * scale
        p_far = float(q_next) - dist_q_to_p
        p_near = float(q_next) - max(1.0, (p_j - t_j) / 3.0) * scale
        p_lo = min(p_far, p_near)
        p_hi = max(p_far, p_near)
        if p_hi > p_lo + 3 and p_hi < float(r_next if r_next is not None else q_next):
            p_guess = (p_lo + p_hi) / 2.0
        if p_guess is not None and p_guess >= float(r_det) - _ms_to_samples(30.0, fs):
            p_guess = None

    return t_guess, p_guess


def _sync_peak(
    output_dict: Dict,
    cycle_idx: int,
    wave: str,
    global_idx: float,
    one_cycle: pd.DataFrame,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> None:
    if "index" in one_cycle.columns:
        xs = one_cycle["index"].values.astype(int)
    else:
        xs = one_cycle["signal_x"].values.astype(int)
    sig = one_cycle["signal_y"].values.astype(float)
    if len(xs) == 0:
        return

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

    if cfg.use_subsample_peak_refinement and sig.size >= 3:
        anchor = int(np.clip(np.round(best_rel), 1, sig.size - 2))
        from pyhearts.processing.peaks import refine_peak_parabolic

        best_rel = refine_peak_parabolic(sig, anchor)

    center_key = f"{wave}_center_idx"
    volt_key = f"{wave}_center_voltage"
    ms_key = f"{wave}_center_ms"
    global_key = f"{wave}_global_center_idx"

    output_dict[global_key][cycle_idx] = float(global_idx)
    output_dict[center_key][cycle_idx] = best_rel
    if cfg.use_subsample_peak_refinement:
        output_dict[volt_key][cycle_idx] = sample_at_fractional_index(sig, best_rel)
    else:
        i_n = int(np.clip(np.round(best_rel), 0, len(sig) - 1))
        output_dict[volt_key][cycle_idx] = float(sig[i_n])
    output_dict[ms_key][cycle_idx] = (best_rel / sampling_rate) * 1000.0


def _record_pt_source(cfg: ProcessCycleConfig) -> str:
    if cfg.record_delineation_t_search and cfg.record_template_anchor == "s_to_q":
        return "record_t"
    return "record_template"


def _template_offset_guess(
    r_g: float,
    offset_samples: Optional[float],
    scale: float,
) -> Optional[float]:
    if offset_samples is None:
        return None
    return float(r_g) + float(offset_samples) * scale


def _resolve_p_guess(
    *,
    ecg_delim: np.ndarray,
    r_det: int,
    r_next: Optional[int],
    r_g: float,
    tmpl: MedianBeatTemplate,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    scale: float,
    stats: Dict[str, int],
    p_expected_offset: Optional[float] = None,
    t_expected_offset: Optional[float] = None,
) -> Tuple[Optional[float], str]:
    """record-T guess with optional template-delay fallback. Returns (guess, source tag)."""
    p_off = _pt_expected_offset(tmpl.p_offset_samples, p_expected_offset, cfg)
    if tmpl.template_anchor == "s_to_q":
        _, p_guess_f = _record_t_pt_guesses_for_beat(
            ecg_delim,
            r_det,
            r_next,
            tmpl,
            sampling_rate,
            cfg,
            scale,
            p_expected_offset=p_expected_offset,
            t_expected_offset=t_expected_offset,
        )
        if p_guess_f is not None:
            return p_guess_f, _record_pt_source(cfg)
        stats["p_record_miss"] = stats.get("p_record_miss", 0) + 1
        if cfg.record_delineation_template_fallback:
            if cfg.record_t_p_r_anchor:
                from pyhearts.processing.record_t_detection import record_fallback_p_search

                wavelet_off = (
                    float(p_expected_offset) * float(scale)
                    if p_expected_offset is not None
                    else None
                )
                p_fb = record_fallback_p_search(
                    ecg_delim,
                    int(r_det),
                    r_next,
                    tmpl,
                    sampling_rate,
                    cfg,
                    wavelet_pr_offset_samples=wavelet_off,
                )
                if p_fb is not None:
                    stats["p_template_fallback"] = stats.get("p_template_fallback", 0) + 1
                    return p_fb, "record_template_fallback"
            elif p_off is not None:
                guess = _template_offset_guess(r_g, p_off, scale)
                if guess is not None:
                    stats["p_template_fallback"] = stats.get("p_template_fallback", 0) + 1
                    src = (
                        "record_wavelet_fallback"
                        if cfg.record_wavelet_pt_prior and p_expected_offset is not None
                        else "record_template_fallback"
                    )
                    return guess, src
        return None, ""
    guess = _template_offset_guess(r_g, p_off, scale)
    if guess is None:
        return None, ""
    return guess, "record_template"


def _resolve_t_guess(
    *,
    ecg_delim: np.ndarray,
    r_det: int,
    r_next: Optional[int],
    r_g: float,
    tmpl: MedianBeatTemplate,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    scale: float,
    stats: Dict[str, int],
    t_expected_offset: Optional[float] = None,
    p_expected_offset: Optional[float] = None,
) -> Tuple[Optional[float], str]:
    t_off = _pt_expected_offset(tmpl.t_offset_samples, t_expected_offset, cfg)
    if tmpl.template_anchor == "s_to_q":
        t_guess_f, _ = _record_t_pt_guesses_for_beat(
            ecg_delim,
            r_det,
            r_next,
            tmpl,
            sampling_rate,
            cfg,
            scale,
            p_expected_offset=p_expected_offset,
            t_expected_offset=t_expected_offset,
        )
        if t_guess_f is not None:
            return t_guess_f, _record_pt_source(cfg)
        stats["t_record_miss"] = stats.get("t_record_miss", 0) + 1
        if cfg.record_delineation_template_fallback:
            guess = _template_offset_guess(r_g, t_off, scale)
            if guess is not None:
                stats["t_template_fallback"] = stats.get("t_template_fallback", 0) + 1
                src = (
                    "record_wavelet_fallback"
                    if cfg.record_wavelet_pt_prior and t_expected_offset is not None
                    else "record_template_fallback"
                )
                return guess, src
        return None, ""
    guess = _template_offset_guess(r_g, t_off, scale)
    if guess is None:
        return None, ""
    return guess, "record_template"


