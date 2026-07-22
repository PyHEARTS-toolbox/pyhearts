"""
Record-level P/T detection via record-T template windows (w1, w2).

Beat-wise search on median-baseline delineation signal (optional Savitzky–Golay),
anchored to S/Q from local QRS extrema, with template-derived amplitude thresholds.

P search (when ``record_t_p_r_anchor``): fixed PR window before ``current_r`` (q1c)
or ``next_r`` (S→Q beat assignment), not S→Q fraction w2.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import numpy as np

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.delineation_signal import smooth_search_window
from pyhearts.processing.qrs_extrema import (
    find_q_wave_before_r,
    find_s_wave_after_r,
    q_search_before_r_ms,
    s_search_after_r_ms,
)

PAnchorMode = Literal["current_r", "next_r"]


def _apex_with_threshold(
    segment: np.ndarray,
    *,
    prefer: str,
    threshold_up: float,
    threshold_down: float,
    check_biphasic: bool = False,
) -> Tuple[Optional[int], str]:
    if segment.size < 2:
        return None, prefer
    i_max = int(np.argmax(segment))
    i_min = int(np.argmin(segment))
    candidates: List[Tuple[int, str, float]] = []
    if segment[i_max] >= threshold_up:
        candidates.append((i_max, "positive", float(segment[i_max])))
    if segment[i_min] <= threshold_down:
        candidates.append((i_min, "negative", float(segment[i_min])))
    if check_biphasic and not candidates:
        if abs(segment[i_max]) >= abs(threshold_up):
            candidates.append((i_max, "positive", float(segment[i_max])))
        if abs(segment[i_min]) >= abs(threshold_down):
            candidates.append((i_min, "negative", float(segment[i_min])))
    if candidates:
        full_candidates = list(candidates)
        if prefer == "max":
            same_pol = [c for c in candidates if c[2] > 0.0]
        elif prefer == "min":
            same_pol = [c for c in candidates if c[2] < 0.0]
        else:
            same_pol = candidates
        candidates = same_pol if same_pol else full_candidates
    if not candidates:
        if prefer == "max":
            return i_max, "positive" if segment[i_max] >= 0 else "negative"
        return i_min, "negative" if segment[i_min] <= 0 else "positive"
    best = max(candidates, key=lambda c: abs(c[2]))
    return best[0], best[1]


def _tpl_index_to_sample(
    s_i: int,
    q_next: int,
    tpl_idx: float,
    n_tpl: int,
) -> int:
    """Map record-T template index (0=S … n-1≈Q) to a full-signal sample."""
    if n_tpl < 2:
        return int(s_i)
    frac = float(np.clip(tpl_idx, 0, n_tpl - 1)) / float(n_tpl - 1)
    return int(round(s_i + frac * (q_next - s_i)))


def _ms_before_r(sampling_rate: float, ms: float) -> int:
    return int(round(ms * sampling_rate / 1000.0))


def _inverted_plateau_apex_forward(
    ecg: np.ndarray,
    s_i: int,
    q_next: int,
    tmpl,
    t_lo: int,
    t_hi: int,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> Optional[Tuple[int, str]]:
    """
    Beat-level trough search for plateau_apex + inverted_t: latest sample within 95% of
    max |amp| in [t_j, t_j + forward_ms] ∩ w1 (flat inverted plateau body, not onset).
    """
    if getattr(tmpl, "t_landmark_source", "") != "plateau_apex":
        return None
    if getattr(tmpl, "t_morphology", "") != "inverted_t":
        return None
    if tmpl.t_landmark_idx is None:
        return None
    n_tpl = tmpl.template.size
    if n_tpl < 2:
        return None
    anchor = _tpl_index_to_sample(s_i, q_next, float(tmpl.t_landmark_idx), n_tpl)
    forward = _ms_before_r(sampling_rate, cfg.record_t_plateau_apex_forward_ms)
    search_lo = max(int(t_lo), int(anchor))
    search_hi = min(int(t_hi), int(anchor) + forward)
    if search_hi <= search_lo:
        return None
    seg = ecg[search_lo : search_hi + 1].astype(float, copy=False)
    if seg.size < 1:
        return None
    abs_seg = np.abs(seg)
    peak_abs = float(np.max(abs_seg))
    if peak_abs <= 0.0:
        return None
    # Flat inverted plateaus: pick the latest sample near max |amp| (trough body, not onset).
    near = np.flatnonzero(abs_seg >= 0.95 * peak_abs)
    rel = int(near[-1]) if near.size else int(np.argmax(abs_seg))
    idx = search_lo + rel
    pol = "negative" if float(seg[rel]) <= 0.0 else "positive"
    return int(idx), pol


def _p_r_anchor(
    r_det: int,
    r_next: Optional[int],
    cfg: ProcessCycleConfig,
) -> Optional[int]:
    if not cfg.record_t_p_r_anchor:
        return None
    mode = getattr(cfg, "record_t_p_r_anchor_mode", "current_r")
    if mode == "current_r":
        return int(r_det)
    if mode == "next_r" and r_next is not None:
        return int(r_next)
    return None


def p_pr_window_samples(
    r_anchor: int,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    *,
    signal_len: Optional[int] = None,
    s_i: Optional[int] = None,
    q_next: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Physiological P window: R_anchor − PR_max … R_anchor − PR_min.

    Optional ``s_i``/``q_next`` clamps apply for next-R Rahul S→Q context only.
    """
    p_lo = int(r_anchor) - _ms_before_r(sampling_rate, cfg.record_t_p_pr_max_ms)
    p_hi = int(r_anchor) - _ms_before_r(sampling_rate, cfg.record_t_p_pr_min_ms)
    p_lo = max(0, p_lo)
    p_hi = min(int(r_anchor) - 1, p_hi)
    if signal_len is not None:
        p_hi = min(int(signal_len) - 1, p_hi)
    if s_i is not None and q_next is not None:
        p_lo = max(int(s_i) + 3, p_lo)
        p_hi = min(int(q_next) - 1, p_hi)
    if p_hi < p_lo + 3:
        p_hi = min(
            int(r_anchor) - 1,
            (int(signal_len) - 1 if signal_len is not None else p_lo + 3),
            p_lo + 3,
        )
    return p_lo, p_hi


def _is_early_peak_landmark(tmpl) -> bool:
    return str(getattr(tmpl, "t_landmark_source", "") or "") == "early_peak"


def _early_peak_landmark_frac(tmpl) -> float:
    tpl = getattr(tmpl, "template", None)
    n = len(tpl) if tpl is not None else 0
    t_j = getattr(tmpl, "t_landmark_idx", None)
    if n < 2 or t_j is None:
        return 1.0
    return float(t_j) / float(n - 1)


def _suppress_early_peak_beat_tuning(tmpl) -> bool:
    """
    sel301-class: landmark very early in the S→Q template — use wide record-T + smooth,
    not narrow per-beat early_peak search (differs from sel230 / sele0114).
    """
    if tmpl is None or not _is_early_peak_landmark(tmpl):
        return False
    if str(getattr(tmpl, "t_morphology", "normal") or "normal") != "normal":
        return False
    return _early_peak_landmark_frac(tmpl) <= 0.21


def _early_peak_record_t_tuning_applies(tmpl) -> bool:
    """
    Beat-level early_peak record-T for upright positive T (sel230 / sele0114).

    Not used in ``delineate_record_t_template``. sel301 (early landmark) and q2c defer
    (sel114) keep per-beat / per-cycle behaviour instead.
    """
    if tmpl is None or not _is_early_peak_landmark(tmpl):
        return False
    if str(getattr(tmpl, "t_morphology", "normal") or "normal") != "normal":
        return False
    if _suppress_early_peak_beat_tuning(tmpl):
        return False
    return not _t_polarity_is_negative(tmpl)


def _record_t_search_end_tpl_idx(
    t_j: float,
    p_j: float,
    cfg: Optional[ProcessCycleConfig] = None,
    *,
    early_peak: bool = False,
) -> float:
    """Template-axis upper bound for T apex search (w1 / template delineation)."""
    mid_tp = t_j + (p_j - t_j) / 2.0
    mode = getattr(cfg, "record_t_w1_end_mode", "mid_tp") if cfg is not None else "mid_tp"
    if mode == "template_tj_margin":
        if early_peak and cfg is not None:
            post = float(getattr(cfg, "record_t_early_peak_w1_post_tj_frac", 0.05))
        else:
            post = float(getattr(cfg, "record_t_w1_post_tj_frac", 0.15))
        return t_j + post * (mid_tp - t_j)
    return mid_tp


def _apply_w1_hi_sq_frac_floor(
    end_tpl: float,
    t_j: float,
    p_j: float,
    n_tpl: int,
    cfg: Optional[ProcessCycleConfig] = None,
) -> float:
    """
    Extend formula w1_hi when its S→Q fraction is below ``record_t_w1_hi_min_sq_frac``.

    Capped at ``p_j − record_t_w1_hi_pj_margin_sq_frac`` so the window stays clear of P.
    """
    if cfg is None or n_tpl < 2:
        return end_tpl
    min_frac = float(getattr(cfg, "record_t_w1_hi_min_sq_frac", 0.0) or 0.0)
    if min_frac <= 0.0:
        return end_tpl
    pj_margin = float(getattr(cfg, "record_t_w1_hi_pj_margin_sq_frac", 0.15))
    denom = float(n_tpl - 1)
    end_frac = float(end_tpl) / denom
    p_j_frac = float(p_j) / denom
    t_j_frac = float(t_j) / denom
    cap_frac = max(t_j_frac, p_j_frac - pj_margin)
    if end_frac < min_frac:
        end_frac = min(min_frac, cap_frac)
        return end_frac * denom
    return end_tpl


def _record_t_window_samples(
    s_i: int,
    q_next: int,
    t_j: float,
    p_j: float,
    n_tpl: int,
    cfg: Optional[ProcessCycleConfig] = None,
    *,
    tmpl=None,
    force_wide_w1: bool = False,
) -> Tuple[int, int]:
    """w1: S + Tj → mid(Tj,Pj) or template Tj + margin (Mode 1)."""
    early = (
        not force_wide_w1
        and tmpl is not None
        and _early_peak_record_t_tuning_applies(tmpl)
    )
    narrow_w1 = early or (tmpl is not None and _t_search_prefer_negative(tmpl))
    end_tpl = _record_t_search_end_tpl_idx(t_j, p_j, cfg, early_peak=early)
    if not narrow_w1:
        end_tpl = _apply_w1_hi_sq_frac_floor(end_tpl, t_j, p_j, n_tpl, cfg)
    t_lo = _tpl_index_to_sample(s_i, q_next, t_j, n_tpl)
    t_hi = _tpl_index_to_sample(s_i, q_next, end_tpl, n_tpl)
    if early:
        min_span = max(3, int(round(0.06 * (q_next - s_i))))
        if t_hi - t_lo < 3:
            end_tpl = _record_t_search_end_tpl_idx(t_j, p_j, cfg, early_peak=False)
            end_tpl = _apply_w1_hi_sq_frac_floor(end_tpl, t_j, p_j, n_tpl, cfg)
            t_hi = _tpl_index_to_sample(s_i, q_next, end_tpl, n_tpl)
            t_hi = min(q_next - 3, max(t_lo + min_span, t_hi))
        elif t_hi - t_lo < min_span:
            t_hi = min(q_next - 3, t_lo + min_span)
    if t_hi <= t_lo:
        t_hi = min(q_next - 3, t_lo + max(3, int(0.05 * (q_next - s_i))))
    return t_lo, t_hi


def apply_record_t_rt_cap(
    t_lo: int,
    t_hi: int,
    s_i: int,
    q_next: int,
    r_idx: Optional[int],
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> Tuple[int, int]:
    """Cap w1 at max RT; keep at least a beat-scaled minimum span."""
    if r_idx is not None and cfg.record_t_max_rt_ms > 0:
        rt_cap = int(r_idx) + _ms_before_r(sampling_rate, cfg.record_t_max_rt_ms)
        t_hi = min(t_hi, rt_cap)
    min_span = max(3, int(round(0.06 * (q_next - s_i))))
    if t_hi - t_lo < min_span:
        t_lo = max(s_i + 3, t_hi - min_span)
    return t_lo, t_hi


def _record_t_p_window_samples(
    s_i: int,
    q_next: int,
    t_j: float,
    p_j: float,
    n_tpl: int,
) -> Tuple[int, int]:
    """Legacy w2: Q − Pj → Q − (Pj−Tj)/3 in template coordinates."""
    l_end = float(n_tpl - 1)
    p_far_tpl = l_end - p_j
    p_near_tpl = l_end - (p_j - t_j) / 3.0
    p_lo = _tpl_index_to_sample(s_i, q_next, min(p_far_tpl, p_near_tpl), n_tpl)
    p_hi = _tpl_index_to_sample(s_i, q_next, max(p_far_tpl, p_near_tpl), n_tpl)
    p_lo = max(s_i + 3, p_lo)
    p_hi = min(q_next - 1, max(p_hi, p_lo + 3))
    return p_lo, p_hi


def record_t_p_window_samples(
    *,
    s_i: int,
    q_next: int,
    t_j: float,
    p_j: float,
    n_tpl: int,
    r_det: int,
    r_next: Optional[int],
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    signal_len: Optional[int] = None,
) -> Tuple[int, int]:
    """Beat-wise record-T P search window (R-anchored PR or S→Q w2)."""
    r_anchor = _p_r_anchor(r_det, r_next, cfg)
    if r_anchor is not None:
        mode = getattr(cfg, "record_t_p_r_anchor_mode", "current_r")
        clamp_sq = mode == "next_r"
        return p_pr_window_samples(
            r_anchor,
            sampling_rate,
            cfg,
            signal_len=signal_len,
            s_i=int(s_i) if clamp_sq else None,
            q_next=int(q_next) if clamp_sq else None,
        )
    return _record_t_p_window_samples(s_i, q_next, t_j, p_j, n_tpl)


def projected_p_pr_delay_ms(tmpl, cfg: ProcessCycleConfig) -> float:
    """PR interval (ms) before R anchor used to project P center."""
    pr_min = float(cfg.record_t_p_pr_min_ms)
    pr_max = float(cfg.record_t_p_pr_max_ms)
    if getattr(tmpl, "p_pr_center_ms", None) is not None:
        return float(tmpl.p_pr_center_ms)
    if _p_anchor_mode(cfg) == "current_r":
        return float(getattr(cfg, "record_t_p_pr_center_ms", 0.5 * (pr_min + pr_max)))
    if tmpl.p_landmark_idx is None or tmpl.template.size < 2:
        return 0.5 * (pr_min + pr_max)
    frac = float(tmpl.p_landmark_idx) / float(max(tmpl.template.size - 1, 1))
    return pr_min + (1.0 - frac) * (pr_max - pr_min)


def estimate_record_p_pr_center_ms(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> float:
    """
    Estimate record-level PR interval from mean morphology in the P search window.

    Finds the dominant apex in the ensemble PR segment (same idea as beat overlay plots).
    """
    pr_max = float(cfg.record_t_p_pr_max_ms)
    pr_min = float(cfg.record_t_p_pr_min_ms)
    pr_max_s = _ms_before_r(sampling_rate, pr_max)
    pr_min_s = _ms_before_r(sampling_rate, pr_min)
    target = max(3, pr_max_s - pr_min_s + 1)
    segs: List[np.ndarray] = []
    for r in np.asarray(r_peaks, dtype=int):
        lo = int(r) - pr_max_s
        hi = int(r) - pr_min_s
        if lo < 0 or hi >= len(ecg) or hi <= lo:
            continue
        seg = ecg[lo : hi + 1].astype(float, copy=False)
        if seg.size == target:
            segs.append(seg)
        elif seg.size >= 2:
            x_old = np.linspace(0.0, 1.0, seg.size)
            x_new = np.linspace(0.0, 1.0, target)
            segs.append(np.interp(x_new, x_old, seg))
    if not segs:
        return float(getattr(cfg, "record_t_p_pr_center_ms", 0.5 * (pr_min + pr_max)))
    mean_seg = np.mean(np.vstack(segs), axis=0)
    peak_i = int(np.argmax(mean_seg))
    span = max(target - 1, 1)
    pr_ms = pr_max - (float(peak_i) / float(span)) * (pr_max - pr_min)
    return float(np.clip(pr_ms, pr_min, pr_max))


def project_p_center_sample(
    r_anchor: int,
    tmpl,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    *,
    wavelet_pr_offset_samples: Optional[float] = None,
) -> int:
    """
  Projected P apex sample index before ``r_anchor``.

  Uses wavelet prior (R-relative samples), R-centered template offset, or record-T landmark PR.
    """
    if wavelet_pr_offset_samples is not None:
        return int(round(float(r_anchor) + float(wavelet_pr_offset_samples)))
    if (
        getattr(tmpl, "template_anchor", "s_to_q") != "s_to_q"
        and tmpl.p_offset_samples is not None
    ):
        return int(round(float(r_anchor) + float(tmpl.p_offset_samples)))
    pr_ms = projected_p_pr_delay_ms(tmpl, cfg)
    return int(r_anchor) - _ms_before_r(sampling_rate, pr_ms)


def _resolve_record_t_tpl_idx_for_projection(tmpl, cfg: Optional[ProcessCycleConfig]) -> Optional[float]:
    """
    Template index used to project T onto each beat.

    For ``early_peak`` landmarks, ``delineated`` offset is often a late repolarization lobe;
    anchor record-T at Tⱼ instead of the wide template-delineation apex.
    """
    if tmpl is None:
        return None
    if (
        cfg is not None
        and getattr(cfg, "record_biphasic_pm_lobe_search", False)
        and getattr(tmpl, "t_morphology", "") == "biphasic_positive_negative"
    ):
        land = getattr(tmpl, "t_landmark_idx", None)
        return float(land) if land is not None else None
    src = getattr(cfg, "record_t_project_from", "delineated") if cfg is not None else "delineated"
    land = getattr(tmpl, "t_landmark_idx", None)
    off = getattr(tmpl, "t_offset_samples", None)
    if _early_peak_record_t_tuning_applies(tmpl) and land is not None:
        if src == "landmark":
            return float(land)
        if src == "delineated":
            if off is None or float(off) <= float(land):
                return float(land)
            return float(land)
        if src == "blend":
            blend = float(getattr(cfg, "record_t_landmark_blend_frac", 0.35)) if cfg else 0.35
            return float(land) + blend * (float(off) - float(land))
    if src == "blend" and land is not None:
        blend = float(getattr(cfg, "record_t_landmark_blend_frac", 0.35)) if cfg else 0.35
        if off is not None and float(off) > float(land):
            return float(land) + blend * (float(off) - float(land))
        return float(land)
    if src == "landmark":
        return float(land) if land is not None else None
    tpl_idx = off
    if tpl_idx is None:
        tpl_idx = land
    return float(tpl_idx) if tpl_idx is not None else None


def project_t_center_sample(
    s_i: int,
    q_next: int,
    tmpl,
    n_tpl: int,
    cfg: Optional[ProcessCycleConfig] = None,
) -> Optional[int]:
    """Project template T apex onto beat S→Q coordinates."""
    if n_tpl < 2 or tmpl is None:
        return None
    tpl_idx = _resolve_record_t_tpl_idx_for_projection(tmpl, cfg)
    if tpl_idx is None:
        return None
    return _tpl_index_to_sample(int(s_i), int(q_next), float(tpl_idx), int(n_tpl))


def _local_extrema_indices(seg: np.ndarray, *, prefer: str) -> List[int]:
    if seg.size < 2:
        return [0] if seg.size else []
    if seg.size == 2:
        return [int(np.argmax(np.abs(seg)))]
    out: List[int] = []
    for i in range(1, seg.size - 1):
        if prefer == "max":
            if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]:
                out.append(i)
        else:
            if seg[i] <= seg[i - 1] and seg[i] <= seg[i + 1]:
                out.append(i)
    if not out:
        out = [int(np.argmin(seg))] if prefer == "min" else [int(np.argmax(seg))]
    return out


def _search_p_template_guided(
    ecg: np.ndarray,
    p_lo: int,
    p_hi: int,
    p_center: int,
    tmpl,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> Tuple[Optional[int], str]:
    """
    Pick the largest local apex nearest the template-projected P center.

    Score = |amplitude| − distance_penalty × |index − center| (prefers high, near template).
    """
    p_hi = min(len(ecg), int(p_hi) + 1)
    if p_hi - p_lo < 3:
        return None, "positive"

    half = _ms_before_r(sampling_rate, cfg.record_t_p_template_guided_half_window_ms)
    loc_lo = max(int(p_lo), int(p_center) - half)
    loc_hi = min(int(p_hi), int(p_center) + half)
    if loc_hi - loc_lo < 3:
        loc_lo, loc_hi = int(p_lo), int(p_hi)

    if cfg.record_t_use_savgol:
        seg, lo, _ = smooth_search_window(ecg, loc_lo, loc_hi, sampling_rate, cfg)
    else:
        lo = int(loc_lo)
        seg = ecg[lo : min(len(ecg), int(loc_hi) + 1)].astype(float, copy=False)

    # Always prefer the largest positive apex near the projected center (q1c P is typically upright).
    prefer = "max"
    peaks = _local_extrema_indices(seg, prefer=prefer)
    if not peaks:
        peaks = [int(np.argmax(seg))]
    rel_center = float(p_center) - float(lo)
    penalty = float(cfg.record_t_p_template_guided_distance_penalty)

    def _amp(i: int) -> float:
        return float(seg[i])

    best_i = max(peaks, key=lambda i: _amp(i) - penalty * abs(float(i) - rel_center))
    pol = "positive" if seg[best_i] >= 0 else "negative"
    return lo + int(best_i), pol


def _t_search_prefer_negative(tmpl) -> bool:
    if _suppress_early_peak_beat_tuning(tmpl):
        return False
    if getattr(tmpl, "t_morphology", "") == "inverted_t":
        return True
    return str(getattr(tmpl, "t_polarity", "positive") or "positive") == "negative"


def _t_polarity_is_negative(tmpl) -> bool:
    return str(getattr(tmpl, "t_polarity", "positive") or "positive") == "negative"


def defer_record_t_overwrite(
    tmpl,
    *,
    manual_ann_ext: Optional[str] = None,
) -> bool:
    """
    Keep per-cycle T on q2c negative ``early_peak`` (sel114).

    q1c negative early_peak (sel301) uses record-T + smoothing instead.
    """
    if str(manual_ann_ext or "") != "q2c":
        return False
    if tmpl is None or not _is_early_peak_landmark(tmpl):
        return False
    if str(getattr(tmpl, "t_morphology", "normal") or "normal") != "normal":
        return False
    return _t_polarity_is_negative(tmpl)


def record_t_use_biphasic_fallback(tmpl) -> bool:
    """Negative/inverted T: do not fall back to a late positive lobe in w1."""
    return not _t_search_prefer_negative(tmpl)


def _search_t_rising_edge_onset(
    seg: np.ndarray,
    lo: int,
    *,
    prefer_negative: bool,
    edge_frac: float = 0.5,
) -> Optional[Tuple[int, str]]:
    """First sample reaching 50% of window |excursion| (template rising_edge analogue)."""
    if seg.size < 2:
        return None
    baseline = float(np.median(seg))
    if prefer_negative:
        exc = baseline - seg
        pol_at = lambda i: "negative"
    else:
        exc = seg - baseline
        pol_at = lambda i: "positive" if float(seg[i]) >= baseline else "negative"
    peak = float(np.max(exc))
    if peak <= 0.0:
        return None
    frac = float(np.clip(edge_frac, 0.2, 0.8))
    thr = frac * peak
    for i, v in enumerate(exc):
        if float(v) >= thr:
            return lo + int(i), pol_at(i)
    return None


def _search_t_biphasic_positive_lobe(
    ecg: np.ndarray,
    s_i: int,
    q_next: int,
    tmpl,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> Tuple[Optional[int], str]:
    """
    Biphasic +−: search only between projected positive and negative lobes.

    Picks the dominant positive apex in
    [t_pos − early_margin, min(t_neg − sep_margin, t_pos + late_cap)].
    """
    pos_tpl = getattr(tmpl, "t_biphasic_pos_landmark_idx", None)
    neg_tpl = getattr(tmpl, "t_biphasic_neg_landmark_idx", None)
    n_tpl = int(tmpl.template.size) if tmpl is not None else 0
    if pos_tpl is None or neg_tpl is None or n_tpl < 2:
        return None, "positive"

    t_pos = _tpl_index_to_sample(int(s_i), int(q_next), float(pos_tpl), n_tpl)
    t_neg = _tpl_index_to_sample(int(s_i), int(q_next), float(neg_tpl), n_tpl)
    if t_neg <= t_pos:
        return None, "positive"

    early = _ms_before_r(sampling_rate, cfg.record_t_biphasic_pm_early_margin_ms)
    sep = _ms_before_r(sampling_rate, cfg.record_t_biphasic_pm_sep_before_neg_ms)
    late_cap = _ms_before_r(sampling_rate, cfg.record_t_biphasic_pm_late_cap_ms)

    lo = max(int(s_i), int(t_pos) - early)
    hi = min(int(q_next), int(t_neg) - sep, int(t_pos) + late_cap)
    hi = min(len(ecg) - 1, hi)
    if hi - lo < 3:
        hi = min(len(ecg) - 1, int(t_pos) + late_cap)
        lo = max(int(s_i), int(t_pos) - early)
    if hi - lo < 3:
        return None, "positive"

    if cfg.record_t_use_savgol:
        seg, seg_lo, _ = smooth_search_window(ecg, lo, hi, sampling_rate, cfg)
    else:
        seg_lo = int(lo)
        seg = ecg[seg_lo : min(len(ecg), int(hi) + 1)].astype(float, copy=False)

    st_lo = max(int(s_i), seg_lo - _ms_before_r(sampling_rate, 60.0))
    baseline = float(np.median(ecg[st_lo:seg_lo])) if seg_lo > st_lo else float(seg[0])
    rel = seg - baseline

    peaks = _local_extrema_indices(rel, prefer="max")
    pos_peaks = [i for i in peaks if rel[i] > 0]
    if not pos_peaks:
        pos_peaks = [int(np.argmax(rel))] if float(np.max(rel)) > 0 else []

    if not pos_peaks:
        return None, "positive"

    best_i = max(pos_peaks, key=lambda i: float(rel[i]))
    return int(seg_lo + best_i), "positive"


def _search_t_early_peak_apex(
    ecg: np.ndarray,
    t_lo: int,
    t_hi: int,
    t_center: int,
    tmpl,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> Tuple[Optional[int], str]:
    """
    early_peak Tⱼ: rising-edge onset, else earliest qualified extremum near landmark.

    Avoids late-lobe max-amplitude picks on biphasic / broad T waves (q1c panel records).
    """
    t_hi = min(len(ecg), int(t_hi) + 1)
    if t_hi - t_lo < 3 or t_center is None:
        return None, "positive"

    half = _ms_before_r(sampling_rate, cfg.record_t_template_guided_half_window_ms)
    loc_lo = max(int(t_lo), int(t_center) - half)
    loc_hi = min(int(t_hi), int(t_center) + half)
    if loc_hi - loc_lo < 3:
        loc_lo, loc_hi = int(t_lo), int(t_hi)

    if cfg.record_t_use_savgol:
        seg, lo, _ = smooth_search_window(ecg, loc_lo, loc_hi, sampling_rate, cfg)
    else:
        lo = int(loc_lo)
        seg = ecg[lo : min(len(ecg), int(loc_hi) + 1)].astype(float, copy=False)

    prefer_negative = _t_search_prefer_negative(tmpl)
    late_cap = int(t_center) + _ms_before_r(
        sampling_rate,
        float(getattr(cfg, "record_t_early_peak_max_late_from_center_ms", 15.0)),
    )

    def _within_cap(idx: int) -> bool:
        return int(idx) <= late_cap

    # Rising-edge onset helps upright early_peak (e.g. sele0114 / sel230).
    if not prefer_negative:
        edge_frac = float(
            getattr(cfg, "record_t_early_peak_rising_edge_frac", 0.35)
        )
        rising = _search_t_rising_edge_onset(
            seg, lo, prefer_negative=False, edge_frac=edge_frac
        )
        if rising is not None:
            idx, pol = rising
            max_dist = _ms_before_r(sampling_rate, cfg.record_t_mode1_max_dist_ms)
            if abs(int(idx) - int(t_center)) <= max_dist and _within_cap(idx):
                return idx, pol

    prefer = "min" if prefer_negative else "max"
    peaks = _local_extrema_indices(seg, prefer=prefer)
    if not peaks:
        peaks = [int(np.argmin(seg))] if prefer == "min" else [int(np.argmax(seg))]

    rel_center = float(t_center) - float(lo)
    max_dist = _ms_before_r(sampling_rate, cfg.record_t_mode1_max_dist_ms)
    min_amp_frac = float(cfg.record_t_mode1_min_amp_frac)
    amp_ref = float(np.max(np.abs(seg))) if seg.size else 0.0
    min_amp = min_amp_frac * amp_ref if amp_ref > 0.0 else 0.0

    qualified: List[int] = []
    for i in peaks:
        abs_idx = lo + int(i)
        if abs(float(i) - rel_center) > max_dist:
            continue
        if abs(float(seg[i])) < min_amp:
            continue
        if not _within_cap(abs_idx):
            continue
        qualified.append(int(i))

    pool = qualified if qualified else [i for i in peaks if _within_cap(lo + int(i))]
    if not pool:
        pool = list(peaks)

    def _rank(i: int) -> Tuple[int, float, float]:
        return (int(i), abs(float(i) - rel_center), -abs(float(seg[i])))

    best_i = min(pool, key=_rank)
    pol = "positive" if seg[best_i] >= 0 else "negative"
    out_idx = lo + int(best_i)
    if not _within_cap(out_idx):
        out_idx = min(int(t_hi), late_cap)
    return out_idx, pol


def _search_t_template_guided(
    ecg: np.ndarray,
    t_lo: int,
    t_hi: int,
    t_center: int,
    tmpl,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> Tuple[Optional[int], str]:
    """Template-guided T apex: Mode 1 = nearest extremum to projected center; else amplitude score."""
    if _early_peak_record_t_tuning_applies(tmpl):
        return _search_t_early_peak_apex(
            ecg, t_lo, t_hi, t_center, tmpl, sampling_rate, cfg
        )

    t_hi = min(len(ecg), int(t_hi) + 1)
    if t_hi - t_lo < 3:
        return None, "positive"

    half = _ms_before_r(sampling_rate, cfg.record_t_template_guided_half_window_ms)
    loc_lo = max(int(t_lo), int(t_center) - half)
    loc_hi = min(int(t_hi), int(t_center) + half)
    if loc_hi - loc_lo < 3:
        loc_lo, loc_hi = int(t_lo), int(t_hi)

    if cfg.record_t_use_savgol:
        seg, lo, _ = smooth_search_window(ecg, loc_lo, loc_hi, sampling_rate, cfg)
    else:
        lo = int(loc_lo)
        seg = ecg[lo : min(len(ecg), int(loc_hi) + 1)].astype(float, copy=False)

    prefer = "min" if getattr(tmpl, "t_morphology", "") == "inverted_t" else "max"
    peaks = _local_extrema_indices(seg, prefer=prefer)
    if not peaks:
        peaks = [int(np.argmax(np.abs(seg)))]
    rel_center = float(t_center) - float(lo)
    penalty = float(cfg.record_t_template_guided_distance_penalty)

    def _score(i: int) -> float:
        return abs(float(seg[i])) - penalty * abs(float(i) - rel_center)

    best_i = max(peaks, key=_score)
    pol = "positive" if seg[best_i] >= 0 else "negative"
    return lo + int(best_i), pol


def _search_p_apex_in_window(
    ecg: np.ndarray,
    p_lo: int,
    p_hi: int,
    tmpl,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    *,
    r_anchor: Optional[int] = None,
    wavelet_pr_offset_samples: Optional[float] = None,
) -> Tuple[Optional[int], str]:
    p_hi = min(len(ecg), int(p_hi) + 1)
    if p_hi - p_lo < 3:
        return None, "positive"

    p_center: Optional[int] = None
    if r_anchor is not None:
        p_center = project_p_center_sample(
            int(r_anchor),
            tmpl,
            sampling_rate,
            cfg,
            wavelet_pr_offset_samples=wavelet_pr_offset_samples,
        )

    if cfg.record_t_use_savgol:
        seg, lo, hi = smooth_search_window(ecg, p_lo, p_hi, sampling_rate, cfg)
    else:
        lo, hi = int(p_lo), min(len(ecg), int(p_hi))
        seg = ecg[lo:hi].astype(float, copy=False)
    th_up = tmpl.th_p_up if tmpl.th_p_up is not None else 0.05
    th_down = tmpl.th_p_down if tmpl.th_p_down is not None else -0.05
    p_rel, p_pol = _apex_with_threshold(
        seg,
        prefer="max",
        threshold_up=th_up,
        threshold_down=th_down,
        check_biphasic=False,
    )
    idx = (lo + int(p_rel)) if p_rel is not None else None

    if cfg.record_t_p_template_guided and p_center is not None:
        reconcile = _ms_before_r(sampling_rate, cfg.record_t_p_template_guided_reconcile_ms)
        off_center = abs(int(idx) - p_center) if idx is not None else reconcile + 1
        if idx is None or off_center > reconcile:
            guided = _search_p_template_guided(
                ecg, p_lo, p_hi, p_center, tmpl, sampling_rate, cfg
            )
            if guided[0] is not None:
                return guided
    return idx, p_pol


def _beat_derivative_zero_crossing(
    seg: np.ndarray,
    lo: int,
    *,
    before_abs: Optional[int] = None,
) -> Optional[int]:
    """Latest upslope d1 zero-crossing in beat search segment (absolute sample)."""
    if seg.size < 4:
        return None
    d1 = np.gradient(seg.astype(float))
    hits: List[int] = []
    for i in range(1, d1.size):
        if d1[i - 1] <= 0.0 < d1[i]:
            abs_i = lo + i
            if before_abs is None or abs_i < before_abs:
                hits.append(abs_i)
    return hits[-1] if hits else None


def _search_t_post_apex_dz_vs_positive_peak(
    ecg: np.ndarray,
    t_lo: int,
    t_hi: int,
    t_center: int,
    tmpl,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> Tuple[Optional[int], str]:
    """
    sel16420-style branch: compare template-guided ``positive_peak`` vs
    ``post_apex_dz`` (downslope_dz_after_positive_peak).

    Prefer post_apex_dz when it is qualified (+10..+80 ms after positive_peak,
    before terminal negative / late cap), within RT tolerance of the projected
    template landmark, and positive_peak is systematically early vs that landmark.
    """
    from pyhearts.processing.record_post_apex_dz_morphology import (
        downslope_dz_after_positive_peak,
        qualified_post_apex_dz_pair,
    )

    pos_idx, pol = _search_t_template_guided(
        ecg, t_lo, t_hi, t_center, tmpl, sampling_rate, cfg
    )
    if pos_idx is None:
        return None, pol

    t_hi_clip = min(len(ecg), int(t_hi) + 1)
    if t_hi_clip - t_lo < 4:
        return int(pos_idx), pol

    if cfg.record_t_use_savgol:
        seg, lo, _ = smooth_search_window(ecg, t_lo, t_hi, sampling_rate, cfg)
    else:
        lo = int(t_lo)
        seg = ecg[lo:t_hi_clip].astype(float, copy=False)

    dz_idx = downslope_dz_after_positive_peak(seg, lo, after_abs=int(pos_idx))
    if dz_idx is None:
        return int(pos_idx), pol

    st_ref = max(0, min(seg.size - 1, int(pos_idx) - lo - max(1, _ms_before_r(sampling_rate, 30.0))))
    baseline = float(np.median(seg[: st_ref + 1])) if st_ref >= 0 else float(np.median(seg))

    if not qualified_post_apex_dz_pair(
        seg, lo, int(pos_idx), int(dz_idx), baseline, sampling_rate, cfg
    ):
        return int(pos_idx), pol

    # Amplitude apex (positive_peak) is systematically early on this morphology;
    # manual / ECGPUWAVE align with the later post_apex_dz on the same positive lobe.
    late_cap = _ms_before_r(
        sampling_rate,
        float(getattr(cfg, "record_t_post_apex_dz_rt_tolerance_ms", 45.0)),
    )
    if abs(int(dz_idx) - int(t_center)) > late_cap:
        return int(pos_idx), pol
    return int(dz_idx), "positive"


def record_detect_t_peak(
    ecg: np.ndarray,
    s_i: int,
    q_next: int,
    tmpl,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    *,
    r_idx: Optional[int] = None,
    force_wide_w1: bool = False,
) -> Tuple[Optional[int], str]:
    """T apex on delineation signal within record-T window w1."""
    if tmpl.t_landmark_idx is None or tmpl.p_landmark_idx is None:
        return None, "negative"
    n_tpl = tmpl.template.size
    if n_tpl < 6 or q_next <= s_i + 5:
        return None, "negative"

    t_j = float(tmpl.t_landmark_idx)
    p_j = float(tmpl.p_landmark_idx)
    t_lo, t_hi = _record_t_window_samples(
        s_i, q_next, t_j, p_j, n_tpl, cfg, tmpl=tmpl, force_wide_w1=force_wide_w1
    )
    t_lo, t_hi = apply_record_t_rt_cap(
        t_lo, t_hi, int(s_i), int(q_next), r_idx, sampling_rate, cfg
    )

    plateau_fwd = _inverted_plateau_apex_forward(
        ecg,
        int(s_i),
        int(q_next),
        tmpl,
        int(t_lo),
        min(int(t_hi), len(ecg) - 1),
        sampling_rate,
        cfg,
    )
    if plateau_fwd is not None:
        return plateau_fwd

    if (
        getattr(cfg, "record_biphasic_pm_lobe_search", False)
        and getattr(tmpl, "t_morphology", "") == "biphasic_positive_negative"
    ):
        bi = _search_t_biphasic_positive_lobe(
            ecg, int(s_i), int(q_next), tmpl, sampling_rate, cfg
        )
        if bi[0] is not None:
            return bi

    t_hi = min(len(ecg), t_hi + 1)
    if t_hi - t_lo < 3:
        return None, "negative"

    t_center = project_t_center_sample(int(s_i), int(q_next), tmpl, n_tpl, cfg)

    if (
        getattr(cfg, "record_t_post_apex_dz_preference", False)
        and getattr(tmpl, "t_post_apex_dz_preference", False)
        and t_center is not None
    ):
        cmp_t = _search_t_post_apex_dz_vs_positive_peak(
            ecg, t_lo, t_hi, int(t_center), tmpl, sampling_rate, cfg
        )
        if cmp_t[0] is not None:
            return cmp_t

    apex_mode = getattr(cfg, "record_t_apex_mode", "threshold")
    if _early_peak_record_t_tuning_applies(tmpl) and t_center is not None:
        early = _search_t_early_peak_apex(
            ecg, t_lo, t_hi, t_center, tmpl, sampling_rate, cfg
        )
        if early[0] is not None:
            return early

    if (
        apex_mode == "mode1"
        and cfg.record_t_template_guided
        and t_center is not None
    ):
        guided = _search_t_template_guided(
            ecg, t_lo, t_hi, t_center, tmpl, sampling_rate, cfg
        )
        if guided[0] is not None:
            return guided

    if cfg.record_t_use_savgol:
        seg, lo, hi = smooth_search_window(ecg, t_lo, t_hi, sampling_rate, cfg)
    else:
        lo, hi = int(t_lo), min(len(ecg), int(t_hi))
        seg = ecg[lo:hi].astype(float, copy=False)
    th_up = tmpl.th_t_up if tmpl.th_t_up is not None else 0.05
    th_down = tmpl.th_t_down if tmpl.th_t_down is not None else -0.05
    prefer = "min" if _t_search_prefer_negative(tmpl) else "max"
    t_rel, t_pol = _apex_with_threshold(
        seg,
        prefer=prefer,
        threshold_up=th_up,
        threshold_down=th_down,
        check_biphasic=record_t_use_biphasic_fallback(tmpl),
    )
    idx = lo + int(t_rel) if t_rel is not None else None

    if cfg.record_t_template_guided and t_center is not None and apex_mode != "mode1":
        reconcile = _ms_before_r(sampling_rate, cfg.record_t_template_guided_reconcile_ms)
        use_guided = idx is None
        if idx is not None and abs(int(idx) - int(t_center)) > reconcile:
            use_guided = True
        if use_guided:
            guided = (
                _search_t_early_peak_apex(
                    ecg, t_lo, t_hi, t_center, tmpl, sampling_rate, cfg
                )
                if _early_peak_record_t_tuning_applies(tmpl)
                else _search_t_template_guided(
                    ecg, t_lo, t_hi, t_center, tmpl, sampling_rate, cfg
                )
            )
            if guided[0] is not None:
                return guided
    if idx is None:
        return None, t_pol
    return int(idx), t_pol


def record_detect_p_peak(
    ecg: np.ndarray,
    q_next: int,
    s_i: int,
    tmpl,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    *,
    r_det: Optional[int] = None,
    r_next: Optional[int] = None,
    wavelet_pr_offset_samples: Optional[float] = None,
) -> Tuple[Optional[int], str]:
    """P apex on delineation signal within PR-anchored or record-T w2 window."""
    if tmpl.p_landmark_idx is None or tmpl.t_landmark_idx is None:
        return None, "positive"
    n_tpl = tmpl.template.size
    if n_tpl < 6:
        return None, "positive"

    r_anchor = _p_r_anchor(int(r_det) if r_det is not None else 0, r_next, cfg)
    if r_anchor is None and (q_next <= s_i + 5):
        return None, "positive"

    t_j = float(tmpl.t_landmark_idx)
    p_j = float(tmpl.p_landmark_idx)
    p_lo, p_hi = record_t_p_window_samples(
        s_i=int(s_i),
        q_next=int(q_next),
        t_j=t_j,
        p_j=p_j,
        n_tpl=n_tpl,
        r_det=int(r_det) if r_det is not None else int(r_anchor or 0),
        r_next=r_next,
        sampling_rate=sampling_rate,
        cfg=cfg,
        signal_len=len(ecg),
    )
    r_anchor = _p_r_anchor(int(r_det) if r_det is not None else 0, r_next, cfg)
    return _search_p_apex_in_window(
        ecg,
        p_lo,
        p_hi,
        tmpl,
        sampling_rate,
        cfg,
        r_anchor=r_anchor,
        wavelet_pr_offset_samples=wavelet_pr_offset_samples,
    )


def record_fallback_p_search(
    ecg: np.ndarray,
    r_det: int,
    r_next: Optional[int],
    tmpl,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    *,
    wavelet_pr_offset_samples: Optional[float] = None,
) -> Optional[float]:
    """Template-guided P search in the PR window when record-T misses."""
    if not cfg.record_t_p_r_anchor:
        return None
    r_anchor = _p_r_anchor(int(r_det), r_next, cfg)
    if r_anchor is None:
        return None
    p_lo, p_hi = p_pr_window_samples(
        r_anchor,
        sampling_rate,
        cfg,
        signal_len=len(ecg),
    )
    p_center = project_p_center_sample(
        r_anchor,
        tmpl,
        sampling_rate,
        cfg,
        wavelet_pr_offset_samples=wavelet_pr_offset_samples,
    )
    idx, _ = _search_p_template_guided(
        ecg, p_lo, p_hi, p_center, tmpl, sampling_rate, cfg
    )
    if idx is None and cfg.record_t_p_template_guided:
        idx, _ = record_detect_p_peak(
            ecg,
            q_next=len(ecg),
            s_i=0,
            tmpl=tmpl,
            sampling_rate=sampling_rate,
            cfg=cfg,
            r_det=int(r_det),
            r_next=r_next,
            wavelet_pr_offset_samples=wavelet_pr_offset_samples,
        )
    return float(idx) if idx is not None else None


def _p_anchor_mode(cfg: ProcessCycleConfig) -> str:
    return getattr(cfg, "record_t_p_r_anchor_mode", "current_r")


def _validate_p_guess(
    p_guess: Optional[float],
    t_guess: Optional[float],
    r_det: int,
    r_next: Optional[int],
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> Optional[float]:
    if p_guess is None:
        return None
    fs = float(sampling_rate)
    min_before_r = _ms_before_r(fs, 40.0)
    mode = _p_anchor_mode(cfg)

    if cfg.record_t_p_r_anchor and mode == "current_r":
        pr_lo = float(r_det) - _ms_before_r(fs, cfg.record_t_p_pr_max_ms)
        pr_hi = float(r_det) - _ms_before_r(fs, cfg.record_t_p_pr_min_ms)
        if p_guess < pr_lo or p_guess > pr_hi or p_guess >= float(r_det) - min_before_r:
            return None
        return p_guess

    if p_guess <= float(r_det) + min_before_r:
        return None
    if r_next is not None and p_guess >= float(r_next) - _ms_before_r(fs, 20.0):
        return None
    if t_guess is not None and p_guess <= t_guess:
        return None
    return p_guess


def record_t_pt_guesses(
    ecg_delim: np.ndarray,
    r_det: int,
    r_next: Optional[int],
    tmpl,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    scale: float,
    *,
    wavelet_pr_offset_samples: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Record-level P/T global indices for one beat (scale ignored; windows are beat-specific).

    Returns (t_guess, p_guess) like ``_record_t_pt_guesses``.
    """
    del scale
    if r_next is None:
        return None, None

    s_i = find_s_wave_after_r(
        ecg_delim, r_det, sampling_rate, search_window_ms=s_search_after_r_ms(cfg), inverted=False
    )
    q_next = find_q_wave_before_r(
        ecg_delim, int(r_next), sampling_rate, search_window_ms=q_search_before_r_ms(cfg), inverted=False
    )

    t_guess: Optional[float] = None
    if s_i is not None and q_next is not None and q_next > s_i + 5:
        t_idx, _ = record_detect_t_peak(
            ecg_delim, int(s_i), int(q_next), tmpl, sampling_rate, cfg, r_idx=int(r_det)
        )
        if t_idx is None:
            t_idx, _ = record_detect_t_peak(
                ecg_delim,
                int(s_i),
                int(q_next),
                tmpl,
                sampling_rate,
                cfg,
                r_idx=int(r_det),
                force_wide_w1=True,
            )
        t_guess = float(t_idx) if t_idx is not None else None

    p_guess: Optional[float] = None
    if s_i is not None and q_next is not None and q_next > s_i + 5:
        p_idx, _ = record_detect_p_peak(
            ecg_delim,
            int(q_next),
            int(s_i),
            tmpl,
            sampling_rate,
            cfg,
            r_det=int(r_det),
            r_next=r_next,
            wavelet_pr_offset_samples=wavelet_pr_offset_samples,
        )
        p_guess = float(p_idx) if p_idx is not None else None
    elif cfg.record_t_p_r_anchor and _p_anchor_mode(cfg) == "current_r":
        p_idx, _ = record_detect_p_peak(
            ecg_delim,
            len(ecg_delim),
            0,
            tmpl,
            sampling_rate,
            cfg,
            r_det=int(r_det),
            r_next=r_next,
            wavelet_pr_offset_samples=wavelet_pr_offset_samples,
        )
        p_guess = float(p_idx) if p_idx is not None else None

    if p_guess is None and cfg.record_t_p_template_guided and cfg.record_t_p_r_anchor:
        p_fb = record_fallback_p_search(
            ecg_delim,
            int(r_det),
            r_next,
            tmpl,
            sampling_rate,
            cfg,
            wavelet_pr_offset_samples=wavelet_pr_offset_samples,
        )
        p_guess = p_fb

    min_after_r = _ms_before_r(sampling_rate, 40.0)
    max_rt = _ms_before_r(sampling_rate, cfg.record_t_max_rt_ms)
    if t_guess is not None:
        if t_guess <= float(r_det) + min_after_r or t_guess >= float(r_next) - min_after_r:
            t_guess = None
        elif cfg.record_t_max_rt_ms > 0 and t_guess > float(r_det) + max_rt:
            t_guess = None

    p_guess = _validate_p_guess(p_guess, t_guess, int(r_det), r_next, sampling_rate, cfg)

    return t_guess, p_guess
