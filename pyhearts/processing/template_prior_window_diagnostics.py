"""
Diagnostics for template-prior T search windows.

Answers per bad-window beat:
  - Is the landmark wrong?
  - Is the window width wrong?
  - Is the template aligned incorrectly?
  - Is the beat unlike the template?

Also estimates per-record T timing σ for uncertainty-based windows (prediction ± 2σ).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.delineation_signal import prepare_record_delineation_signal
from pyhearts.processing.record_beat_clustering import extract_stpq_segments
from pyhearts.processing.record_delineation import (
    MedianBeatTemplate,
    _resample_segment,
    _stpq_s_q_anchor_indices,
)
from pyhearts.processing.record_stpq_detection import _apply_stpq_t_rt_cap
from pyhearts.processing.t_morphology_routing import morphology_rescue_landmark_global


LANDMARK_OK_MS = 40.0
ALIGN_OK_CORR = 0.82
RR_STRETCH_OK = (0.85, 1.15)
ANCHOR_SKEW_MS = 35.0


@dataclass(frozen=True)
class BadWindowBeatDiagnostics:
    """Per-beat measurements for a manual-T beat whose window excluded gold."""

    manual_t: int
    landmark: Optional[int]
    landmark_source: str
    t_lo: int
    t_hi: int
    manual_rt_ms: float
    landmark_rt_ms: float
    landmark_err_ms: float
    window_width_ms: float
    margin_lo_ms: float
    margin_hi_ms: float
    beat_template_corr: float
    rr_stretch_ratio: float
    cluster_id: Optional[int] = None
    record_landmark_err_ms: Optional[float] = None
    cluster_landmark_err_ms: Optional[float] = None
    bad_window_cause: str = ""
    diagnosis_bucket: str = ""


def _ms(delta_samples: float, fs: float) -> float:
    return float(delta_samples) / float(fs) * 1000.0


def _beat_template_correlation(
    ecg: np.ndarray,
    s_i: int,
    q_next: int,
    tmpl: MedianBeatTemplate,
) -> float:
    if not tmpl.valid or tmpl.template.size < 4:
        return float("nan")
    seg = ecg[int(s_i) : int(q_next)].astype(float, copy=False)
    if seg.size < 4:
        return float("nan")
    beat = _resample_segment(seg, int(tmpl.template.size))
    a = beat - float(np.mean(beat))
    b = tmpl.template.astype(float) - float(np.mean(tmpl.template))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def _median_stpq_span(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> float:
    spans: List[float] = []
    ecg_work = prepare_record_delineation_signal(ecg, sampling_rate, cfg)
    r_peaks = np.asarray(r_peaks, dtype=int)
    for i in range(len(r_peaks) - 1):
        s_i, q_next = _stpq_s_q_anchor_indices(
            ecg_work,
            int(r_peaks[i]),
            int(r_peaks[i + 1]),
            sampling_rate,
            cfg,
        )
        if s_i is not None and q_next is not None and q_next > s_i:
            spans.append(float(q_next - s_i))
    return float(np.median(spans)) if spans else float("nan")


def estimate_record_t_timing_sigma_ms(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    tmpl: MedianBeatTemplate,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    *,
    default_sigma_ms: float = 28.0,
) -> float:
    """
    σ of per-beat T apex RT (ms) in the template T search band.

    Used for uncertainty windows: landmark ± ``sigma_mult`` × σ.
    """
    if not tmpl.valid or tmpl.t_landmark_idx is None:
        return default_sigma_ms

    fs = float(sampling_rate)
    n_tpl = max(2, int(tmpl.template.size))
    t_frac = float(tmpl.t_landmark_idx) / float(n_tpl - 1)
    r_peaks = np.asarray(r_peaks, dtype=int)
    ecg_work = prepare_record_delineation_signal(ecg, sampling_rate, cfg)

    rt_ms: List[float] = []
    for i in range(len(r_peaks) - 1):
        r_i = int(r_peaks[i])
        s_i, q_next = _stpq_s_q_anchor_indices(
            ecg_work, r_i, int(r_peaks[i + 1]), sampling_rate, cfg
        )
        if s_i is None or q_next is None or q_next <= s_i + 8:
            continue
        span = int(q_next - s_i)
        t_center = int(s_i) + int(round(t_frac * span))
        half = max(3, int(round(0.12 * span)))
        lo = max(int(s_i) + 1, t_center - half)
        hi = min(int(q_next) - 1, t_center + half)
        if hi <= lo + 2:
            continue
        seg = ecg_work[lo:hi]
        rel = seg - float(np.median(seg))
        apex = int(lo + int(np.argmax(np.abs(rel))))
        rt_ms.append(_ms(apex - r_i, fs))

    if len(rt_ms) < 3:
        return default_sigma_ms
    sigma = float(np.std(rt_ms))
    return max(8.0, min(sigma, 120.0))


def t_uncertainty_window_samples(
    landmark: int,
    sigma_ms: float,
    ecg_len: int,
    s_i: int,
    q_next: int,
    r_idx: int,
    fs: float,
    cfg: ProcessCycleConfig,
    *,
    sigma_mult: float = 2.0,
    min_half_width_ms: float = 40.0,
) -> Tuple[int, int]:
    """Landmark-centered window: prediction ± max(min_half, sigma_mult × σ)."""
    half_ms = max(float(min_half_width_ms), float(sigma_mult) * float(sigma_ms))
    half_samp = max(1, int(round(half_ms * fs / 1000.0)))
    t_lo = max(int(s_i) + 3, int(landmark) - half_samp)
    t_hi = min(int(q_next) - 3, int(landmark) + half_samp)
    t_lo, t_hi = _apply_stpq_t_rt_cap(t_lo, t_hi, int(s_i), int(q_next), int(r_idx), fs, cfg)
    t_hi = min(ecg_len - 1, int(t_hi))
    if int(landmark) < t_lo:
        t_lo = max(int(s_i) + 3, int(landmark) - half_samp)
    if int(landmark) > t_hi:
        t_hi = min(int(q_next) - 3, int(landmark) + half_samp)
        t_hi = min(ecg_len - 1, int(t_hi))
    if t_hi <= t_lo:
        t_hi = min(ecg_len - 1, t_lo + max(3, half_samp))
    return int(t_lo), int(t_hi)


def _diagnosis_bucket(cause: str) -> str:
    if cause == "window_width_wrong":
        return "window_width_wrong"
    if cause in ("landmark_too_early", "landmark_too_late"):
        return "landmark_wrong"
    if cause == "template_misaligned":
        return "template_misaligned"
    if cause in ("beat_morphology_changed", "cluster_mismatch"):
        return "beat_unlike_template"
    if cause == "rr_adaptation_failure":
        return "template_misaligned"
    return "landmark_wrong"


def classify_bad_window_cause(
    *,
    manual_t: int,
    landmark: Optional[int],
    t_lo: int,
    t_hi: int,
    r_idx: int,
    fs: float,
    beat_template_corr: float,
    rr_stretch_ratio: float,
    cluster_landmark_err_ms: Optional[float] = None,
    record_landmark_err_ms: Optional[float] = None,
) -> str:
    """
    Priority-3 taxonomy for beats where manual T ∉ [t_lo, t_hi].

    Returns one of:
      landmark_too_early, landmark_too_late, window_width_wrong,
      template_misaligned, cluster_mismatch, beat_morphology_changed,
      rr_adaptation_failure
    """
    margin_lo_ms = _ms(manual_t - t_lo, fs)
    margin_hi_ms = _ms(t_hi - manual_t, fs)

    if landmark is None:
        if np.isfinite(beat_template_corr) and beat_template_corr < ALIGN_OK_CORR:
            return "template_misaligned"
        return "landmark_too_late"

    landmark_err_ms = _ms(landmark - manual_t, fs)

    if abs(landmark_err_ms) <= LANDMARK_OK_MS:
        return "window_width_wrong"

    if margin_hi_ms < 0 and landmark_err_ms < -LANDMARK_OK_MS:
        return "landmark_too_early"

    if margin_lo_ms < 0 and landmark_err_ms > LANDMARK_OK_MS:
        return "landmark_too_late"

    if (
        cluster_landmark_err_ms is not None
        and record_landmark_err_ms is not None
        and abs(cluster_landmark_err_ms) > LANDMARK_OK_MS
        and abs(record_landmark_err_ms) <= LANDMARK_OK_MS
    ):
        return "cluster_mismatch"

    lo_rr, hi_rr = RR_STRETCH_OK
    if np.isfinite(rr_stretch_ratio) and (
        rr_stretch_ratio < lo_rr or rr_stretch_ratio > hi_rr
    ):
        if np.isfinite(beat_template_corr) and beat_template_corr >= ALIGN_OK_CORR:
            return "rr_adaptation_failure"

    if np.isfinite(beat_template_corr) and beat_template_corr < ALIGN_OK_CORR:
        return "beat_morphology_changed"

    if landmark_err_ms < 0:
        return "landmark_too_early"
    return "landmark_too_late"


def analyze_bad_window_beat(
    *,
    manual_t: int,
    prior_t_lo: int,
    prior_t_hi: int,
    r_idx: int,
    s_i: int,
    q_next: int,
    tmpl: MedianBeatTemplate,
    cfg: ProcessCycleConfig,
    ecg_delim: np.ndarray,
    fs: float,
    median_stpq_span: float,
    record_tmpl: Optional[MedianBeatTemplate] = None,
    cluster_id: Optional[int] = None,
    cluster_tmpl: Optional[MedianBeatTemplate] = None,
) -> BadWindowBeatDiagnostics:
    landmark, land_src = morphology_rescue_landmark_global(
        int(s_i), int(q_next), tmpl, fs, cfg
    )
    beat_corr = _beat_template_correlation(ecg_delim, s_i, q_next, tmpl)
    span = float(q_next - s_i)
    rr_ratio = span / median_stpq_span if median_stpq_span > 0 else float("nan")

    rec_land_err: Optional[float] = None
    clust_land_err: Optional[float] = None
    if record_tmpl is not None and record_tmpl.valid:
        rec_lm, _ = morphology_rescue_landmark_global(
            int(s_i), int(q_next), record_tmpl, fs, cfg
        )
        if rec_lm is not None:
            rec_land_err = _ms(rec_lm - manual_t, fs)
    if cluster_tmpl is not None and cluster_tmpl.valid and cluster_id is not None:
        cl_lm, _ = morphology_rescue_landmark_global(
            int(s_i), int(q_next), cluster_tmpl, fs, cfg
        )
        if cl_lm is not None:
            clust_land_err = _ms(cl_lm - manual_t, fs)

    cause = classify_bad_window_cause(
        manual_t=manual_t,
        landmark=landmark,
        t_lo=prior_t_lo,
        t_hi=prior_t_hi,
        r_idx=r_idx,
        fs=fs,
        beat_template_corr=beat_corr,
        rr_stretch_ratio=rr_ratio,
        cluster_landmark_err_ms=clust_land_err,
        record_landmark_err_ms=rec_land_err,
    )

    land_rt = _ms(landmark - r_idx, fs) if landmark is not None else float("nan")
    return BadWindowBeatDiagnostics(
        manual_t=int(manual_t),
        landmark=landmark,
        landmark_source=land_src,
        t_lo=int(prior_t_lo),
        t_hi=int(prior_t_hi),
        manual_rt_ms=_ms(manual_t - r_idx, fs),
        landmark_rt_ms=land_rt,
        landmark_err_ms=_ms(landmark - manual_t, fs) if landmark is not None else float("nan"),
        window_width_ms=_ms(prior_t_hi - prior_t_lo, fs),
        margin_lo_ms=_ms(manual_t - prior_t_lo, fs),
        margin_hi_ms=_ms(prior_t_hi - manual_t, fs),
        beat_template_corr=beat_corr,
        rr_stretch_ratio=rr_ratio,
        cluster_id=cluster_id,
        record_landmark_err_ms=rec_land_err,
        cluster_landmark_err_ms=clust_land_err,
        bad_window_cause=cause,
        diagnosis_bucket=_diagnosis_bucket(cause),
    )


def summarize_bad_window_causes(rows: List[BadWindowBeatDiagnostics]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.bad_window_cause] = counts.get(row.bad_window_cause, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def summarize_diagnosis_buckets(rows: List[BadWindowBeatDiagnostics]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.diagnosis_bucket] = counts.get(row.diagnosis_bucket, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
