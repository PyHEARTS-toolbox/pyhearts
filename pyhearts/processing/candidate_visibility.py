"""
Per-beat P/T candidate visibility diagnostics (research / benchmarking).

Answers, for manual vs production-selected peaks:
  - Is the manual peak inside the internal search window?
  - Was a manual-like candidate generated?
  - Was a manual-like candidate generated but not selected?
  - Selected − manual timing delta (ms).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks, peak_prominences

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.delineation_signal import smooth_search_window
from pyhearts.processing.record_delineation import MedianBeatTemplate
from pyhearts.processing.record_stpq_detection import (
    _apply_stpq_t_rt_cap,
    _local_extrema_indices,
    _stpq_t_window_samples,
    project_p_center_sample,
    stpq_p_window_samples,
)
from pyhearts.processing.t_candidate_scoring import (
    TCandidate,
    TBeatCandidateContext,
    generate_t_candidates,
    nearest_candidate,
)


@dataclass(frozen=True)
class PeakCandidateVisibility:
    wave: str
    manual_sample: Optional[float]
    selected_sample: Optional[float]
    window_lo: int
    window_hi: int
    n_candidates: int
    manual_inside_window: bool
    manual_like_candidate: bool
    manual_like_not_selected: bool
    nearest_candidate_delta_ms: float
    nearest_candidate_source: str
    selected_delta_ms: float
    manual_like_tolerance_ms: float

    def as_dict(self, prefix: str = "") -> Dict[str, object]:
        p = prefix
        return {
            f"{p}manual_sample": self.manual_sample,
            f"{p}selected_sample": self.selected_sample,
            f"{p}window_lo": self.window_lo,
            f"{p}window_hi": self.window_hi,
            f"{p}n_candidates": self.n_candidates,
            f"{p}manual_inside_window": self.manual_inside_window,
            f"{p}manual_like_candidate": self.manual_like_candidate,
            f"{p}manual_like_not_selected": self.manual_like_not_selected,
            f"{p}nearest_candidate_delta_ms": self.nearest_candidate_delta_ms,
            f"{p}nearest_candidate_source": self.nearest_candidate_source,
            f"{p}selected_delta_ms": self.selected_delta_ms,
            f"{p}manual_like_tolerance_ms": self.manual_like_tolerance_ms,
        }


def _ms_to_samples(ms: float, fs: float) -> int:
    return int(round(ms * fs / 1000.0))


def _sample_delta_ms(a: Optional[float], b: Optional[float], fs: float) -> float:
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return (float(a) - float(b)) * 1000.0 / fs


def _inside_window(sample: Optional[float], lo: int, hi: int) -> bool:
    if sample is None or not np.isfinite(sample):
        return False
    return int(lo) <= int(round(sample)) <= int(hi)


def diagnose_peak_candidates(
    *,
    wave: str,
    manual_sample: Optional[float],
    selected_sample: Optional[float],
    window_lo: int,
    window_hi: int,
    candidates: Sequence[TCandidate],
    fs: float,
    manual_like_tolerance_ms: float = 15.0,
) -> PeakCandidateVisibility:
    """Classify manual peak visibility against a candidate set and production pick."""
    tol_samp = _ms_to_samples(manual_like_tolerance_ms, fs)
    manual_inside = _inside_window(manual_sample, window_lo, window_hi)

    nearest, nearest_d_samp = nearest_candidate(manual_sample, candidates)
    nearest_delta_ms = _sample_delta_ms(manual_sample, nearest.sample_idx if nearest else None, fs)
    manual_like = (
        nearest is not None
        and np.isfinite(nearest_d_samp)
        and abs(nearest_d_samp) <= tol_samp
    )

    selected_delta_ms = _sample_delta_ms(selected_sample, manual_sample, fs)
    selected_ok = (
        selected_sample is not None
        and manual_sample is not None
        and np.isfinite(selected_sample)
        and np.isfinite(manual_sample)
        and abs(float(selected_sample) - float(manual_sample)) <= tol_samp
    )
    manual_like_not_selected = bool(manual_like and not selected_ok)

    return PeakCandidateVisibility(
        wave=wave,
        manual_sample=float(manual_sample) if manual_sample is not None and np.isfinite(manual_sample) else None,
        selected_sample=float(selected_sample) if selected_sample is not None and np.isfinite(selected_sample) else None,
        window_lo=int(window_lo),
        window_hi=int(window_hi),
        n_candidates=len(candidates),
        manual_inside_window=manual_inside,
        manual_like_candidate=manual_like,
        manual_like_not_selected=manual_like_not_selected,
        nearest_candidate_delta_ms=float(nearest_delta_ms),
        nearest_candidate_source=str(nearest.source) if nearest is not None else "",
        selected_delta_ms=float(selected_delta_ms),
        manual_like_tolerance_ms=float(manual_like_tolerance_ms),
    )


def _append_p_candidate(
    out: List[TCandidate],
    seen: Dict[int, str],
    *,
    sample_idx: int,
    r_idx: int,
    fs: float,
    source: str,
    signed_amp: float,
    prominence: float,
    merge_ms: float = 6.0,
) -> None:
    key = int(round(sample_idx / max(1, _ms_to_samples(merge_ms, fs))))
    if key in seen:
        seen[key] = f"{seen[key]}+{source}"
        return
    seen[key] = source
    pr_ms = (float(r_idx) - float(sample_idx)) * 1000.0 / fs
    out.append(
        TCandidate(
            sample_idx=int(sample_idx),
            rt_ms=float(pr_ms),
            source=source,
            signed_amp=float(signed_amp),
            prominence=float(prominence),
            width_ms=20.0,
            curvature=0.0,
            is_terminal=False,
            before_first_pos=False,
            sign_matches_template=signed_amp > 0,
            features={"pr_ms": float(pr_ms)},
        )
    )


def generate_p_candidates(
    ecg: np.ndarray,
    r_idx: int,
    s_i: int,
    q_next: int,
    r_next: Optional[int],
    fs: float,
    *,
    tmpl: Optional[MedianBeatTemplate],
    cfg: ProcessCycleConfig,
    p_lo: int,
    p_hi: int,
) -> List[TCandidate]:
    """Deduplicated P apex candidates in the beat-wise PR / STPQ search window."""
    r_idx = int(r_idx)
    p_lo = max(0, int(p_lo))
    p_hi = min(len(ecg) - 1, int(p_hi))
    if p_hi - p_lo < 3:
        return []

    if cfg.record_stpq_use_savgol:
        seg, lo, _ = smooth_search_window(ecg, p_lo, p_hi, fs, cfg)
    else:
        lo = p_lo
        seg = ecg[p_lo : p_hi + 1].astype(float, copy=False)

    baseline_lo = max(0, lo - _ms_to_samples(40.0, fs))
    baseline = float(np.median(ecg[baseline_lo:lo])) if lo > baseline_lo else float(seg[0])
    rel = seg - baseline
    prom_thresh = max(0.01, 0.12 * float(np.std(rel)))
    dist = max(1, _ms_to_samples(25.0, fs))

    candidates: List[TCandidate] = []
    seen: Dict[int, str] = {}

    pos_idx, _ = find_peaks(rel, prominence=prom_thresh, distance=dist)
    neg_idx, _ = find_peaks(-rel, prominence=prom_thresh, distance=dist)
    for idx in pos_idx:
        prom = float(peak_prominences(rel, [idx])[0][0])
        _append_p_candidate(
            candidates,
            seen,
            sample_idx=lo + int(idx),
            r_idx=r_idx,
            fs=fs,
            source="positive_peak",
            signed_amp=float(rel[idx]),
            prominence=prom,
        )
    for idx in neg_idx:
        prom = float(peak_prominences(-rel, [idx])[0][0])
        _append_p_candidate(
            candidates,
            seen,
            sample_idx=lo + int(idx),
            r_idx=r_idx,
            fs=fs,
            source="negative_peak",
            signed_amp=float(rel[idx]),
            prominence=prom,
        )

    for prefer in ("max", "min"):
        for i in _local_extrema_indices(seg, prefer=prefer):
            amp = float(seg[i])
            _append_p_candidate(
                candidates,
                seen,
                sample_idx=lo + int(i),
                r_idx=r_idx,
                fs=fs,
                source=f"local_{prefer}",
                signed_amp=amp - baseline,
                prominence=abs(amp - baseline),
            )

    if seg.size >= 4:
        d1 = np.gradient(seg.astype(float))
        for i in range(1, d1.size):
            if d1[i - 1] > 0 and d1[i] <= 0:
                _append_p_candidate(
                    candidates,
                    seen,
                    sample_idx=lo + i,
                    r_idx=r_idx,
                    fs=fs,
                    source="derivative_zero_crossing",
                    signed_amp=float(rel[i]),
                    prominence=float(abs(d1[i - 1])),
                )

    if tmpl is not None and tmpl.valid and tmpl.p_landmark_idx is not None:
        r_anchor = int(r_next) if cfg.record_stpq_p_r_anchor and getattr(cfg, "record_stpq_p_r_anchor_mode", "current_r") == "next_r" and r_next is not None else r_idx
        p_center = project_p_center_sample(int(r_anchor), tmpl, fs, cfg)
        if p_lo <= p_center <= p_hi:
            _append_p_candidate(
                candidates,
                seen,
                sample_idx=int(p_center),
                r_idx=r_idx,
                fs=fs,
                source="template_projected",
                signed_amp=float(ecg[int(p_center)] - baseline),
                prominence=0.1,
            )

    candidates.sort(key=lambda c: c.sample_idx)
    return candidates


def t_search_window_samples(
    ecg_len: int,
    s_i: int,
    q_next: int,
    r_idx: int,
    tmpl: MedianBeatTemplate,
    fs: float,
    cfg: ProcessCycleConfig,
) -> Tuple[int, int]:
    """Production STPQ T search window (w1 + RT cap)."""
    n_tpl = int(tmpl.template.size)
    t_j = float(tmpl.t_landmark_idx)
    p_j = float(tmpl.p_landmark_idx)
    t_lo, t_hi = _stpq_t_window_samples(
        int(s_i), int(q_next), t_j, p_j, n_tpl, cfg, tmpl=tmpl
    )
    t_lo, t_hi = _apply_stpq_t_rt_cap(t_lo, t_hi, int(s_i), int(q_next), int(r_idx), fs, cfg)
    t_hi = min(ecg_len - 1, int(t_hi))
    return int(t_lo), int(t_hi)


def p_search_window_samples(
    ecg_len: int,
    s_i: int,
    q_next: int,
    r_idx: int,
    r_next: Optional[int],
    tmpl: MedianBeatTemplate,
    fs: float,
    cfg: ProcessCycleConfig,
) -> Tuple[int, int]:
    """Production STPQ / PR-anchored P search window."""
    n_tpl = int(tmpl.template.size)
    p_lo, p_hi = stpq_p_window_samples(
        s_i=int(s_i),
        q_next=int(q_next),
        t_j=float(tmpl.t_landmark_idx),
        p_j=float(tmpl.p_landmark_idx),
        n_tpl=n_tpl,
        r_det=int(r_idx),
        r_next=r_next,
        sampling_rate=fs,
        cfg=cfg,
        signal_len=ecg_len,
    )
    p_hi = min(ecg_len - 1, int(p_hi))
    return int(p_lo), int(p_hi)


def build_t_candidate_context(
    *,
    r_idx: int,
    s_i: int,
    q_next: int,
    fs: float,
    tmpl: Optional[MedianBeatTemplate],
    expected_rt_ms: float,
    prev_t: Optional[float],
    next_t: Optional[float],
) -> TBeatCandidateContext:
    return TBeatCandidateContext(
        r_idx=int(r_idx),
        s_i=int(s_i),
        q_next=int(q_next),
        fs=float(fs),
        baseline=0.0,
        expected_rt_ms=float(expected_rt_ms),
        template_t_polarity=str(getattr(tmpl, "t_polarity", "positive") or "positive"),
        template_morphology=str(getattr(tmpl, "t_morphology", "normal") or "normal"),
    )


def generate_t_candidates_for_beat(
    ecg: np.ndarray,
    r_idx: int,
    s_i: int,
    q_next: int,
    fs: float,
    *,
    tmpl: Optional[MedianBeatTemplate],
    cfg: ProcessCycleConfig,
    expected_rt_ms: float,
    prev_t: Optional[float] = None,
    next_t: Optional[float] = None,
) -> List[TCandidate]:
    ctx = build_t_candidate_context(
        r_idx=r_idx,
        s_i=s_i,
        q_next=q_next,
        fs=fs,
        tmpl=tmpl,
        expected_rt_ms=expected_rt_ms,
        prev_t=prev_t,
        next_t=next_t,
    )
    cands, _ = generate_t_candidates(
        ecg,
        int(r_idx),
        int(s_i),
        int(q_next),
        fs,
        tmpl=tmpl,
        cfg=cfg,
        ctx=ctx,
        neighbor_t_samples=(prev_t, next_t),
    )
    return cands
