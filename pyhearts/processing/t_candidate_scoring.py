"""
ECGPUWAVE-style T candidate generation and hand-tuned scoring (research / diagnostics).

Does not alter production T selection or export presets. Use from analysis scripts
to compare candidates against manual / ECGPUWAVE before changing delineation logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.delineation_signal import smooth_search_window
from pyhearts.processing.record_delineation import (
    MedianBeatTemplate,
    _rising_edge_t_landmark,
    _stpq_s_q_anchor_indices,
)
from pyhearts.processing.record_stpq_detection import (
    _local_extrema_indices,
    _resolve_stpq_t_tpl_idx_for_projection,
    _search_t_rising_edge_onset,
    _tpl_index_to_sample,
    project_t_center_sample,
)
from pyhearts.processing.record_template_biphasic import (
    MORPH_BIPHASIC_POS_NEG,
    classify_biphasic_positive_negative,
)


@dataclass(frozen=True)
class TCandidateScoreWeights:
    """Hand-tuned linear score (higher = preferred). Not trained yet."""

    rt_distance_per_ms: float = -0.018
    prominence: float = 0.32
    curvature_norm: float = 0.12
    sign_consistency: float = 0.28
    neighbor_distance_per_ms: float = -0.012
    before_first_pos_per_ms: float = -0.035
    terminal_component: float = -0.22
    template_landmark_bonus: float = 0.18
    derivative_zero_bonus: float = 0.08
    shoulder_bonus: float = 0.06


@dataclass
class TBeatCandidateContext:
    r_idx: int
    s_i: int
    q_next: int
    fs: float
    baseline: float
    expected_rt_ms: float
    first_pos_rt_ms: Optional[float] = None
    template_t_polarity: str = "positive"
    template_morphology: str = "normal"
    neighbor_rt_ms: Optional[float] = None
    t_window_rt_lo_ms: float = 80.0
    t_window_rt_hi_ms: float = 500.0


@dataclass
class TCandidate:
    sample_idx: int
    rt_ms: float
    source: str
    signed_amp: float
    prominence: float
    width_ms: float
    curvature: float
    is_terminal: bool
    before_first_pos: bool
    sign_matches_template: bool
    features: Dict[str, float] = field(default_factory=dict)
    score: float = float("nan")

    def feature_row(self, prefix: str = "") -> Dict[str, float]:
        p = prefix
        out = {
            f"{p}sample_idx": float(self.sample_idx),
            f"{p}rt_ms": float(self.rt_ms),
            f"{p}signed_amp": float(self.signed_amp),
            f"{p}prominence": float(self.prominence),
            f"{p}width_ms": float(self.width_ms),
            f"{p}curvature": float(self.curvature),
            f"{p}is_terminal": float(self.is_terminal),
            f"{p}before_first_pos": float(self.before_first_pos),
            f"{p}sign_matches_template": float(self.sign_matches_template),
            f"{p}score": float(self.score),
        }
        for k, v in self.features.items():
            out[f"{p}{k}"] = float(v)
        return out


def _ms_to_samples(ms: float, fs: float) -> int:
    return int(round(ms * fs / 1000.0))


def _st_baseline(ecg: np.ndarray, s_i: int, r_idx: int, fs: float) -> float:
    st_lo = max(0, int(s_i))
    st_hi = min(len(ecg), int(r_idx) + _ms_to_samples(40.0, fs))
    if st_hi <= st_lo:
        return float(ecg[int(s_i)])
    return float(np.median(ecg[st_lo:st_hi]))


def _beat_search_segment(
    ecg: np.ndarray,
    lo: int,
    hi: int,
    fs: float,
    cfg: Optional[ProcessCycleConfig],
) -> Tuple[np.ndarray, int]:
    lo = max(0, int(lo))
    hi = min(len(ecg) - 1, int(hi))
    if hi - lo < 3:
        seg = ecg[lo : hi + 1].astype(float, copy=False)
        return seg, lo
    if cfg is not None and cfg.record_stpq_use_savgol:
        seg, seg_lo, _ = smooth_search_window(ecg, lo, hi, fs, cfg)
        return seg.astype(float, copy=False), int(seg_lo)
    return ecg[lo : hi + 1].astype(float, copy=False), lo


def _curvature_at(seg: np.ndarray, rel_i: int) -> float:
    if seg.size < 5:
        return 0.0
    d2 = np.gradient(np.gradient(seg))
    i = int(np.clip(rel_i, 0, d2.size - 1))
    return float(abs(d2[i]))


def _parabolic_subsample_peak(y: np.ndarray, i: int) -> float:
    """Sub-sample peak index via 3-point parabola; returns absolute float index into y."""
    i = int(i)
    if i < 1 or i >= len(y) - 1:
        return float(i)
    y0, y1, y2 = float(y[i - 1]), float(y[i]), float(y[i + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(i)
    delta = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
    return float(i) + delta


def _amplitude_plateau_midpoint(rel: np.ndarray, *, frac: float = 0.85) -> Optional[int]:
    """Midpoint of contiguous high-amplitude plateau around dominant signed extreme."""
    if rel.size < 5:
        return None
    sign = 1.0 if abs(float(np.max(rel))) >= abs(float(np.min(rel))) else -1.0
    signed = sign * rel
    peak = float(np.max(signed))
    if peak < 1e-9:
        return None
    above = np.where(signed >= frac * peak)[0]
    if above.size < 2:
        return int(np.argmax(signed))
    return int(round(0.5 * (float(above[0]) + float(above[-1]))))


def _refined_dzc_near_apex(seg: np.ndarray, rel: np.ndarray, apex_rel: int) -> Optional[int]:
    """Nearest derivative zero-crossing to the dominant apex (refined DZC)."""
    if seg.size < 5:
        return None
    d1 = np.gradient(seg.astype(float))
    zc = np.where(d1[:-1] * d1[1:] <= 0)[0]
    if zc.size == 0:
        return None
    j = int(np.argmin(np.abs(zc.astype(float) - float(apex_rel))))
    return int(zc[j])


def _append_candidate(
    out: List[TCandidate],
    seen: Dict[int, str],
    *,
    sample_idx: int,
    r_idx: int,
    fs: float,
    source: str,
    signed_amp: float,
    prominence: float,
    width_ms: float,
    curvature: float,
    ctx: TBeatCandidateContext,
    merge_ms: float = 6.0,
    merge_same_primary_only: bool = False,
) -> None:
    rt_ms = (float(sample_idx) - float(r_idx)) * 1000.0 / fs
    if rt_ms < ctx.t_window_rt_lo_ms or rt_ms > ctx.t_window_rt_hi_ms:
        return
    key = int(round(sample_idx / max(1, _ms_to_samples(merge_ms, fs))))
    primary = source.split("+")[0].strip()
    if key in seen:
        existing_primaries = {
            s.split("+")[0].strip() for s in seen[key].split("+")
        }
        # Default: merge any source at the same time bin.
        # For refined landmarks: still append if the primary source is new.
        if (not merge_same_primary_only) or (primary in existing_primaries):
            seen[key] = f"{seen[key]}+{source}"
            return
        seen[key] = f"{seen[key]}+{source}"
    else:
        seen[key] = source
    is_terminal = rt_ms > ctx.t_window_rt_lo_ms + 0.6 * (
        ctx.t_window_rt_hi_ms - ctx.t_window_rt_lo_ms
    )
    before_first = False
    if ctx.first_pos_rt_ms is not None and np.isfinite(ctx.first_pos_rt_ms):
        before_first = rt_ms < float(ctx.first_pos_rt_ms) - 8.0
    want_pos = ctx.template_t_polarity != "negative"
    sign_ok = (signed_amp > 0 and want_pos) or (signed_amp < 0 and not want_pos)
    out.append(
        TCandidate(
            sample_idx=int(sample_idx),
            rt_ms=float(rt_ms),
            source=source,
            signed_amp=float(signed_amp),
            prominence=float(prominence),
            width_ms=float(width_ms),
            curvature=float(curvature),
            is_terminal=bool(is_terminal),
            before_first_pos=bool(before_first),
            sign_matches_template=bool(sign_ok),
            features={
                "dist_expected_rt_ms": float(rt_ms - ctx.expected_rt_ms),
                "dist_neighbor_rt_ms": float("nan"),
                "dist_first_pos_ms": float(rt_ms - ctx.first_pos_rt_ms)
                if ctx.first_pos_rt_ms is not None
                else float("nan"),
            },
        )
    )


def generate_t_candidates(
    ecg: np.ndarray,
    r_idx: int,
    s_i: int,
    q_next: int,
    fs: float,
    *,
    tmpl: Optional[MedianBeatTemplate] = None,
    cfg: Optional[ProcessCycleConfig] = None,
    ctx: Optional[TBeatCandidateContext] = None,
    neighbor_t_samples: Optional[Tuple[Optional[int], Optional[int]]] = None,
) -> Tuple[List[TCandidate], TBeatCandidateContext]:
    """
    Build a deduplicated candidate set for one beat's ST–T segment (S→Q anchors).
    """
    r_idx = int(r_idx)
    s_i = int(s_i)
    q_next = int(q_next)
    if q_next <= s_i + 3:
        q_next = min(len(ecg) - 1, s_i + _ms_to_samples(400.0, fs))

    if ctx is None:
        expected_rt = 200.0
        if tmpl is not None and tmpl.t_landmark_idx is not None and tmpl.template.size >= 2:
            n_tpl = int(tmpl.template.size)
            frac = float(tmpl.t_landmark_idx) / float(n_tpl - 1)
            expected_rt = frac * (q_next - s_i) / fs * 1000.0
        ctx = TBeatCandidateContext(
            r_idx=r_idx,
            s_i=s_i,
            q_next=q_next,
            fs=float(fs),
            baseline=_st_baseline(ecg, s_i, r_idx, fs),
            expected_rt_ms=float(expected_rt),
            template_t_polarity=str(getattr(tmpl, "t_polarity", "positive") or "positive"),
            template_morphology=str(getattr(tmpl, "t_morphology", "normal") or "normal"),
        )

    if neighbor_t_samples is not None:
        prev_t, next_t = neighbor_t_samples
        nbs = [
            (float(prev_t) - r_idx) * 1000.0 / fs
            for prev_t in [prev_t]
            if prev_t is not None and np.isfinite(prev_t)
        ] + [
            (float(next_t) - r_idx) * 1000.0 / fs
            for next_t in [next_t]
            if next_t is not None and np.isfinite(next_t)
        ]
        if nbs:
            ctx.neighbor_rt_ms = float(np.median(nbs))

    if tmpl is not None and getattr(tmpl, "t_morphology", "") == MORPH_BIPHASIC_POS_NEG:
        pos_tpl = getattr(tmpl, "t_biphasic_pos_landmark_idx", None)
        if pos_tpl is not None and tmpl.template.size >= 2:
            pos_s = _tpl_index_to_sample(s_i, q_next, float(pos_tpl), int(tmpl.template.size))
            ctx.first_pos_rt_ms = (float(pos_s) - r_idx) * 1000.0 / fs

    seg, lo = _beat_search_segment(ecg, s_i, q_next, fs, cfg)
    rel = seg - float(ctx.baseline)
    prom_thresh = max(0.01, 0.12 * float(np.std(rel)))
    dist = max(1, _ms_to_samples(30.0, fs))

    candidates: List[TCandidate] = []
    seen: Dict[int, str] = {}

    pos_idx, _ = find_peaks(rel, prominence=prom_thresh, distance=dist)
    neg_idx, _ = find_peaks(-rel, prominence=prom_thresh, distance=dist)

    for idx in pos_idx:
        prom = float(peak_prominences(rel, [idx])[0][0])
        w = float(peak_widths(rel, [idx], rel_height=0.5)[0][0]) * 1000.0 / fs
        _append_candidate(
            candidates,
            seen,
            sample_idx=lo + int(idx),
            r_idx=r_idx,
            fs=fs,
            source="positive_peak",
            signed_amp=float(rel[idx]),
            prominence=prom,
            width_ms=w,
            curvature=_curvature_at(seg, idx),
            ctx=ctx,
        )
        try:
            mid = float(peak_widths(rel, [idx], rel_height=0.5)[3][0])
            mid_i = int(round(mid))
            if 0 <= mid_i < rel.size:
                _append_candidate(
                    candidates,
                    seen,
                    sample_idx=lo + mid_i,
                    r_idx=r_idx,
                    fs=fs,
                    source="plateau_midpoint",
                    signed_amp=float(rel[mid_i]),
                    prominence=prom * 0.85,
                    width_ms=w,
                    curvature=_curvature_at(seg, mid_i),
                    ctx=ctx,
                )
        except Exception:
            pass

    for idx in neg_idx:
        prom = float(peak_prominences(-rel, [idx])[0][0])
        w = float(peak_widths(-rel, [idx], rel_height=0.5)[0][0]) * 1000.0 / fs
        _append_candidate(
            candidates,
            seen,
            sample_idx=lo + int(idx),
            r_idx=r_idx,
            fs=fs,
            source="negative_peak",
            signed_amp=float(rel[idx]),
            prominence=prom,
            width_ms=w,
            curvature=_curvature_at(seg, idx),
            ctx=ctx,
        )

    if pos_idx.size:
        dom_pos = int(pos_idx[np.argmax(rel[pos_idx])])
        search_end = int(neg_idx[0]) if neg_idx.size else rel.size - 1
        if search_end > dom_pos + 2:
            shoulder_rel = int(dom_pos + 1 + np.argmin(rel[dom_pos + 1 : search_end]))
            _append_candidate(
                candidates,
                seen,
                sample_idx=lo + shoulder_rel,
                r_idx=r_idx,
                fs=fs,
                source="post_positive_shoulder",
                signed_amp=float(rel[shoulder_rel]),
                prominence=float(rel[dom_pos] - rel[shoulder_rel]),
                width_ms=30.0,
                curvature=_curvature_at(seg, shoulder_rel),
                ctx=ctx,
            )

    if seg.size < 5:
        for c in candidates:
            if ctx.neighbor_rt_ms is not None and np.isfinite(ctx.neighbor_rt_ms):
                c.features["dist_neighbor_rt_ms"] = float(c.rt_ms - ctx.neighbor_rt_ms)
        candidates.sort(key=lambda c: c.sample_idx)
        return candidates, ctx

    d1 = np.gradient(seg)
    d2 = np.gradient(d1)
    if d2.size >= 5:
        cur_i = int(np.argmax(np.abs(d2[2:-2]))) + 2
        _append_candidate(
            candidates,
            seen,
            sample_idx=lo + cur_i,
            r_idx=r_idx,
            fs=fs,
            source="max_curvature",
            signed_amp=float(rel[cur_i]),
            prominence=float(abs(rel[cur_i])),
            width_ms=20.0,
            curvature=float(abs(d2[cur_i])),
            ctx=ctx,
        )

    for i in range(1, d1.size):
        if d1[i - 1] > 0 and d1[i] <= 0:
            _append_candidate(
                candidates,
                seen,
                sample_idx=lo + i,
                r_idx=r_idx,
                fs=fs,
                source="derivative_zero_crossing",
                signed_amp=float(rel[i]),
                prominence=float(abs(d1[i - 1])),
                width_ms=15.0,
                curvature=_curvature_at(seg, i),
                ctx=ctx,
            )
        if d1[i - 1] < 0 and d1[i] >= 0:
            _append_candidate(
                candidates,
                seen,
                sample_idx=lo + i,
                r_idx=r_idx,
                fs=fs,
                source="derivative_zero_crossing",
                signed_amp=float(rel[i]),
                prominence=float(abs(d1[i - 1])),
                width_ms=15.0,
                curvature=_curvature_at(seg, i),
                ctx=ctx,
            )

    abs_exc = np.abs(rel)
    rising = _rising_edge_t_landmark(abs_exc, 0)
    if rising is not None:
        ri = int(rising)
        _append_candidate(
            candidates,
            seen,
            sample_idx=lo + ri,
            r_idx=r_idx,
            fs=fs,
            source="rising_edge_onset",
            signed_amp=float(rel[ri]),
            prominence=float(abs_exc[ri]),
            width_ms=25.0,
            curvature=_curvature_at(seg, ri),
            ctx=ctx,
        )

    rising_stpq = _search_t_rising_edge_onset(seg, lo, prefer_negative=False, edge_frac=0.5)
    if rising_stpq is not None:
        idx, _ = rising_stpq
        _append_candidate(
            candidates,
            seen,
            sample_idx=int(idx),
            r_idx=r_idx,
            fs=fs,
            source="rising_edge_onset",
            signed_amp=float(ecg[int(idx)] - ctx.baseline),
            prominence=0.05,
            width_ms=20.0,
            curvature=0.0,
            ctx=ctx,
        )

    # Free geometry refinements: amplitude plateau midpoint, parabolic apex, refined DZC.
    # These are distinct from peak_widths plateau_midpoint / raw peak samples.
    amp_mid = _amplitude_plateau_midpoint(rel, frac=0.85)
    if amp_mid is not None:
        _append_candidate(
            candidates,
            seen,
            sample_idx=lo + int(amp_mid),
            r_idx=r_idx,
            fs=fs,
            source="amplitude_plateau_midpoint",
            signed_amp=float(rel[int(amp_mid)]),
            prominence=float(abs(rel[int(amp_mid)])),
            width_ms=30.0,
            curvature=_curvature_at(seg, int(amp_mid)),
            ctx=ctx,
            merge_same_primary_only=True,
        )

    apex_rel = int(np.argmax(np.abs(rel))) if rel.size else None
    if apex_rel is not None:
        sub = _parabolic_subsample_peak(rel, apex_rel)
        apex_i = int(round(sub))
        apex_i = int(np.clip(apex_i, 0, rel.size - 1))
        _append_candidate(
            candidates,
            seen,
            sample_idx=lo + apex_i,
            r_idx=r_idx,
            fs=fs,
            source="refined_apex",
            signed_amp=float(rel[apex_i]),
            prominence=float(abs(rel[apex_i])),
            width_ms=25.0,
            curvature=_curvature_at(seg, apex_i),
            ctx=ctx,
            merge_same_primary_only=True,
        )
        dzc_rel = _refined_dzc_near_apex(seg, rel, apex_rel)
        if dzc_rel is not None:
            _append_candidate(
                candidates,
                seen,
                sample_idx=lo + int(dzc_rel),
                r_idx=r_idx,
                fs=fs,
                source="refined_dzc",
                signed_amp=float(rel[int(dzc_rel)]),
                prominence=float(abs(rel[int(dzc_rel)])),
                width_ms=15.0,
                curvature=_curvature_at(seg, int(dzc_rel)),
                ctx=ctx,
                merge_same_primary_only=True,
            )

    if tmpl is not None and tmpl.valid and tmpl.template.size >= 2:
        n_tpl = int(tmpl.template.size)
        proj = project_t_center_sample(s_i, q_next, tmpl, n_tpl, cfg)
        if proj is not None:
            _append_candidate(
                candidates,
                seen,
                sample_idx=int(proj),
                r_idx=r_idx,
                fs=fs,
                source="template_projected_apex",
                signed_amp=float(ecg[int(proj)] - ctx.baseline),
                prominence=0.1,
                width_ms=40.0,
                curvature=0.0,
                ctx=ctx,
            )
        tpl_idx = _resolve_stpq_t_tpl_idx_for_projection(tmpl, cfg)
        if tpl_idx is not None:
            land_s = _tpl_index_to_sample(s_i, q_next, float(tpl_idx), n_tpl)
            _append_candidate(
                candidates,
                seen,
                sample_idx=int(land_s),
                r_idx=r_idx,
                fs=fs,
                source="template_landmark",
                signed_amp=float(ecg[int(land_s)] - ctx.baseline),
                prominence=0.1,
                width_ms=40.0,
                curvature=0.0,
                ctx=ctx,
            )
        land = getattr(tmpl, "t_landmark_idx", None)
        if land is not None:
            ls = _tpl_index_to_sample(s_i, q_next, float(land), n_tpl)
            _append_candidate(
                candidates,
                seen,
                sample_idx=int(ls),
                r_idx=r_idx,
                fs=fs,
                source="template_landmark",
                signed_amp=float(ecg[int(ls)] - ctx.baseline),
                prominence=0.1,
                width_ms=40.0,
                curvature=0.0,
                ctx=ctx,
            )

    for c in candidates:
        if ctx.neighbor_rt_ms is not None and np.isfinite(ctx.neighbor_rt_ms):
            c.features["dist_neighbor_rt_ms"] = float(c.rt_ms - ctx.neighbor_rt_ms)

    candidates.sort(key=lambda c: c.sample_idx)
    return candidates, ctx


def score_t_candidates(
    candidates: Sequence[TCandidate],
    ctx: TBeatCandidateContext,
    *,
    weights: Optional[TCandidateScoreWeights] = None,
) -> List[TCandidate]:
    """Apply hand-tuned linear score; returns new list sorted by score descending."""
    w = weights or TCandidateScoreWeights()
    if not candidates:
        return []

    prom_ref = max(c.prominence for c in candidates) or 1.0
    curv_ref = max(c.curvature for c in candidates) or 1.0

    scored: List[TCandidate] = []
    for c in candidates:
        dist_rt = abs(float(c.features.get("dist_expected_rt_ms", c.rt_ms - ctx.expected_rt_ms)))
        dist_nb = c.features.get("dist_neighbor_rt_ms", float("nan"))
        dist_nb_abs = abs(float(dist_nb)) if np.isfinite(dist_nb) else 0.0

        s = 0.0
        s += w.rt_distance_per_ms * dist_rt
        s += w.prominence * (c.prominence / prom_ref)
        s += w.curvature_norm * (c.curvature / curv_ref)
        s += w.sign_consistency * (1.0 if c.sign_matches_template else 0.0)
        s += w.neighbor_distance_per_ms * dist_nb_abs
        if c.before_first_pos and ctx.first_pos_rt_ms is not None:
            s += w.before_first_pos_per_ms * max(
                0.0, float(ctx.first_pos_rt_ms) - c.rt_ms
            )
        if c.is_terminal:
            s += w.terminal_component
        if "template" in c.source:
            s += w.template_landmark_bonus
        if c.source == "derivative_zero_crossing":
            s += w.derivative_zero_bonus
        if c.source == "post_positive_shoulder":
            s += w.shoulder_bonus

        scored.append(
            TCandidate(
                sample_idx=c.sample_idx,
                rt_ms=c.rt_ms,
                source=c.source,
                signed_amp=c.signed_amp,
                prominence=c.prominence,
                width_ms=c.width_ms,
                curvature=c.curvature,
                is_terminal=c.is_terminal,
                before_first_pos=c.before_first_pos,
                sign_matches_template=c.sign_matches_template,
                features=dict(c.features),
                score=float(s),
            )
        )

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored


def pick_best_t_candidate(
    candidates: Sequence[TCandidate],
    ctx: TBeatCandidateContext,
    *,
    weights: Optional[TCandidateScoreWeights] = None,
) -> Optional[TCandidate]:
    scored = score_t_candidates(candidates, ctx, weights=weights)
    return scored[0] if scored else None


def nearest_candidate(
    target_sample: Optional[float],
    candidates: Sequence[TCandidate],
) -> Tuple[Optional[TCandidate], float]:
    if target_sample is None or not np.isfinite(target_sample) or not candidates:
        return None, float("nan")
    best = min(candidates, key=lambda c: abs(c.sample_idx - float(target_sample)))
    return best, float(target_sample) - float(best.sample_idx)


def build_record_template_context(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    fs: float,
    cfg: ProcessCycleConfig,
    *,
    manual_ann_ext: Optional[str] = None,
) -> Tuple[Optional[MedianBeatTemplate], float]:
    """Median STPQ template + record-level expected RT (ms post-R) for scoring prior."""
    from pyhearts.processing.record_delineation import (
        build_record_beat_template,
        delineate_record_template,
    )

    raw = build_record_beat_template(ecg, r_peaks, fs, cfg)
    tmpl = delineate_record_template(raw, fs, cfg, manual_ann_ext=manual_ann_ext)
    if not tmpl.valid or tmpl.t_landmark_idx is None:
        return tmpl, 200.0
    n = int(tmpl.template.size)
    frac = float(tmpl.t_landmark_idx) / max(1, n - 1)
    median_rr_ms = float(tmpl.median_rr_samples) / fs * 1000.0 if tmpl.median_rr_samples else 800.0
    expected_rt = frac * median_rr_ms * 0.45
    return tmpl, float(expected_rt)


def enrich_template_biphasic_landmarks(
    template: np.ndarray,
    cfg: ProcessCycleConfig,
    fs: float,
    tmpl: MedianBeatTemplate,
) -> MedianBeatTemplate:
    """Attach biphasic landmarks on template for first+ peak context (no production side effects)."""
    morph, pos_i, neg_i = classify_biphasic_positive_negative(template, cfg, fs)
    if morph != MORPH_BIPHASIC_POS_NEG:
        return tmpl
    from dataclasses import replace

    return replace(
        tmpl,
        t_morphology=morph,
        t_biphasic_pos_landmark_idx=float(pos_i) if pos_i is not None else None,
        t_biphasic_neg_landmark_idx=float(neg_i) if neg_i is not None else None,
    )
