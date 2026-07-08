"""
sel16420-style post-positive-peak derivative-zero preference (ablation only).

Landmark-definition fix within the positive T lobe: template-guided max amplitude
(``positive_peak``) is systematically early vs manual/ECGPUWAVE; the preferred
fiducial is the first downslope d1 zero-crossing after that apex
(``post_apex_dz`` / ``downslope_dz_after_positive_peak``).

Record gate (ablation cohort):
  - upright positive T morphology (not biphasic +− routing)
  - majority of beats show qualified post-apex dz (+10..+80 ms after positive peak,
    still on positive lobe, before terminal negative component)

Enabled only when ``record_stpq_post_apex_dz_preference`` is True.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.signal import find_peaks

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.delineation_signal import savgol_search_segment

MORPH_POST_APEX_DZ_PREFERENCE = "post_apex_dz_preference"


def _ms_to_samples(ms: float, fs: float) -> int:
    return int(round(ms * fs / 1000.0))


def _t_search_window(n: int, cfg: ProcessCycleConfig) -> Tuple[int, int]:
    """S→Q fractional window for positive-peak / post-apex dz probe (full T lobe)."""
    lo_f, hi_f = cfg.record_template_t_amplitude_norm_sq_frac
    i0 = int(round(float(lo_f) * (n - 1)))
    i1 = int(round(float(hi_f) * (n - 1)))
    i0 = max(0, min(i0, n - 2))
    i1 = max(i0 + 1, min(i1, n - 1))
    return i0, i1


def downslope_dz_after_positive_peak(
    seg: np.ndarray,
    lo: int,
    *,
    after_abs: int,
) -> Optional[int]:
    """First downslope d1 zero-crossing at/after positive apex (absolute sample)."""
    if seg.size < 4:
        return None
    d1 = np.gradient(seg.astype(float))
    for i in range(1, d1.size):
        if d1[i - 1] >= 0.0 and d1[i] < 0.0:
            abs_i = lo + i
            if abs_i >= int(after_abs):
                return abs_i
    return None


def _segment_baseline(sig: np.ndarray, i0: int, fs: float) -> float:
    st_ref = max(0, i0 - max(1, _ms_to_samples(30.0, fs)))
    if i0 > st_ref:
        return float(np.median(sig[st_ref:i0]))
    return float(np.median(sig[: max(1, i0 + 1)]))


def _dominant_positive_peak_abs(
    seg: np.ndarray,
    lo: int,
    baseline: float,
    fs: float,
) -> Optional[int]:
    rel = seg.astype(float) - baseline
    if rel.size < 3:
        return None
    prom = max(0.01, 0.10 * float(np.std(rel)))
    dist = max(1, _ms_to_samples(25.0, fs))
    pos_idx, _ = find_peaks(rel, prominence=prom, distance=dist)
    if pos_idx.size == 0:
        j = int(np.argmax(rel))
        return lo + j if rel[j] > 0 else None
    j = int(pos_idx[np.argmax(rel[pos_idx])])
    return lo + j


def post_apex_dz_before_terminal_negative(
    seg: np.ndarray,
    lo: int,
    pos_abs: int,
    dz_abs: int,
    baseline: float,
    fs: float,
) -> bool:
    """True when dz remains on the positive lobe and precedes terminal negative peak."""
    rel = seg.astype(float) - baseline
    dz_rel = int(dz_abs) - int(lo)
    pos_rel = int(pos_abs) - int(lo)
    if dz_rel < 0 or dz_rel >= rel.size or pos_rel < 0 or pos_rel >= rel.size:
        return False
    if rel[dz_rel] <= 0.0:
        return False
    tail = rel[pos_rel:]
    if tail.size < 3:
        return True
    prom = max(0.01, 0.12 * float(np.std(rel)))
    dist = max(1, _ms_to_samples(30.0, fs))
    neg_idx, _ = find_peaks(-tail, prominence=prom, distance=dist)
    if neg_idx.size == 0:
        return True
    first_neg_abs = int(lo) + int(pos_rel) + int(neg_idx[0])
    return int(dz_abs) < first_neg_abs


def qualified_post_apex_dz_pair(
    seg: np.ndarray,
    lo: int,
    pos_abs: int,
    dz_abs: int,
    baseline: float,
    fs: float,
    cfg: ProcessCycleConfig,
) -> bool:
    """Post-apex dz within configured late window and before terminal negative lobe."""
    min_late = _ms_to_samples(
        float(getattr(cfg, "record_stpq_post_apex_dz_min_late_ms", 10.0)),
        fs,
    )
    max_late = _ms_to_samples(
        float(getattr(cfg, "record_stpq_post_apex_dz_max_late_ms", 80.0)),
        fs,
    )
    late = int(dz_abs) - int(pos_abs)
    if late < min_late or late > max_late:
        return False
    return post_apex_dz_before_terminal_negative(seg, lo, pos_abs, dz_abs, baseline, fs)


def probe_post_apex_dz_beat_fraction(
    ecg: np.ndarray,
    beat_anchors: Sequence[Tuple[int, int, int]],
    tmpl,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> float:
    """
    Fraction of beats with template-guided positive_peak + qualified post_apex_dz.

    Uses the same STPQ search path as runtime (template-guided max amplitude).
    """
    from pyhearts.processing.delineation_signal import smooth_search_window
    from pyhearts.processing.record_stpq_detection import (
        _search_t_template_guided,
        _stpq_t_window_samples,
        project_t_center_sample,
    )

    wins = 0
    n = 0
    template = getattr(tmpl, "template", None)
    n_tpl = int(len(template)) if template is not None else 0
    if n_tpl < 6:
        return 0.0
    t_j = float(getattr(tmpl, "t_landmark_idx", 0))
    p_j = float(getattr(tmpl, "p_landmark_idx", 0))

    for s_i, q_next, r_idx in beat_anchors:
        if q_next <= s_i + 5:
            continue
        t_center = project_t_center_sample(int(s_i), int(q_next), tmpl, n_tpl, cfg)
        if t_center is None:
            continue
        t_lo, t_hi = _stpq_t_window_samples(
            int(s_i), int(q_next), t_j, p_j, n_tpl, cfg, tmpl=tmpl
        )
        pos_idx, _ = _search_t_template_guided(
            ecg, t_lo, t_hi, int(t_center), tmpl, sampling_rate, cfg
        )
        if pos_idx is None:
            continue
        n += 1
        if cfg.record_stpq_use_savgol:
            seg, lo, _ = smooth_search_window(ecg, t_lo, t_hi, sampling_rate, cfg)
        else:
            lo = int(t_lo)
            seg = ecg[lo : min(len(ecg), int(t_hi) + 1)].astype(float, copy=False)
        dz_idx = downslope_dz_after_positive_peak(seg, lo, after_abs=int(pos_idx))
        if dz_idx is None:
            continue
        st_ref = max(0, int(pos_idx) - lo - max(1, _ms_to_samples(30.0, sampling_rate)))
        baseline = float(np.median(seg[: st_ref + 1])) if st_ref >= 0 else float(np.median(seg))
        if qualified_post_apex_dz_pair(
            seg, lo, int(pos_idx), int(dz_idx), baseline, sampling_rate, cfg
        ):
            wins += 1
    return float(wins) / float(n) if n else 0.0


def probe_post_apex_dz_segment_fraction(
    beat_segments: Sequence[Union[np.ndarray, Iterable[float]]],
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> float:
    """
    Fraction of S→Q segments with qualified post-apex dz after dominant positive peak.

    sel16420 ≈ high; sel847 lower on QTDB v3.2.1 crop.
    """
    wins = 0
    n = 0
    for raw in beat_segments:
        seg = np.asarray(raw, dtype=float)
        if seg.size < 8:
            continue
        sig = (
            savgol_search_segment(seg, float(sampling_rate), cfg)
            if cfg.p_t_search_savgol
            else seg
        )
        i0, i1 = _t_search_window(sig.size, cfg)
        seg2 = sig[i0 : i1 + 1]
        baseline = _segment_baseline(sig, i0, sampling_rate)
        pos_abs = _dominant_positive_peak_abs(seg2, i0, baseline, sampling_rate)
        if pos_abs is None:
            continue
        dz_abs = downslope_dz_after_positive_peak(seg2, i0, after_abs=int(pos_abs))
        n += 1
        if dz_abs is None:
            continue
        if qualified_post_apex_dz_pair(
            seg2, i0, int(pos_abs), int(dz_abs), baseline, sampling_rate, cfg
        ):
            wins += 1
    return float(wins) / float(n) if n else 0.0


def _positive_t_morphology(tmpl) -> bool:
    morph = str(getattr(tmpl, "t_morphology", "normal") or "normal")
    if morph in ("inverted_t", "biphasic_positive_negative"):
        return False
    pol = str(getattr(tmpl, "t_polarity", "positive") or "positive")
    return pol == "positive" or morph in ("normal", "large_t")


def classify_post_apex_dz_preference_template(
    tmpl,
    cfg: ProcessCycleConfig,
    sampling_rate: float,
    *,
    beat_segments: Optional[Sequence[Union[np.ndarray, Iterable[float]]]] = None,
    beat_anchors: Optional[Sequence[Tuple[int, int, int]]] = None,
    ecg_work: Optional[np.ndarray] = None,
) -> bool:
    """
    True when record matches sel16420-style post-apex dz ablation cohort.

    Requires positive T morphology (not biphasic +−) and sufficient beats where
    template-guided positive_peak coexists with qualified post_apex_dz (+10..+80 ms,
    before terminal negative).
    """
    if not getattr(cfg, "record_stpq_post_apex_dz_preference", False):
        return False
    if tmpl is None or not getattr(tmpl, "valid", False):
        return False
    if not _positive_t_morphology(tmpl):
        return False

    if beat_anchors is not None and ecg_work is not None:
        frac = probe_post_apex_dz_beat_fraction(
            ecg_work, beat_anchors, tmpl, sampling_rate, cfg
        )
    elif beat_segments is not None:
        frac = probe_post_apex_dz_segment_fraction(beat_segments, sampling_rate, cfg)
    else:
        return False

    min_frac = float(getattr(cfg, "record_stpq_post_apex_dz_min_beat_frac", 0.20))
    max_frac = getattr(cfg, "record_stpq_post_apex_dz_max_beat_frac", 0.38)
    if frac < min_frac:
        return False
    if max_frac is not None and frac > float(max_frac):
        return False
    return True
