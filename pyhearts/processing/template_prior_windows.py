"""
Phase 1: S→Q template windows as priors for per-cycle P/T detection.

Builds a record-level STPQ template, projects beat-wise T (and optional P) search
windows from S_i → Q_{i+1}, and returns global sample bounds for ``process_cycle``.
Does not select or overwrite peaks — only constrains search regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.candidate_visibility import (
    p_search_window_samples,
    t_search_window_samples,
)
from pyhearts.processing.template_prior_window_diagnostics import (
    estimate_record_t_timing_sigma_ms,
    t_uncertainty_window_samples,
)
from pyhearts.processing.delineation_signal import prepare_record_delineation_signal
from pyhearts.processing.record_delineation import (
    MedianBeatTemplate,
    _stpq_s_q_anchor_indices,
    build_record_beat_template,
    delineate_record_template,
)


@dataclass(frozen=True)
class TemplatePriorBeatWindows:
    """Global-sample search bounds projected from the record STPQ template."""

    t_lo: int
    t_hi: int
    p_lo: Optional[int] = None
    p_hi: Optional[int] = None
    s_i: Optional[int] = None
    q_next: Optional[int] = None
    r_det: Optional[int] = None
    r_next: Optional[int] = None


def global_window_to_cycle_relative(
    lo_global: int,
    hi_global: int,
    cycle_start_global: int,
    cycle_len: int,
    *,
    min_size: int = 3,
) -> Optional[Tuple[int, int]]:
    """
    Map inclusive global [lo, hi] to cycle-relative [start, end) for slicing.

    ``end`` is exclusive (matches ``detect_t_wave_derivative_based``).
    """
    lo_rel = max(0, int(lo_global) - int(cycle_start_global))
    hi_rel = min(int(cycle_len), int(hi_global) - int(cycle_start_global) + 1)
    if hi_rel - lo_rel < min_size:
        return None
    return lo_rel, hi_rel


def build_delineated_stpq_template(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    *,
    manual_ann_ext: Optional[str] = None,
) -> MedianBeatTemplate:
    """Build and delineate the record S→Q template (landmarks required for projection)."""
    tmpl = build_record_beat_template(ecg, r_peaks, sampling_rate, cfg)
    if not tmpl.valid:
        return tmpl
    return delineate_record_template(
        tmpl, sampling_rate, cfg, manual_ann_ext=manual_ann_ext
    )


def compute_template_prior_windows(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    cycles: List[int],
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    *,
    manual_ann_ext: Optional[str] = None,
    cluster_templates: Optional[Dict[int, MedianBeatTemplate]] = None,
    cluster_by_epoch: Optional[Dict[int, int]] = None,
) -> Tuple[Optional[MedianBeatTemplate], Dict[int, TemplatePriorBeatWindows]]:
    """
    Per-cycle template-projected search windows keyed by ``cycle_idx`` (0..n_cycles-1).

    Requires a valid delineated template with P/T landmarks. Beats without valid
    S/Q anchors or without a following R are omitted from the returned map.
    """
    r_peaks = np.asarray(r_peaks, dtype=int)
    if r_peaks.size < cfg.record_delineation_min_beats + 1:
        return None, {}

    tmpl = build_delineated_stpq_template(
        ecg, r_peaks, sampling_rate, cfg, manual_ann_ext=manual_ann_ext
    )
    if not tmpl.valid:
        return tmpl, {}
    if tmpl.t_landmark_idx is None:
        return tmpl, {}

    ecg_delim = prepare_record_delineation_signal(ecg, sampling_rate, cfg)
    ecg_len = int(ecg_delim.size)
    windows: Dict[int, TemplatePriorBeatWindows] = {}

    t_sigma_ms: Optional[float] = None
    if cfg.record_template_prior_uncertainty_windows:
        t_sigma_ms = estimate_record_t_timing_sigma_ms(
            ecg,
            r_peaks,
            tmpl,
            sampling_rate,
            cfg,
            default_sigma_ms=float(cfg.record_template_prior_default_sigma_ms),
        )

    for cycle_idx, cycle_label in enumerate(cycles):
        epoch_i = int(cycle_label)
        if epoch_i < 0 or epoch_i >= len(r_peaks) - 1:
            continue

        r_det = int(r_peaks[epoch_i])
        r_next = int(r_peaks[epoch_i + 1])
        s_i, q_next = _stpq_s_q_anchor_indices(
            ecg_delim, r_det, r_next, sampling_rate, cfg
        )
        if s_i is None or q_next is None:
            continue

        beat_tmpl = tmpl
        if cluster_templates and cluster_by_epoch is not None:
            cid = cluster_by_epoch.get(epoch_i)
            if cid is not None and cid in cluster_templates:
                cluster_tmpl = cluster_templates[cid]
                if (
                    cluster_tmpl.valid
                    and cluster_tmpl.t_landmark_idx is not None
                    and cluster_tmpl.p_landmark_idx is not None
                ):
                    beat_tmpl = cluster_tmpl

        t_lo, t_hi = t_search_window_samples(
            ecg_len, int(s_i), int(q_next), r_det, beat_tmpl, sampling_rate, cfg
        )
        if cfg.record_template_prior_uncertainty_windows and t_sigma_ms is not None:
            from pyhearts.processing.t_landmark_ensemble import pick_ensemble_landmark
            from pyhearts.processing.t_morphology_routing import (
                morphology_rescue_landmark_global,
            )

            if cfg.record_template_prior_landmark_ensemble:
                landmark, _, _ = pick_ensemble_landmark(
                    ecg_delim,
                    r_det,
                    int(s_i),
                    int(q_next),
                    sampling_rate,
                    beat_tmpl,
                    cfg,
                )
            else:
                landmark, _ = morphology_rescue_landmark_global(
                    int(s_i), int(q_next), beat_tmpl, sampling_rate, cfg
                )
            if landmark is not None:
                t_lo, t_hi = t_uncertainty_window_samples(
                    int(landmark),
                    t_sigma_ms,
                    ecg_len,
                    int(s_i),
                    int(q_next),
                    r_det,
                    sampling_rate,
                    cfg,
                    sigma_mult=float(cfg.record_template_prior_sigma_mult),
                    min_half_width_ms=float(cfg.record_template_prior_min_half_width_ms),
                )

        p_lo: Optional[int] = None
        p_hi: Optional[int] = None
        if cfg.record_template_prior_apply_p and beat_tmpl.p_landmark_idx is not None:
            p_lo, p_hi = p_search_window_samples(
                ecg_len,
                int(s_i),
                int(q_next),
                r_det,
                r_next,
                beat_tmpl,
                sampling_rate,
                cfg,
            )

        windows[int(cycle_idx)] = TemplatePriorBeatWindows(
            t_lo=int(t_lo),
            t_hi=int(t_hi),
            p_lo=p_lo,
            p_hi=p_hi,
            s_i=int(s_i),
            q_next=int(q_next),
            r_det=r_det,
            r_next=r_next,
        )

    return tmpl, windows
