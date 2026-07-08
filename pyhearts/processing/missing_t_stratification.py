"""
Classify missing-T beats into candidate-absent vs candidate-existed buckets.

candidate_absent  — search window excludes manual T and/or generator returned no candidates
candidate_existed — window covers manual T and ≥1 candidate was generated, but export T is NaN
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.candidate_visibility import (
    generate_t_candidates_for_beat,
    t_search_window_samples,
)
from pyhearts.processing.delineation_signal import prepare_record_delineation_signal
from pyhearts.processing.record_delineation import _stpq_s_q_anchor_indices
from pyhearts.processing.t_candidate_scoring import (
    build_record_template_context,
    enrich_template_biphasic_landmarks,
    generate_t_candidates,
)
from pyhearts.processing.template_prior_windows import compute_template_prior_windows

PresetFamily = Literal["v321", "template_prior"]


@dataclass(frozen=True)
class MissingTCandidateState:
    bucket: Literal["candidate_absent", "candidate_existed"]
    reason: str
    manual_in_window: bool
    n_candidates: int


def _inside(sample: float, lo: int, hi: int) -> bool:
    return int(lo) <= int(round(sample)) <= int(hi)


def classify_missing_t_candidate_state(
    *,
    manual_t_crop: float,
    preset_family: PresetFamily,
    ecg_delim: np.ndarray,
    r_crop: int,
    s_i: int,
    q_next: int,
    fs: float,
    tmpl,
    cfg: ProcessCycleConfig,
    prior=None,
    prev_t: Optional[float] = None,
    next_t: Optional[float] = None,
) -> MissingTCandidateState:
    """Diagnose why a manual-T beat has no finite exported T."""
    if preset_family == "template_prior":
        if prior is None:
            return MissingTCandidateState(
                "candidate_absent", "no_prior", False, 0
            )
        t_lo, t_hi = int(prior.t_lo), int(prior.t_hi)
    else:
        if tmpl is None or not tmpl.valid:
            return MissingTCandidateState(
                "candidate_absent", "no_template", False, 0
            )
        t_lo, t_hi = t_search_window_samples(
            len(ecg_delim), int(s_i), int(q_next), int(r_crop), tmpl, fs, cfg
        )

    manual_in = _inside(manual_t_crop, t_lo, t_hi)
    if not manual_in:
        return MissingTCandidateState(
            "candidate_absent", "bad_window", False, 0
        )

    if preset_family == "template_prior":
        candidates, _ = generate_t_candidates(
            ecg_delim,
            int(r_crop),
            int(s_i),
            int(q_next),
            fs,
            tmpl=tmpl,
            cfg=cfg,
            neighbor_t_samples=(prev_t, next_t),
        )
    else:
        expected_rt = (
            (float(tmpl.t_landmark_idx) - float(r_crop)) * 1000.0 / fs
            if tmpl is not None and tmpl.t_landmark_idx is not None
            else 300.0
        )
        candidates = generate_t_candidates_for_beat(
            ecg_delim,
            int(r_crop),
            int(s_i),
            int(q_next),
            fs,
            tmpl=tmpl,
            cfg=cfg,
            expected_rt_ms=expected_rt,
            prev_t=prev_t,
            next_t=next_t,
        )

    n_cand = len(candidates)
    if n_cand == 0:
        return MissingTCandidateState(
            "candidate_absent", "no_candidate", True, 0
        )
    return MissingTCandidateState(
        "candidate_existed", f"n_candidates={n_cand}", True, n_cand
    )


def build_v321_template(seg: np.ndarray, r_crop: np.ndarray, fs: float, cfg: ProcessCycleConfig, ann_ext: str):
    tmpl, _ = build_record_template_context(seg, r_crop, fs, cfg, manual_ann_ext=ann_ext)
    if tmpl is not None and tmpl.valid:
        tmpl = enrich_template_biphasic_landmarks(tmpl.template, cfg, fs, tmpl)
    return tmpl


def anchors_for_beat(
    ecg_delim: np.ndarray,
    r_crop: int,
    r_next_crop: Optional[int],
    fs: float,
    cfg: ProcessCycleConfig,
) -> Tuple[Optional[int], Optional[int]]:
    try:
        s_i, q_next = _stpq_s_q_anchor_indices(
            ecg_delim, int(r_crop), r_next_crop, fs, cfg
        )
    except (ValueError, IndexError):
        return None, None
    if s_i is None or q_next is None:
        return None, None
    return int(s_i), int(q_next)


def summarize_missing_t_buckets(rows: List[MissingTCandidateState]) -> Dict[str, int]:
    out = {"candidate_absent": 0, "candidate_existed": 0}
    reasons: Dict[str, int] = {}
    for row in rows:
        out[row.bucket] += 1
        reasons[row.reason] = reasons.get(row.reason, 0) + 1
    return {**out, "reasons": reasons}  # type: ignore[return-value]
