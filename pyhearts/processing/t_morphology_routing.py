"""
Morphology-specific T candidate scoring and rescue landmark routing (Phase 2C).

Does not change production STPQ overwrite; used with template-prior + rescue path.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.record_delineation import MedianBeatTemplate
from pyhearts.processing.record_stpq_detection import (
    _resolve_stpq_t_tpl_idx_for_projection,
    _tpl_index_to_sample,
    project_t_center_sample,
)
from pyhearts.processing.record_template_biphasic import MORPH_BIPHASIC_POS_NEG
from pyhearts.processing.t_candidate_scoring import TCandidateScoreWeights


def normalize_t_morphology_tag(morphology: str) -> str:
    m = str(morphology or "normal").strip().lower()
    if m in ("biphasic_positive_negative", "biphasic_pm", "biphasic_-+") or (
        "biphasic" in m and "positive" in m and "negative" in m
    ):
        return MORPH_BIPHASIC_POS_NEG
    if m in ("inverted_t", "inverted"):
        return "inverted_t"
    if m in ("rising_edge", "rising_edge_inverted_morphology", "large_t"):
        return "rising_edge"
    if m in ("plateau",):
        return "plateau"
    return "normal"


def morphology_scoring_weights(morphology: str) -> TCandidateScoreWeights:
    """Hand-tuned linear score weights per template morphology class."""
    tag = normalize_t_morphology_tag(morphology)
    if tag == MORPH_BIPHASIC_POS_NEG:
        return TCandidateScoreWeights(
            template_landmark_bonus=0.40,
            before_first_pos_per_ms=-0.02,
            prominence=0.28,
            sign_consistency=0.32,
            rt_distance_per_ms=-0.012,
        )
    if tag == "plateau":
        return TCandidateScoreWeights(
            template_landmark_bonus=0.30,
            derivative_zero_bonus=0.14,
            prominence=0.22,
            rt_distance_per_ms=-0.010,
        )
    if tag == "rising_edge":
        return TCandidateScoreWeights(
            template_landmark_bonus=0.25,
            shoulder_bonus=0.10,
            before_first_pos_per_ms=-0.04,
            rt_distance_per_ms=-0.008,
        )
    if tag == "inverted_t":
        return TCandidateScoreWeights(
            template_landmark_bonus=0.35,
            sign_consistency=0.35,
            prominence=0.30,
            rt_distance_per_ms=-0.015,
        )
    return TCandidateScoreWeights()


def morphology_rescue_landmark_global(
    s_i: int,
    q_next: int,
    tmpl: MedianBeatTemplate,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> Tuple[Optional[int], str]:
    """
    Global-sample T rescue target from morphology-aware template projection.

    Returns (sample_idx, source_tag).
    """
    if not tmpl.valid or tmpl.template.size < 2:
        return None, "none"

    n_tpl = int(tmpl.template.size)
    tag = normalize_t_morphology_tag(tmpl.t_morphology)

    if tag == MORPH_BIPHASIC_POS_NEG:
        mode = getattr(cfg, "record_template_prior_biphasic_rescue_lobe", "negative")
        neg = getattr(tmpl, "t_biphasic_neg_landmark_idx", None)
        pos = getattr(tmpl, "t_biphasic_pos_landmark_idx", None)
        if mode == "positive" and pos is not None:
            return int(_tpl_index_to_sample(s_i, q_next, float(pos), n_tpl)), "biphasic_pos"
        if neg is not None:
            return int(_tpl_index_to_sample(s_i, q_next, float(neg), n_tpl)), "biphasic_neg"
        if pos is not None:
            return int(_tpl_index_to_sample(s_i, q_next, float(pos), n_tpl)), "biphasic_pos"

    if tag == "rising_edge":
        proj = project_t_center_sample(s_i, q_next, tmpl, n_tpl, cfg)
        if proj is not None:
            return int(proj), "rising_edge_projected"

    tpl_idx = _resolve_stpq_t_tpl_idx_for_projection(tmpl, cfg)
    if tpl_idx is not None:
        return int(_tpl_index_to_sample(s_i, q_next, float(tpl_idx), n_tpl)), "template_tpl_idx"

    land = getattr(tmpl, "t_landmark_idx", None)
    if land is not None:
        return int(_tpl_index_to_sample(s_i, q_next, float(land), n_tpl)), "template_landmark"

    proj = project_t_center_sample(s_i, q_next, tmpl, n_tpl, cfg)
    if proj is not None:
        return int(proj), "template_projected"
    return None, "none"


def rescue_candidate_passes_plausibility(
    candidate_rt_ms: float,
    signed_amp: float,
    tmpl: MedianBeatTemplate,
    cfg: ProcessCycleConfig,
    *,
    prominence: float,
    st_baseline: float,
    shallow_dip_inverted: bool = False,
) -> bool:
    rt_min, rt_max = cfg.t_rt_bounds_ms
    if not (rt_min <= candidate_rt_ms <= rt_max):
        return False

    min_prom = float(cfg.record_template_prior_rescue_min_prominence_frac)
    seg_ptp = max(abs(float(np.ptp([signed_amp, st_baseline]))), 1e-9)
    if prominence < min_prom * seg_ptp and abs(signed_amp) < min_prom * seg_ptp:
        return False

    tag = normalize_t_morphology_tag(tmpl.t_morphology)
    want_neg = tag in ("inverted_t",) or str(tmpl.t_polarity) == "negative"
    if tag == MORPH_BIPHASIC_POS_NEG:
        lobe = getattr(cfg, "record_template_prior_biphasic_rescue_lobe", "negative")
        want_neg = lobe == "negative"
    if want_neg and signed_amp > 0 and not shallow_dip_inverted:
        return False
    if not want_neg and signed_amp < 0 and tag != MORPH_BIPHASIC_POS_NEG:
        return False
    return True
