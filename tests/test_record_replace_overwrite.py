"""Record P/T replace must honor overwrite_existing_* (s_to_q outlier gate)."""

import numpy as np
from dataclasses import replace

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.record_delineation import (
    MedianBeatTemplate,
    _should_replace_p,
    _should_replace_t,
)


def _tmpl_s_to_q() -> MedianBeatTemplate:
    n = 400
    return MedianBeatTemplate(
        template=np.zeros(n, dtype=float),
        pre_r_samples=100,
        r_center_idx=100,
        p_offset_samples=-30.0,
        t_offset_samples=80.0,
        p_polarity="positive",
        t_polarity="negative",
        median_rr_samples=200.0,
        n_beats=12,
        valid=True,
        template_anchor="s_to_q",
    )


def test_gated_replace_t_false_when_inside_fence():
    cfg = replace(
        ProcessCycleConfig(),
        record_delineation_replace_t=True,
        record_delineation_overwrite_existing_t=False,
        record_template_anchor="s_to_q",
        record_delineation_replace_t_if_outlier=True,
    )
    tmpl = _tmpl_s_to_q()
    existing_t = 900.0 + 80.0  # exactly template RT
    r_g = 900.0
    assert not _should_replace_t(
        cfg,
        existing_t,
        r_g,
        tmpl,
        1.0,
        250.0,
        t_delay_mad=25.0,
    )


def test_overwrite_t_true_despite_finite_per_cycle_and_delay_mad():
    cfg = replace(
        ProcessCycleConfig.for_human_unified(),
        record_template_anchor="s_to_q",
    )
    tmpl = _tmpl_s_to_q()
    existing_t = 1000.0
    r_g = 900.0
    assert _should_replace_t(
        cfg,
        existing_t,
        r_g,
        tmpl,
        1.0,
        250.0,
        t_delay_mad=25.0,
    )


def test_defer_stpq_overwrite_negative_early_peak():
    cfg = replace(
        ProcessCycleConfig.for_human_unified(),
        record_template_anchor="s_to_q",
    )
    tmpl = replace(_tmpl_s_to_q(), t_landmark_source="early_peak", t_morphology="normal")
    existing_t = 1000.0
    r_g = 900.0
    assert not _should_replace_t(
        cfg,
        existing_t,
        r_g,
        tmpl,
        1.0,
        250.0,
        t_delay_mad=25.0,
        manual_ann_ext="q2c",
    )


def test_defer_stpq_overwrite_false_without_q2c():
    cfg = replace(
        ProcessCycleConfig.for_human_unified(),
        record_template_anchor="s_to_q",
    )
    tmpl = replace(_tmpl_s_to_q(), t_landmark_source="early_peak", t_morphology="normal")
    assert _should_replace_t(
        cfg,
        1000.0,
        900.0,
        tmpl,
        1.0,
        250.0,
        t_delay_mad=25.0,
        manual_ann_ext="q1c",
    )


def test_defer_stpq_overwrite_false_for_inverted_t_morphology():
    cfg = replace(
        ProcessCycleConfig.for_human_unified(),
        record_template_anchor="s_to_q",
    )
    tmpl = replace(
        _tmpl_s_to_q(),
        t_landmark_source="early_peak",
        t_morphology="inverted_t",
    )
    assert _should_replace_t(
        cfg,
        1000.0,
        900.0,
        tmpl,
        1.0,
        250.0,
        t_delay_mad=25.0,
    )


def test_overwrite_p_true_despite_finite_per_cycle_and_delay_mad():
    cfg = replace(
        ProcessCycleConfig.for_human_unified(),
        record_template_anchor="s_to_q",
        record_delineation_overwrite_existing_p=True,
    )
    tmpl = _tmpl_s_to_q()
    assert _should_replace_p(
        cfg,
        1000.0,
        900.0,
        tmpl,
        1.0,
        250.0,
        t_delay_mad=25.0,
    )
