"""Tests for human_unified production preset + v3.2.1 alias."""

import pytest

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.pt_detection_mode import p_t_detection_is_record_only


def test_human_unified_v321_production_defaults():
    cfg = ProcessCycleConfig.for_human_unified()
    assert cfg.r_detection_method == "derivative"
    assert cfg.record_delineation is True
    assert cfg.record_template_anchor == "s_to_q"
    assert cfg.record_delineation_stpq_search is True
    assert not p_t_detection_is_record_only(cfg)
    assert cfg.p_use_derivative_validated_method is True
    assert cfg.record_delineation_overwrite_existing_p is False
    assert cfg.record_delineation_overwrite_existing_t is True
    assert cfg.p_t_search_savgol is False
    assert cfg.record_stpq_use_savgol is False
    assert cfg.t_wave_use_record_prior is False
    assert cfg.t_wave_use_secondary_detector is False
    assert cfg.record_delineation_fill_missing_t is False
    assert cfg.record_delineation_template_fallback is True
    assert cfg.record_delineation_refresh_shape is True
    assert cfg.version == "v3.2.1-human-unified"
    assert cfg.record_stpq_t_w1_end_mode == "template_tj_margin"
    assert cfg.record_stpq_t_apex_mode == "threshold"
    assert cfg.record_stpq_t_project_from == "delineated"
    assert cfg.record_refine_t_operator == "derivative_apex"
    assert cfg.record_stpq_t_template_guided is True
    assert cfg.record_template_t_landmark_inverted_peak is True
    assert cfg.record_template_t_morphology_sq_frac == (0.20, 0.60)
    assert cfg.record_stpq_w1_hi_min_sq_frac == 0.40
    assert cfg.record_stpq_w1_hi_pj_margin_sq_frac == 0.15
    assert cfg.record_stpq_t_w1_post_tj_frac == 0.15
    assert cfg.record_stpq_t_template_guided_half_window_ms == 60.0
    assert cfg.record_stpq_t_template_guided_distance_penalty == 0.002


def test_human_unified_v321_alias_matches_production():
    assert ProcessCycleConfig.for_human_unified_v321().version == "v3.2.1-human-unified"
    assert (
        ProcessCycleConfig.for_human_unified_v321().version
        == ProcessCycleConfig.for_human_unified().version
    )


def test_human_unified_v33a_archived_fill_only():
    cfg = ProcessCycleConfig.for_human_unified_v33a()
    assert cfg.version == "v3.3a-v321-human-unified-fill-only"
    assert cfg.record_delineation_fill_missing_t is True
    assert cfg.record_delineation_template_fallback is True
    assert cfg.t_wave_use_record_prior is True
    assert cfg.record_clinical_verify is False


def test_human_unified_requires_record_delineation_for_record_only():
    with pytest.raises(ValueError, match="record_only"):
        ProcessCycleConfig(
            record_delineation=False,
            p_t_detection_method="record_only",
        )
