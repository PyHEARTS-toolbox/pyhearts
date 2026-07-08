"""Tests for T morphology stratification labels."""

from __future__ import annotations

import numpy as np

from scripts.classify_bad_record_t_components import (
    MORPHOLOGY_NEEDED_DIAGNOSTIC,
    classify_t_failure_mechanism,
    classify_t_morphology_stratified,
    needed_diagnostic_for_morphology,
)


def _synthetic_beat(
    *,
    fs: float = 250.0,
    pre_s: float = 0.25,
    post_s: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    n_pre = int(round(pre_s * fs))
    n_post = int(round(post_s * fs))
    x_ms = (np.arange(n_pre + n_post + 1) - n_pre) * 1000.0 / fs
    y = np.zeros_like(x_ms)
    return x_ms, y


def test_biphasic_pm_morphology():
    x_ms, y = _synthetic_beat()
    baseline = 0.0
    # Early positive then late negative
    pos_i = np.argmin(np.abs(x_ms - 180))
    neg_i = np.argmin(np.abs(x_ms - 320))
    y[pos_i] = 0.5
    y[neg_i] = -0.4
    cands = [
        {"time_ms": 180.0, "signed_amp": 0.5, "abs_amp": 0.5, "prominence": 0.4, "width_ms": 40},
        {"time_ms": 320.0, "signed_amp": -0.4, "abs_amp": 0.4, "prominence": 0.3, "width_ms": 50},
    ]
    assert classify_t_morphology_stratified(x_ms, y, cands, baseline) == "biphasic +-"


def test_needed_diagnostic_mapping():
    assert needed_diagnostic_for_morphology("plateau") == MORPHOLOGY_NEEDED_DIAGNOSTIC["plateau"]
    assert "ST segment" in needed_diagnostic_for_morphology("rising-edge / ST deflection")


def test_failure_mechanism_not_pooled():
    mech = classify_t_failure_mechanism(
        ph_delta_ms=-80.0,
        t_failure_mode="scoring",
        manual_nearest_source="derivative_zero_crossing",
        ph_nearest_source="positive_peak",
        manual_like_not_selected=True,
        manual_like_candidate=True,
        morphology="biphasic +-",
        record_t_landmark_source="plateau_apex",
        t_source="record_stpq",
    )
    assert mech == "template_guided_max_amplitude"

    late = classify_t_failure_mechanism(
        ph_delta_ms=60.0,
        t_failure_mode="scoring",
        manual_nearest_source="positive_peak",
        ph_nearest_source="positive_peak",
        manual_like_not_selected=True,
        manual_like_candidate=False,
        morphology="upright monophasic",
        t_source="record_stpq",
    )
    assert late == "stpq_late_guess"

    rising = classify_t_failure_mechanism(
        ph_delta_ms=-40.0,
        t_failure_mode="scoring",
        manual_nearest_source="derivative_zero_crossing",
        ph_nearest_source="rising_edge_onset",
        manual_like_not_selected=True,
        manual_like_candidate=True,
        morphology="rising-edge / ST deflection",
        record_t_landmark_source="rising_edge",
        t_source="record_stpq",
    )
    assert rising == "rising_edge_inverted_morphology"
