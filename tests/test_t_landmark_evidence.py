"""Tests for independent landmark evidence features."""

from __future__ import annotations

import numpy as np
import pytest

from pyhearts.processing.t_candidate_scoring import TBeatCandidateContext, TCandidate
from pyhearts.processing.t_landmark_evidence import (
    FEATURE_NAMES,
    LandmarkEvidence,
    candidate_perturbation_stability,
    compute_landmark_evidence,
    local_t_apex_shape_score,
    neighborhood_rt_consistency,
)


def _synthetic_t_apex(fs: float = 250.0, center: int = 100) -> np.ndarray:
    n = 250
    ecg = np.zeros(n)
    half = int(0.03 * fs)
    for i in range(-half, half + 1):
        idx = center + i
        if 0 <= idx < n:
            ecg[idx] = 1.0 - abs(i) / half
    return ecg


def test_local_shape_high_at_true_apex():
    ecg = _synthetic_t_apex()
    score = local_t_apex_shape_score(ecg, 100, 250.0, 0.0, half_width_ms=30.0)
    off = local_t_apex_shape_score(ecg, 70, 250.0, 0.0, half_width_ms=30.0)
    assert score > 0.5
    assert score > off


def test_neighborhood_flags_outlier_rt():
    neighbors = [242.0, 245.0]
    consistent = neighborhood_rt_consistency(244.0, neighbors)
    outlier = neighborhood_rt_consistency(389.0, neighbors)
    assert consistent == 1.0
    assert outlier < 0.15


def test_neighborhood_preserves_beat_to_beat_variation():
    """242, 245, 248 ms RT all neutral — no pull toward neighbor median."""
    neighbors = [242.0, 245.0]
    for rt in (242.0, 245.0, 248.0):
        assert neighborhood_rt_consistency(rt, neighbors) == 1.0


def test_neighborhood_penalty_zero_in_range():
    ev = LandmarkEvidence(neighborhood_consistency=1.0)
    assert ev.neighborhood_penalty() == 0.0
    ev_out = LandmarkEvidence(neighborhood_consistency=0.1)
    assert ev_out.neighborhood_penalty() == pytest.approx(0.9)


def test_enabled_sum_excludes_neighborhood():
    ev = LandmarkEvidence(local_shape_score=0.8, neighborhood_consistency=0.2, candidate_stability=0.6)
    assert ev.enabled_sum(shape=True, neighborhood=True, stability=True) == pytest.approx(0.7)


def test_neighborhood_nan_without_neighbors():
    assert np.isnan(neighborhood_rt_consistency(240.0, []))


def test_stability_lower_for_knife_edge():
    fs = 250.0
    ecg = _synthetic_t_apex(fs=fs, center=100)
    ctx = TBeatCandidateContext(
        r_idx=50, s_i=60, q_next=150, fs=fs, baseline=0.0, expected_rt_ms=200.0
    )
    a = TCandidate(100, 200.0, "positive_peak", 0.8, 0.5, 30.0, 0.2, False, False, True)
    b = TCandidate(101, 204.0, "derivative_zero_crossing", 0.7, 0.48, 25.0, 0.18, False, False, True)

    def score_a(c):
        return 1.0 if c.sample_idx == 100 else 0.99

    stab_tight = candidate_perturbation_stability(ecg, a, [a, b], ctx, score_a, perturb_ms=5.0)
    stab_clear = candidate_perturbation_stability(
        ecg,
        TCandidate(80, 120.0, "positive_peak", 0.9, 0.6, 30.0, 0.3, False, False, True),
        [a, b],
        ctx,
        lambda c: 2.0 if c.sample_idx == 80 else 0.5,
        perturb_ms=5.0,
    )
    assert stab_clear >= stab_tight


def test_feature_names_for_learning_export():
    assert len(FEATURE_NAMES) == 3
    ev = LandmarkEvidence(0.8, 0.9, 0.7)
    assert ev.enabled_sum(shape=True, neighborhood=False, stability=True) == pytest.approx(0.75)
