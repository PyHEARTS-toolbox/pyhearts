"""Tests for T candidate generation and scoring (diagnostic module)."""

from __future__ import annotations

import numpy as np

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.record_delineation import MedianBeatTemplate
from pyhearts.processing.t_candidate_scoring import (
    generate_t_candidates,
    pick_best_t_candidate,
    score_t_candidates,
    TBeatCandidateContext,
)


def _synthetic_biphasic_beat(fs: float = 250.0) -> tuple:
    n = 600
    ecg = np.zeros(n, dtype=float)
    r_idx = 150
    s_i = 170
    q_next = 450
    t_pos = 250
    t_neg = 340
    ecg[t_pos] = 0.5
    ecg[t_neg] = -0.35
    ecg[s_i:q_next] += np.linspace(-0.05, 0.02, q_next - s_i)
    tmpl = MedianBeatTemplate(
        template=np.zeros(200),
        pre_r_samples=50,
        r_center_idx=0,
        p_offset_samples=None,
        t_offset_samples=80.0,
        p_polarity="positive",
        t_polarity="positive",
        median_rr_samples=200,
        n_beats=5,
        valid=True,
        template_anchor="s_to_q",
        t_landmark_idx=80.0,
        p_landmark_idx=150.0,
        t_morphology="biphasic_positive_negative",
        t_biphasic_pos_landmark_idx=80.0,
        t_biphasic_neg_landmark_idx=120.0,
    )
    ctx = TBeatCandidateContext(
        r_idx=r_idx,
        s_i=s_i,
        q_next=q_next,
        fs=fs,
        baseline=0.0,
        expected_rt_ms=200.0,
        first_pos_rt_ms=160.0,
    )
    return ecg, r_idx, s_i, q_next, tmpl, ctx


class TestTCandidateScoring:
    def test_generates_multiple_sources(self):
        ecg, r_idx, s_i, q_next, tmpl, ctx = _synthetic_biphasic_beat()
        cfg = ProcessCycleConfig.for_human_unified()
        cands, _ = generate_t_candidates(
            ecg, r_idx, s_i, q_next, 250.0, tmpl=tmpl, cfg=cfg, ctx=ctx
        )
        sources = {c.source for c in cands}
        assert "positive_peak" in sources or "negative_peak" in sources
        assert len(cands) >= 2

    def test_generates_refined_geometry_sources(self):
        ecg, r_idx, s_i, q_next, tmpl, ctx = _synthetic_biphasic_beat()
        # Make a small plateau around the positive peak so amplitude midpoint is defined.
        ecg[248:253] = 0.48
        cands, _ = generate_t_candidates(ecg, r_idx, s_i, q_next, 250.0, tmpl=tmpl, ctx=ctx)
        sources = {c.source.split("+")[0] for c in cands}
        assert "refined_apex" in sources
        assert "refined_dzc" in sources or "derivative_zero_crossing" in sources
        assert "amplitude_plateau_midpoint" in sources or "plateau_midpoint" in sources

    def test_score_orders_candidates(self):
        ecg, r_idx, s_i, q_next, tmpl, ctx = _synthetic_biphasic_beat()
        cands, ctx = generate_t_candidates(ecg, r_idx, s_i, q_next, 250.0, tmpl=tmpl, ctx=ctx)
        scored = score_t_candidates(cands, ctx)
        assert scored[0].score >= scored[-1].score

    def test_pick_best_returns_candidate(self):
        ecg, r_idx, s_i, q_next, tmpl, ctx = _synthetic_biphasic_beat()
        cands, ctx = generate_t_candidates(ecg, r_idx, s_i, q_next, 250.0, tmpl=tmpl, ctx=ctx)
        best = pick_best_t_candidate(cands, ctx)
        assert best is not None
        assert np.isfinite(best.score)
