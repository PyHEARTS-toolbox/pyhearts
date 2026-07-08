"""Phase 2: rescue, morphology routing, clustering helpers."""

from __future__ import annotations

import numpy as np

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.record_beat_clustering import stpq_clusters_are_heterogeneous
from pyhearts.processing.record_delineation import MedianBeatTemplate, finalize_stpq_median_template
from pyhearts.processing.template_prior_rescue import rescue_t_for_beat
from pyhearts.processing.template_prior_windows import TemplatePriorBeatWindows


def _cfg_phase2() -> ProcessCycleConfig:
    return ProcessCycleConfig.for_human_unified_template_prior_phase2()


def test_finalize_stpq_median_template_sets_landmarks():
    n = 40
    template = np.zeros(n, dtype=float)
    template[20:30] = np.linspace(0, 1, 10)
    template[30:38] = np.linspace(1, -0.8, 8)
    cfg = _cfg_phase2()
    tmpl = finalize_stpq_median_template(
        template,
        cfg,
        250.0,
        pre_r_samples=50,
        median_rr_samples=200,
        n_beats=10,
    )
    assert tmpl.valid
    assert tmpl.t_landmark_idx is not None
    assert tmpl.p_landmark_idx is not None


def test_stpq_clusters_heterogeneous_requires_enough_beats():
    seg = np.sin(np.linspace(0, np.pi, 32))
    segments = [(i, seg.copy(), i * 40, i * 40 + 32) for i in range(3)]
    assert stpq_clusters_are_heterogeneous(segments, 2) is False


def test_stpq_clusters_not_heterogeneous_for_identical_segments():
    seg = np.sin(np.linspace(0, np.pi, 32))
    segments = [(i, seg.copy(), i * 40, i * 40 + 32) for i in range(6)]
    assert stpq_clusters_are_heterogeneous(segments, 2) is False


def test_rescue_keeps_per_cycle_when_closer_to_landmark_on_dispute():
    cfg = _cfg_phase2()
    n = 120
    ecg = np.zeros(n, dtype=float)
    ecg[70] = -1.0
    ecg[90] = -0.5
    tmpl = finalize_stpq_median_template(
        ecg,
        cfg,
        250.0,
        pre_r_samples=20,
        median_rr_samples=80,
        n_beats=5,
    )
    prior = TemplatePriorBeatWindows(t_lo=50, t_hi=100, s_i=40, q_next=110)
    per_cycle_t = 70.0
    rescued, dec = rescue_t_for_beat(
        ecg,
        r_idx=30,
        prior=prior,
        tmpl=tmpl,
        per_cycle_t=per_cycle_t,
        sampling_rate=250.0,
        cfg=cfg,
    )
    assert dec.trigger in ("ok", "dispute")
    if dec.trigger == "dispute":
        assert dec.applied is False
        assert dec.reason == "kept_closer_per_cycle"
        assert rescued == 70
