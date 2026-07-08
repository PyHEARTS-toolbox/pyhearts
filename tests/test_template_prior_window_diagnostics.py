"""Tests for template-prior window diagnostics and uncertainty projection."""

from __future__ import annotations

import numpy as np

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.record_delineation import finalize_stpq_median_template
from pyhearts.processing.template_prior_window_diagnostics import (
    classify_bad_window_cause,
    t_uncertainty_window_samples,
)


def test_classify_window_width_when_landmark_matches_manual():
    cause = classify_bad_window_cause(
        manual_t=1000,
        landmark=1002,
        t_lo=990,
        t_hi=998,
        r_idx=900,
        fs=250.0,
        beat_template_corr=0.9,
        rr_stretch_ratio=1.0,
    )
    assert cause == "window_width_wrong"


def test_classify_landmark_too_early():
    cause = classify_bad_window_cause(
        manual_t=1100,
        landmark=1040,
        t_lo=1000,
        t_hi=1080,
        r_idx=900,
        fs=250.0,
        beat_template_corr=0.9,
        rr_stretch_ratio=1.0,
    )
    assert cause == "landmark_too_early"


def test_uncertainty_window_centers_on_landmark():
    cfg = ProcessCycleConfig.for_human_unified_template_prior_phase1_uncertainty()
    landmark = 500
    t_lo, t_hi = t_uncertainty_window_samples(
        landmark,
        sigma_ms=28.0,
        ecg_len=2000,
        s_i=400,
        q_next=700,
        r_idx=350,
        fs=250.0,
        cfg=cfg,
        sigma_mult=2.0,
        min_half_width_ms=40.0,
    )
    assert t_lo <= landmark <= t_hi
    assert (t_hi - t_lo) / 250.0 * 1000.0 >= 80.0


def test_estimate_sigma_positive_on_synthetic_record():
    from pyhearts.processing.template_prior_window_diagnostics import (
        estimate_record_t_timing_sigma_ms,
    )

    fs = 250.0
    n = 4000
    ecg = np.zeros(n)
    r_peaks = np.arange(400, 3600, 200, dtype=int)
    for r in r_peaks:
        ecg[r] = 2.0
        t = r + 60 + int((r // 200) % 5)
        if t < n:
            ecg[t] = -0.5

    cfg = ProcessCycleConfig.for_human_unified_template_prior_phase1()
    template = np.zeros(40)
    template[25:32] = np.linspace(0, -1, 7)
    tmpl = finalize_stpq_median_template(
        template, cfg, fs, pre_r_samples=50, median_rr_samples=200, n_beats=10
    )
    sigma = estimate_record_t_timing_sigma_ms(ecg, r_peaks, tmpl, fs, cfg)
    assert sigma >= 8.0
