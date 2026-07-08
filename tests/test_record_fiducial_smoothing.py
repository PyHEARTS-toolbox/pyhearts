"""Tests for record-level P/T delay smoothing (step 5)."""

import numpy as np

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.record_fiducial_smoothing import (
    estimate_record_delay_prior,
    smooth_wave_global_indices,
)


class TestRecordDelayPrior:
    def test_prior_valid_with_enough_beats(self):
        fs = 250.0
        n = 20
        r = np.arange(n) * 200.0
        p = r - 30.0 + np.random.default_rng(0).normal(0, 1, n)
        prior = estimate_record_delay_prior(
            r, p, min_beats=5, default_delay_ms=120.0, sampling_rate=fs
        )
        assert prior.valid
        assert abs(prior.median_delay_samples - (-30.0)) < 3.0

    def test_outlier_is_pulled_toward_median(self):
        r = np.array([1000.0, 1200.0, 1400.0, 1600.0, 1800.0])
        p = np.array([970.0, 1170.0, 1370.0, 1500.0, 1770.0])  # last beat outlier
        prior = estimate_record_delay_prior(
            r, p, min_beats=5, default_delay_ms=120.0, sampling_rate=250.0
        )
        out, n_adj = smooth_wave_global_indices(
            r,
            p,
            prior,
            max_deviation_mad=3.0,
            strength=1.0,
        )
        assert n_adj >= 1
        assert abs((out[-1] - r[-1]) - (p[-2] - r[-2])) < abs(p[-1] - p[-2])

    def test_inliers_unchanged(self):
        r = np.arange(10, dtype=float) * 100
        delays = np.full(10, -25.0)
        p = r + delays
        prior = estimate_record_delay_prior(
            r, p, min_beats=5, default_delay_ms=120.0, sampling_rate=250.0
        )
        out, n_adj = smooth_wave_global_indices(
            r, p, prior, max_deviation_mad=3.0, strength=1.0
        )
        assert n_adj == 0
        np.testing.assert_allclose(out, p)

    def test_human_config_enables_smoothing(self):
        cfg = ProcessCycleConfig.for_human_unified()
        assert cfg.record_fiducial_smoothing is True
