"""Gaussian SD bounds must let a warm-started fit recover a too-narrow wave (standard-2)."""
from __future__ import annotations

from pyhearts._morphology.processing.bounds import calc_bounds
from pyhearts.standard import STANDARD_PIPELINE_VERSION


def test_upper_sd_bound_exceeds_current_sd_for_small_sd():
    for sd in (1.0, 2.0, 3.0, 4.0, 5.0, 10.0):
        lo, hi = calc_bounds(center=100, height=0.1, std=sd, bound_factor=0.2)
        assert lo[2] < sd < hi[2] or (lo[2] == 1.0 and hi[2] > sd)


def test_sd_can_recover_over_beats():
    # A fit that keeps pressing the upper bound must climb back from 3 samples.
    sd = 3.0
    for _ in range(30):
        _, hi = calc_bounds(center=100, height=0.1, std=sd, bound_factor=0.2)
        sd = hi[2]
    assert sd > 50


def test_pipeline_version():
    assert STANDARD_PIPELINE_VERSION.startswith("standard-2")
