"""Gaussian height cap: warm-start height bounds must not ratchet beyond the cycle's amplitude scale."""
import numpy as np
from pyhearts._morphology.config import ProcessCycleConfig
from pyhearts._morphology.processing.bounds import calc_bounds


def test_default_cap_factor():
    assert ProcessCycleConfig().max_height_factor == 1.5


def test_cap_leaves_normal_bounds_unchanged():
    assert calc_bounds(100, 1.0, 10, 0.2) == calc_bounds(100, 1.0, 10, 0.2, height_cap=1.5)


def test_cap_clamps_drifted_heights():
    lo, hi = calc_bounds(100, 211.0, 10, 0.2, height_cap=1.5)
    assert lo[1] < hi[1] <= 1.5
    lo, hi = calc_bounds(100, -210.0, 10, 0.2, height_cap=1.5)
    assert -1.5 <= lo[1] < hi[1] < 0


def test_ratchet_is_bounded():
    h_free = h_cap = 1.0
    for _ in range(30):  # each warm-started beat may take the previous upper bound
        h_free = calc_bounds(100, h_free, 10, 0.2)[1][1]
        h_cap = calc_bounds(100, h_cap, 10, 0.2, height_cap=1.5)[1][1]
    assert h_free > 100  # the bug: unbounded geometric growth
    assert h_cap <= 1.5
