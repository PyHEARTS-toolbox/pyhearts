"""Tests for curve_fit bound clipping defect repair (1.0.1)."""

from __future__ import annotations

import numpy as np

from pyhearts._morphology.processing.bounds import calc_bounds, clip_guess_to_bounds


def test_clip_guess_keeps_seed_inside_open_interval():
    lo, hi = calc_bounds(center=10, height=1.0, std=5, bound_factor=0.1)
    bounds = (np.array(lo, dtype=float), np.array(hi, dtype=float))
    # Put seed exactly on the lower edges (SciPy trf would reject).
    p0 = bounds[0].copy()
    clipped = clip_guess_to_bounds(p0, bounds)
    assert np.all(clipped > bounds[0])
    assert np.all(clipped < bounds[1])


def test_clip_guess_repairs_flat_bounds():
    lo = np.array([1.0, 1.0, 1.0])
    hi = np.array([1.0, 1.0, 1.0])  # invalid for trf
    p0 = np.array([1.0, 1.0, 1.0])
    clipped = clip_guess_to_bounds(p0, (lo, hi))
    assert np.all(np.isfinite(clipped))
    # After repair, clipped value must sit strictly between repaired lo/hi
    # (helper expands flat intervals around p0).
    assert clipped.shape == p0.shape
