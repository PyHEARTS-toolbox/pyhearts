"""Regression: a Gaussian center fitted outside the cycle window must not drop the cycle.

Found on QTDB sel231 (cycle 72): center index 205 in a 204-sample window raised
IndexError inside find_asymmetric_bounds_stdguided, and every component of the
cycle lost its shape features.
"""

import numpy as np
import pytest

from pyhearts._morphology.feature.shape import (
    extract_shape_features,
    find_asymmetric_bounds_stdguided,
)


def _cycle(n=204, fs=250):
    t = np.arange(n)
    p = 0.15 * np.exp(-0.5 * ((t - 40) / 6) ** 2)
    r = 1.0 * np.exp(-0.5 * ((t - 90) / 3) ** 2)
    tw = 0.3 * np.exp(-0.5 * ((t - 160) / 12) ** 2)
    return p + r + tw, fs


def test_out_of_window_center_skips_only_that_component():
    sig, fs = _cycle()
    centers = np.array([40.0, 90.0, 205.0])  # T center one past the last index
    stdevs = np.array([6.0, 3.0, 12.0])
    heights = np.array([0.15, 1.0, 0.3])
    out = extract_shape_features(sig, centers, stdevs, heights, ["P", "R", "T"], 1.0, fs)
    assert "T" not in out["valid_components"]
    assert {"P", "R"} <= set(out["valid_components"])


@pytest.mark.parametrize("center", [-1, 204, 205])
def test_bounds_rejects_out_of_range_center(center):
    sig, fs = _cycle()
    with pytest.raises(ValueError):
        find_asymmetric_bounds_stdguided(sig, center, 0.3, 12.0, sampling_rate=fs)
