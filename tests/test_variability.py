"""
Unit tests for beat-to-beat variability metrics.
"""

import numpy as np

from pyhearts.feature import compute_beat_to_beat_variability, compute_variability_metrics


def test_compute_variability_metrics_basic_values():
    xs = np.array([1.0, 2.0, 3.0, 4.0])
    out = compute_variability_metrics(xs, feature_name="x")

    assert isinstance(out, dict)
    assert set(out.keys()) == {"x_std", "x_cv", "x_iqr", "x_mad", "x_range"}
    assert out["x_std"] > 0
    assert out["x_range"] == 3.0


def test_compute_variability_metrics_handles_nan_and_insufficient_data():
    xs = np.array([np.nan, 1.0, np.nan])
    out = compute_variability_metrics(xs, feature_name="x")
    assert np.isnan(out["x_std"])
    assert np.isnan(out["x_cv"])


def test_compute_beat_to_beat_variability_respects_priority_features():
    output_dict = {
        "QT_interval_ms": [300.0, 320.0, 310.0, np.nan],
        "not_requested": [1, 2, 3],
    }

    out = compute_beat_to_beat_variability(output_dict, priority_features=["QT_interval_ms"])
    assert "QT_interval_ms_std" in out
    assert "QT_interval_ms_range" in out
    assert "not_requested_std" not in out


