"""
Unit tests for shape feature extraction utilities.

These tests are intentionally lightweight: they validate basic behavior and
output structure without depending on real datasets.
"""

from dataclasses import replace

import numpy as np

from pyhearts.config import ProcessCycleConfig
from pyhearts.feature import extract_shape_features


def test_extract_shape_features_returns_expected_structure(sample_epoch_df, sampling_rate):
    sig = sample_epoch_df["signal"].to_numpy()

    # Centers/stdevs roughly match how `sample_epoch_df` was synthesized in `conftest.py`.
    gauss_centers = np.array([90, 210, 250, 290, 450], dtype=float)  # P,Q,R,S,T
    gauss_stdevs = np.array([20, 10, 10, 10, 40], dtype=float)
    gauss_heights = np.array([0.15, -0.10, 1.00, -0.20, 0.30], dtype=float)
    labels = ["P", "Q", "R", "S", "T"]

    cfg = replace(
        ProcessCycleConfig(),
        duration_min_ms=2,  # keep permissive for synthetic short waves
        shape_interdeflection_pairs=[("R", "S")],
        shape_diff_mode="signed",
    )

    out = extract_shape_features(
        signal=sig,
        gauss_centers=gauss_centers,
        gauss_stdevs=gauss_stdevs,
        gauss_heights=gauss_heights,
        component_labels=labels,
        r_height=1.0,
        sampling_rate=int(sampling_rate),
        cfg=cfg,
        verbose=False,
    )

    assert isinstance(out, dict)
    assert "valid_components" in out
    assert "per_component" in out
    assert "pairwise_differences" in out

    assert isinstance(out["valid_components"], list)
    assert isinstance(out["per_component"], dict)
    assert isinstance(out["pairwise_differences"], dict)

    # Pairwise diffs should always be present for requested pairs (even if NaN).
    assert "R_minus_S_voltage_diff_signed" in out["pairwise_differences"]

    # If at least one component was valid, basic per-component keys should exist.
    if out["per_component"]:
        any_label = next(iter(out["per_component"].keys()))
        feat = out["per_component"][any_label]
        assert "duration_ms" in feat
        assert "le_idx" in feat
        assert "ri_idx" in feat




