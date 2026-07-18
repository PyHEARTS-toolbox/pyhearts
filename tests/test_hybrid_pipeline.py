"""Regression tests for the frozen 2025 + STPQ T hybrid."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pyhearts import CoreProcessCycleConfig, ProcessCycleConfig, PyHEARTS
from pyhearts._legacy2025.processing import gaussian as core_gaussian
from pyhearts.core.hybrid import (
    HYBRID_CONFIG_VERSION,
    merge_stpq_t,
)


def _deterministic_ecg(fs: float = 500.0, duration: float = 10.0) -> np.ndarray:
    times = np.linspace(0.0, duration, int(fs * duration))
    signal = np.zeros_like(times)
    for r_time in np.arange(0.5, duration - 0.5, 1.0):
        signal += 0.15 * np.exp(-((times - (r_time - 0.16)) ** 2) / (2 * 0.02**2))
        signal -= 0.10 * np.exp(-((times - (r_time - 0.04)) ** 2) / (2 * 0.01**2))
        signal += 1.00 * np.exp(-((times - r_time) ** 2) / (2 * 0.01**2))
        signal -= 0.20 * np.exp(-((times - (r_time + 0.04)) ** 2) / (2 * 0.01**2))
        signal += 0.30 * np.exp(-((times - (r_time + 0.25)) ** 2) / (2 * 0.04**2))
    return signal


def test_merge_stpq_t_changes_only_t_fiducial_columns():
    features = pd.DataFrame(
        {
            "P_global_center_idx": [80.0, 180.0],
            "R_global_center_idx": [100.0, 200.0],
            "T_global_center_idx": [130.0, 230.0],
            "r_squared": [0.97, 0.96],
        }
    )
    pairs = np.array([[100.0, 140.0], [200.0, 242.0]])

    output = merge_stpq_t(features, pairs)

    assert output["P_global_center_idx"].equals(features["P_global_center_idx"])
    assert output["R_global_center_idx"].equals(features["R_global_center_idx"])
    assert output["r_squared"].equals(features["r_squared"])
    assert output["T_gaussian_global_center_idx"].tolist() == [130.0, 230.0]
    assert output["T_global_center_idx"].tolist() == [140.0, 242.0]
    assert output["t_source"].tolist() == [
        "record_stpq_hybrid",
        "record_stpq_hybrid",
    ]


def test_hybrid_end_to_end_detects_record_stpq_t():
    analyzer = PyHEARTS(sampling_rate=500.0, species="human")
    output, epochs = analyzer.analyze_ecg(_deterministic_ecg())

    assert len(output) == 9
    assert not epochs.empty
    assert output["R_global_center_idx"].notna().sum() == 9
    assert output["T_global_center_idx"].notna().sum() == 9
    assert (output["t_source"] == "record_stpq_hybrid").sum() >= 8
    assert analyzer.last_stpq_stats["template_valid"] == 1
    assert analyzer.cfg.version == HYBRID_CONFIG_VERSION


def test_mouse_core_skips_human_stpq_by_default():
    analyzer = PyHEARTS(sampling_rate=2000.0, species="mouse")
    assert analyzer.apply_stpq_t is False
    assert analyzer.core_cfg.version == "v1-mouse"


def test_symmetric_gaussian_is_the_only_core_fit_option():
    assert not hasattr(CoreProcessCycleConfig(), "use_skewed_gaussian")
    assert not hasattr(ProcessCycleConfig(), "use_skewed_gaussian")
    assert not hasattr(core_gaussian, "skewed_gaussian_function")


def test_save_output_includes_hybrid_metadata(tmp_path):
    analyzer = PyHEARTS(sampling_rate=500.0, species="human")
    analyzer.analyze_ecg(_deterministic_ecg())

    output_path = analyzer.save_output("synthetic", str(tmp_path))
    metadata = json.loads((tmp_path / "synthetic_meta.json").read_text())

    assert output_path == tmp_path / "synthetic_pyhearts.csv"
    assert output_path.exists()
    assert metadata["pyhearts_version"] == HYBRID_CONFIG_VERSION
    assert metadata["pipeline"]["core"] == "pyhearts-2025-symmetric-gaussian"
    assert metadata["pipeline"]["t_detector"] == "human-unified-record-stpq"
