"""Regression tests for the morphology + record-level T pipeline."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pyhearts import PyHEARTS, __version__
from pyhearts.config import ProcessCycleConfig
from pyhearts._morphology.config import ProcessCycleConfig as CoreProcessCycleConfig
from pyhearts._morphology.processing import gaussian as core_gaussian
from pyhearts.core.analyzer import PIPELINE_VERSION, merge_record_t


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


def test_merge_record_t_changes_only_t_fiducial_columns():
    features = pd.DataFrame(
        {
            "P_global_center_idx": [80.0, 180.0],
            "R_global_center_idx": [100.0, 200.0],
            "T_global_center_idx": [130.0, 230.0],
            "r_squared": [0.97, 0.96],
        }
    )
    pairs = np.array([[100.0, 140.0], [200.0, 242.0]])

    output = merge_record_t(features, pairs)

    assert output["P_global_center_idx"].equals(features["P_global_center_idx"])
    assert output["R_global_center_idx"].equals(features["R_global_center_idx"])
    assert output["r_squared"].equals(features["r_squared"])
    assert output["T_gaussian_global_center_idx"].tolist() == [130.0, 230.0]
    assert output["T_global_center_idx"].tolist() == [140.0, 242.0]
    assert output["t_source"].tolist() == [
        "record_t",
        "record_t",
    ]


def test_merge_record_t_gaussian_fill_and_later_preference():
    features = pd.DataFrame(
        {
            "R_global_center_idx": [100.0, 200.0, 300.0],
            # Gaussian apices (pre-merge T)
            "T_global_center_idx": [160.0, 250.0, 360.0],
        }
    )
    # record-T: miss on beat0, early on beat1, aligned on beat2
    pairs = np.array([[100.0, np.nan], [200.0, 220.0], [300.0, 358.0]])

    output = merge_record_t(
        features,
        pairs,
        sampling_rate=500.0,
        fallback_gaussian_on_miss=True,
        prefer_later_gaussian_ms=20.0,
    )

    # beat0: fill miss with gaussian 160
    assert output["T_global_center_idx"].tolist()[0] == 160.0
    assert output["t_source"].tolist()[0] == "gaussian_fill"
    # beat1: gaussian 250 is 60ms later than record 220 at 500Hz → prefer gaussian
    assert output["T_global_center_idx"].tolist()[1] == 250.0
    assert output["t_source"].tolist()[1] == "gaussian_later"
    # beat2: within 20ms → keep record
    assert output["T_global_center_idx"].tolist()[2] == 358.0
    assert output["t_source"].tolist()[2] == "record_t"
    assert output["T_gaussian_global_center_idx"].tolist() == [160.0, 250.0, 360.0]


def test_analyze_ecg_runs_record_t_at_end():
    analyzer = PyHEARTS(sampling_rate=500.0, species="human")
    output, epochs = analyzer.analyze_ecg(_deterministic_ecg())

    assert len(output) == 9
    assert not epochs.empty
    assert output["R_global_center_idx"].notna().sum() == 9
    assert output["T_global_center_idx"].notna().sum() == 9
    assert (output["t_source"] == "record_t").sum() >= 8
    assert analyzer.last_record_t_stats["template_valid"] == 1
    assert analyzer.pipeline_version == PIPELINE_VERSION
    assert analyzer.cfg.version == "v1-fitbounds-clip-human"
    assert not hasattr(analyzer, "core_cfg")
    assert not hasattr(analyzer, "t_cfg")
    assert hasattr(analyzer, "_core_cfg")
    assert hasattr(analyzer, "_t_cfg")


def test_mouse_skips_record_t_by_default():
    analyzer = PyHEARTS(sampling_rate=2000.0, species="mouse")
    assert analyzer.apply_record_t is False
    assert "mouse" in analyzer.cfg.version
    assert analyzer.cfg.t_reseed_if_missing is True
    assert analyzer.cfg.t_ignore_wavelet_guard is True
    assert analyzer.cfg.t_height_above_baseline is True
    assert analyzer.cfg.t_end_margin_ms == 5.0


def test_human_t_search_defaults_unchanged():
    analyzer = PyHEARTS(sampling_rate=500.0, species="human")
    assert analyzer.apply_record_t is True
    assert analyzer.cfg.t_reseed_if_missing is False
    assert analyzer.cfg.reseed_missing_components is True
    assert analyzer.cfg.t_ignore_wavelet_guard is False
    assert analyzer.cfg.t_height_above_baseline is False
    assert analyzer.cfg.t_end_margin_ms is None


def test_symmetric_gaussian_is_the_only_core_fit_option():
    assert not hasattr(CoreProcessCycleConfig(), "use_skewed_gaussian")
    assert not hasattr(ProcessCycleConfig(), "use_skewed_gaussian")
    assert not hasattr(core_gaussian, "skewed_gaussian_function")


def test_save_output_keeps_dual_configs_in_metadata(tmp_path):
    analyzer = PyHEARTS(sampling_rate=500.0, species="human")
    analyzer.analyze_ecg(_deterministic_ecg())

    output_path = analyzer.save_output("synthetic", str(tmp_path))
    metadata = json.loads((tmp_path / "synthetic_meta.json").read_text())

    assert output_path == tmp_path / "synthetic_pyhearts.csv"
    assert output_path.exists()
    assert metadata["pyhearts_version"] == __version__
    assert metadata["pipeline_version"] == PIPELINE_VERSION
    assert metadata["pipeline"]["t_detector"] == "record-t"
    assert "core_config" in metadata
    assert "t_config" in metadata
