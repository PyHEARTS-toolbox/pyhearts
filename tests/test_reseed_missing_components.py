"""Warm-start must not lock later beats out of a wave missed on beat 0."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from pyhearts._morphology.config import ProcessCycleConfig
from pyhearts._morphology.processing.initdict import initialize_output_dict
from pyhearts._morphology.processing.processcycle import process_cycle


PEAK_FEATURES = [
    "global_center_idx",
    "global_le_idx",
    "global_ri_idx",
    "center_ms",
    "le_ms",
    "ri_ms",
    "center_idx",
    "le_idx",
    "ri_idx",
    "gauss_center",
    "gauss_height",
    "gauss_stdev_samples",
    "gauss_stdev_ms",
    "gauss_fwhm_samples",
    "gauss_fwhm_ms",
    "center_voltage",
    "le_voltage",
    "ri_voltage",
    "duration_ms",
    "rise_ms",
    "decay_ms",
    "rdsm",
    "sharpness",
    "voltage_integral_uv_ms",
]
INTERVALS = [
    "PR_interval_ms",
    "PR_segment_ms",
    "QRS_interval_ms",
    "ST_segment_ms",
    "ST_interval_ms",
    "QT_interval_ms",
    "PP_interval_ms",
    "RR_interval_ms",
]


def _synthetic_qrs_cycle(fs: float = 500.0, n: int = 400) -> pd.DataFrame:
    """One cycle with a clearly negative Q trough before R."""
    xs = np.arange(n, dtype=float)
    r = 200.0
    y = np.zeros(n)
    y += 0.18 * np.exp(-((xs - (r - 80.0)) ** 2) / (2 * 10.0**2))
    y -= 0.35 * np.exp(-((xs - (r - 22.0)) ** 2) / (2 * 6.0**2))
    y += 1.20 * np.exp(-((xs - r) ** 2) / (2 * 5.0**2))
    y -= 0.30 * np.exp(-((xs - (r + 18.0)) ** 2) / (2 * 6.0**2))
    y += 0.32 * np.exp(-((xs - (r + 110.0)) ** 2) / (2 * 18.0**2))
    return pd.DataFrame(
        {
            "signal_x": xs / fs,
            "signal_y": y,
            "index": (xs + 1000).astype(int),
            "cycle": 0,
        }
    )


def _empty_output() -> dict:
    return initialize_output_dict(
        cycle_inds=[0],
        components=["P", "Q", "R", "S", "T"],
        peak_features=PEAK_FEATURES,
        intervals=INTERVALS,
        pairwise_differences=[
            "R_minus_S_voltage_diff_signed",
            "R_minus_P_voltage_diff_signed",
            "T_minus_R_voltage_diff_signed",
        ],
    )


PRIOR_WITHOUT_Q = {
    "P": [120.0, 0.18, 10.0],
    "R": [200.0, 1.20, 5.0],
    "S": [218.0, -0.30, 6.0],
    "T": [310.0, 0.32, 18.0],
}


def test_reseed_recovers_q_when_prior_fit_omitted_it():
    fs = 500.0
    one = _synthetic_qrs_cycle(fs)
    cfg = ProcessCycleConfig.for_human()
    assert cfg.reseed_missing_components is True

    output, *_rest, prev = process_cycle(
        one,
        _empty_output(),
        fs,
        0,
        None,
        None,
        previous_gauss_features=PRIOR_WITHOUT_Q,
        expected_max_energy=1.0,
        plot=False,
        verbose=False,
        cfg=cfg,
    )
    assert np.isfinite(output["Q_gauss_height"][0])
    assert output["Q_gauss_height"][0] < 0
    assert "Q" in prev


def test_without_reseed_q_stays_locked_out():
    fs = 500.0
    one = _synthetic_qrs_cycle(fs)
    cfg = replace(ProcessCycleConfig.for_human(), reseed_missing_components=False)

    output, *_rest, prev = process_cycle(
        one,
        _empty_output(),
        fs,
        0,
        None,
        None,
        previous_gauss_features=PRIOR_WITHOUT_Q,
        expected_max_energy=1.0,
        plot=False,
        verbose=False,
        cfg=cfg,
    )
    assert not np.isfinite(output["Q_gauss_height"][0])
    assert "Q" not in prev
