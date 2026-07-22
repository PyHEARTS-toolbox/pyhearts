"""HRV from detected R peaks (independent of morphology retention)."""

from __future__ import annotations

import numpy as np

from pyhearts._morphology.feature import calc_hrv_metrics, rr_intervals_ms_from_r_peaks
from pyhearts._morphology.objs.fit import PyHEARTS as MorphologyPyHEARTS


def test_rr_intervals_ms_from_r_peaks_gates_bounds():
    fs = 500.0
    # 1 s, 0.5 s, 2 s gaps → 1000, 500, 2000 ms
    r_peaks = np.array([0, 500, 750, 1750], dtype=int)
    rr = rr_intervals_ms_from_r_peaks(r_peaks, fs, rr_bounds_ms=(300, 1800))
    assert rr.tolist() == [1000.0, 500.0]


def test_rr_intervals_ms_from_r_peaks_empty_when_too_few():
    rr = rr_intervals_ms_from_r_peaks([10], 250.0, rr_bounds_ms=(300, 1800))
    assert rr.size == 0


def test_compute_hrv_metrics_uses_r_peaks_not_morphology_rr():
    fs = 250.0
    # 80 beats at 800 ms RR → 79 intervals (enough for HRV threshold of 60)
    r_peaks = (np.arange(80) * int(0.8 * fs)).astype(int)
    analyzer = MorphologyPyHEARTS(sampling_rate=fs, species="human")
    analyzer.r_peak_indices = r_peaks
    # Poison morphology RR so a regression to the old path would fail / differ
    analyzer.output_dict = {
        "RR_interval_ms": [np.nan] * 5,
    }

    analyzer.compute_hrv_metrics()
    assert analyzer.hrv_metrics
    assert analyzer.hrv_metrics["rr_source"] == "r_peaks"
    assert analyzer.hrv_metrics["n_r_peaks"] == 80
    assert analyzer.hrv_metrics["n_rr_intervals"] == 79
    assert analyzer.hrv_metrics["average_heart_rate"] == 75
    assert analyzer.rr_intervals_ms is not None
    assert len(analyzer.rr_intervals_ms) == 79

    mean_hr, sdnn, rmssd, nn50 = calc_hrv_metrics(analyzer.rr_intervals_ms)
    assert mean_hr == analyzer.hrv_metrics["average_heart_rate"]
    assert sdnn == analyzer.hrv_metrics["sdnn"]
    assert rmssd == analyzer.hrv_metrics["rmssd"]
    assert nn50 == analyzer.hrv_metrics["nn50"]


def test_compute_hrv_metrics_skips_when_fewer_than_60_rr():
    fs = 250.0
    r_peaks = (np.arange(30) * int(0.8 * fs)).astype(int)
    analyzer = MorphologyPyHEARTS(sampling_rate=fs, species="human")
    analyzer.r_peak_indices = r_peaks
    analyzer.compute_hrv_metrics()
    assert analyzer.hrv_metrics == {}
