"""Sprint 4: record wavelet P/T coarse priors."""

import numpy as np
import pandas as pd
from dataclasses import replace

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.record_delineation import (
    MedianBeatTemplate,
    _pt_expected_offset,
    apply_record_level_delineation,
)
from pyhearts.processing.record_wavelet_delineation import compute_record_wavelet_pt_priors


def _minimal_tmpl(n_beats: int = 4, rr: float = 200.0) -> MedianBeatTemplate:
    tpl = np.zeros(100)
    return MedianBeatTemplate(
        template=tpl,
        pre_r_samples=40,
        r_center_idx=40,
        p_offset_samples=-80.0,
        t_offset_samples=60.0,
        p_polarity="positive",
        t_polarity="negative",
        median_rr_samples=rr,
        n_beats=n_beats,
        valid=True,
        template_anchor="s_to_q",
        t_landmark_idx=55,
        p_landmark_idx=85,
    )


def _synthetic_ecg(fs: float = 250.0, n_beats: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rr = int(0.8 * fs)
    n = rr * (n_beats + 1)
    t = np.arange(n) / fs
    ecg = 0.05 * np.sin(2 * np.pi * 1.2 * t)
    r_peaks = []
    for i in range(n_beats):
        r = rr * (i + 1)
        r_peaks.append(r)
        ecg[r - 8 : r + 9] += np.hanning(17) * 1.2
        if r + int(0.22 * fs) < n:
            ecg[r + int(0.22 * fs)] += 0.35
        if r - int(0.18 * fs) >= 0:
            ecg[r - int(0.18 * fs)] += 0.2
    return ecg, np.asarray(r_peaks, dtype=int)


def test_pt_expected_offset_prefers_wavelet():
    cfg = ProcessCycleConfig(record_wavelet_pt_prior=True)
    assert _pt_expected_offset(10.0, 12.0, cfg) == 12.0
    cfg_off = ProcessCycleConfig(record_wavelet_pt_prior=False)
    assert _pt_expected_offset(10.0, 12.0, cfg_off) == 10.0


def test_compute_wavelet_priors_disabled():
    ecg, r_peaks = _synthetic_ecg()
    tmpl = _minimal_tmpl(len(r_peaks))
    cfg = ProcessCycleConfig(record_wavelet_pt_prior=False)
    priors = compute_record_wavelet_pt_priors(
        ecg, r_peaks, np.arange(len(r_peaks)), tmpl, 250.0, cfg, 1.0
    )
    assert not priors.valid


def test_compute_wavelet_priors_enabled():
    ecg, r_peaks = _synthetic_ecg()
    cycle_labels = np.arange(len(r_peaks))
    tmpl = _minimal_tmpl(len(r_peaks))
    cfg = ProcessCycleConfig(record_wavelet_pt_prior=True)
    priors = compute_record_wavelet_pt_priors(
        ecg, r_peaks, cycle_labels, tmpl, 250.0, cfg, expected_max_energy=0.5
    )
    assert priors.valid
    assert priors.n_beats >= 1
    assert priors.expected_t_offset(0) is not None


def test_apply_record_delineation_wavelet_stats():
    from tests.test_record_delineation import _synthetic_record

    fs = 250.0
    sig, r_peaks = _synthetic_record(fs=fs, n_beats=12)
    cfg = replace(
        ProcessCycleConfig.for_human_unified(),
        record_wavelet_pt_prior=True,
        record_delineation_min_beats=5,
    )
    pre_r = int(0.4 * fs)
    rows = []
    for i, r in enumerate(r_peaks):
        start = r - pre_r
        for j in range(2 * pre_r):
            rows.append({"index": start + j, "signal_y": sig[start + j], "cycle": i})
    epochs_df = pd.DataFrame(rows)
    cycles = np.arange(len(r_peaks))
    n = len(cycles)
    output_dict = {
        "R_global_center_idx": [float(r) for r in r_peaks],
        "P_global_center_idx": [np.nan] * n,
        "T_global_center_idx": [np.nan] * n,
        "R_center_idx": [float(pre_r)] * n,
        "P_center_idx": [np.nan] * n,
        "T_center_idx": [np.nan] * n,
        "R_center_voltage": [1.0] * n,
        "P_center_voltage": [np.nan] * n,
        "T_center_voltage": [np.nan] * n,
        "R_center_ms": [0.0] * n,
        "P_center_ms": [np.nan] * n,
        "T_center_ms": [np.nan] * n,
    }
    stats = apply_record_level_delineation(
        output_dict,
        sig,
        r_peaks,
        epochs_df,
        cycles,
        fs,
        cfg,
        expected_max_energy=1.0,
    )
    assert stats["template_valid"] == 1
    assert stats.get("wavelet_valid_beats", 0) >= 1
    assert cfg.record_wavelet_pt_prior is True
