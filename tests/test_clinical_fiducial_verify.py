"""Sprint 3: clinical second-pass verification."""

import numpy as np
import pandas as pd
from dataclasses import replace

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.clinical_fiducial_verify import (
    _should_clinical_verify_t,
    apply_clinical_fiducial_verification,
)
from pyhearts.processing.fiducial_provenance import init_fiducial_provenance
from pyhearts.processing.record_delineation import apply_record_level_delineation


def _cfg_with_clinical_verify(**kwargs) -> ProcessCycleConfig:
    return replace(
        ProcessCycleConfig.for_human_unified(),
        record_clinical_verify=True,
        clinical_verify_signal="clinical",
        clinical_verify_p_window_ms=40.0,
        clinical_verify_t_operator="derivative_zc",
        clinical_verify_p_operator="derivative_apex",
        clinical_verify_refresh_features=True,
        record_delineation_refine_always=True,
        record_delineation_refine_adaptive=True,
        **kwargs,
    )


def _cfg_conditional_clinical_verify(**kwargs) -> ProcessCycleConfig:
    return replace(
        _cfg_with_clinical_verify(),
        record_delineation_fill_missing_t=True,
        clinical_verify_t_conditional=True,
        **kwargs,
    )


def _synthetic_record(fs: float = 250.0, n_beats: int = 10):
    rr = int(0.8 * fs)
    length = rr * (n_beats + 2)
    sig = 0.05 * np.sin(2 * np.pi * 0.3 * np.arange(length) / fs)
    r_peaks = []
    for i in range(1, n_beats + 1):
        r = i * rr
        r_peaks.append(r)
        sig[r - int(0.12 * fs)] += 0.15
        sig[r + int(0.32 * fs)] -= 0.2
        sig[r] += 1.0
    return sig.astype(float), np.asarray(r_peaks, dtype=int)


def test_clinical_verify_updates_t_source():
    fs = 250.0
    sig, r_peaks = _synthetic_record(n_beats=8)
    cfg = replace(
        _cfg_with_clinical_verify(),
        record_delineation_min_beats=5,
        record_template_anchor="r_centered",
        record_delineation_stpq_search=False,
        p_t_threshold_mode="legacy",
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
    init_fiducial_provenance(output_dict, n)
    apply_record_level_delineation(
        output_dict, sig, r_peaks, epochs_df, cycles, fs, cfg, clinical_ecg=sig
    )
    stats = apply_clinical_fiducial_verification(
        output_dict,
        epochs_df,
        cycles,
        sig,
        r_peaks,
        fs,
        cfg,
    )
    assert stats["p_checked"] == n
    assert stats["t_checked"] == n
    sources = [str(s) for s in output_dict["p_source"] + output_dict["t_source"] if s]
    assert any("clinical" in s for s in sources)


def test_clinical_verify_enabled_on_experimental_cfg():
    cfg = _cfg_with_clinical_verify()
    assert cfg.record_clinical_verify is True
    assert cfg.clinical_verify_t_operator == "derivative_zc"


def test_conditional_clinical_skips_plausible_t():
    cfg = _cfg_conditional_clinical_verify()
    fs = 250.0
    r_g = 1000.0
    t_g = r_g + 0.32 * fs
    assert not _should_clinical_verify_t(
        t_g, r_g, "per_cycle", cfg, fs, median_rt_ms=320.0, mad_rt_ms=20.0
    )
    assert _should_clinical_verify_t(
        t_g + 80, r_g, "per_cycle", cfg, fs, median_rt_ms=320.0, mad_rt_ms=20.0
    )
    assert _should_clinical_verify_t(
        t_g, r_g, "record_fill_missing_template", cfg, fs, 320.0, 20.0
    )


def test_v31_2_conditional_clinical_skips_most_beats():
    fs = 250.0
    sig, r_peaks = _synthetic_record(n_beats=8)
    cfg = _cfg_conditional_clinical_verify()
    cfg = replace(
        cfg,
        record_delineation_min_beats=5,
        record_template_anchor="r_centered",
        record_delineation_stpq_search=False,
        p_t_threshold_mode="legacy",
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
    init_fiducial_provenance(output_dict, n)
    apply_record_level_delineation(
        output_dict, sig, r_peaks, epochs_df, cycles, fs, cfg, clinical_ecg=sig
    )
    stats = apply_clinical_fiducial_verification(
        output_dict, epochs_df, cycles, sig, r_peaks, fs, cfg
    )
    assert stats["t_skipped_conditional"] >= 1
    assert stats["t_checked"] <= n
