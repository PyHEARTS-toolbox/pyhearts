"""Tests for Tier B1 record-level median-beat delineation."""

import numpy as np
import pandas as pd
from dataclasses import replace

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.fiducial_provenance import init_fiducial_provenance
from pyhearts.processing.record_delineation import (
    apply_record_level_delineation,
    build_median_beat_template,
    build_record_beat_template,
    delineate_record_template,
)


def _synthetic_record(fs: float = 250.0, n_beats: int = 12) -> tuple[np.ndarray, np.ndarray]:
    rr = int(0.8 * fs)
    length = rr * (n_beats + 2)
    t = np.arange(length) / fs
    sig = 0.05 * np.sin(2 * np.pi * 0.3 * t)
    r_peaks = []
    p_offset = int(-0.12 * fs)
    t_offset = int(0.32 * fs)
    for i in range(1, n_beats + 1):
        r = i * rr
        r_peaks.append(r)
        sig[r + p_offset] += 0.15
        sig[r + t_offset] -= 0.2
        sig[r] += 1.0
    return sig.astype(float), np.asarray(r_peaks, dtype=int)


class TestMedianBeatTemplate:
    def test_template_finds_p_t_offsets(self):
        fs = 250.0
        sig, r_peaks = _synthetic_record(fs=fs, n_beats=12)
        cfg = ProcessCycleConfig()
        raw = build_median_beat_template(sig, r_peaks, fs, cfg)
        assert raw.valid
        tmpl = delineate_record_template(raw, fs, cfg)
        assert tmpl.p_offset_samples is not None
        assert tmpl.t_offset_samples is not None
        assert tmpl.p_offset_samples < 0
        assert tmpl.t_offset_samples > 0

    def test_apply_updates_output_dict(self):
        fs = 250.0
        sig, r_peaks = _synthetic_record(fs=fs, n_beats=12)
        cfg = replace(ProcessCycleConfig(), record_delineation=True, record_delineation_min_beats=5)
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
        # Minimal keys for shape refresh
        for comp in "PQRST":
            for suffix in ("le_idx", "ri_idx", "duration_ms", "fwhm_le_idx"):
                key = f"{comp}_{suffix}"
                output_dict.setdefault(key, [np.nan] * n)
        stats = apply_record_level_delineation(
            output_dict,
            sig,
            r_peaks,
            epochs_df,
            cycles,
            fs,
            cfg,
        )
        assert stats["template_valid"] == 1
        assert stats["p_mapped"] == n
        assert stats["t_mapped"] == n
        p = np.asarray(output_dict["P_global_center_idx"], dtype=float)
        t = np.asarray(output_dict["T_global_center_idx"], dtype=float)
        assert np.isfinite(p).all()
        assert np.isfinite(t).all()
        assert np.median(p - r_peaks.astype(float)) < -0.08 * fs
        assert np.median(t - r_peaks.astype(float)) > 0.25 * fs

    def test_map_all_beats_overwrites_existing_pt(self):
        fs = 250.0
        sig, r_peaks = _synthetic_record(fs=fs, n_beats=10)
        cfg = replace(
            ProcessCycleConfig(),
            record_delineation=True,
            record_delineation_min_beats=5,
            record_template_anchor="r_centered",
            record_delineation_map_all_beats=True,
            record_delineation_overwrite_existing_p=False,
            record_delineation_overwrite_existing_t=False,
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
        wrong_p = float(r_peaks[0] - 0.05 * fs)
        wrong_t = float(r_peaks[0] + 0.10 * fs)
        output_dict = {
            "R_global_center_idx": [float(r) for r in r_peaks],
            "P_global_center_idx": [wrong_p] + [np.nan] * (n - 1),
            "T_global_center_idx": [wrong_t] + [np.nan] * (n - 1),
            "R_center_idx": [float(pre_r)] * n,
            "P_center_idx": [float(pre_r - 0.05 * fs)] + [np.nan] * (n - 1),
            "T_center_idx": [float(pre_r + 0.05 * fs)] + [np.nan] * (n - 1),
            "R_center_voltage": [1.0] * n,
            "P_center_voltage": [0.1] * n,
            "T_center_voltage": [-0.1] * n,
            "R_center_ms": [0.0] * n,
            "P_center_ms": [0.0] * n,
            "T_center_ms": [0.0] * n,
        }
        init_fiducial_provenance(output_dict, n)
        for comp in "PQRST":
            for suffix in ("le_idx", "ri_idx", "duration_ms"):
                output_dict.setdefault(f"{comp}_{suffix}", [np.nan] * n)
        stats = apply_record_level_delineation(
            output_dict, sig, r_peaks, epochs_df, cycles, fs, cfg
        )
        assert stats["p_mapped"] == n
        assert stats["t_mapped"] == n
        assert stats.get("p_mapped_forced", 0) >= 1
        assert stats.get("t_mapped_forced", 0) >= 1
        p_delays = np.asarray(output_dict["P_global_center_idx"], float) - r_peaks.astype(float)
        t_delays = np.asarray(output_dict["T_global_center_idx"], float) - r_peaks.astype(float)
        assert np.nanmedian(p_delays) < -0.08 * fs
        assert np.nanmedian(t_delays) > 0.25 * fs
        assert output_dict["p_source"][0] in (
            "record_stpq",
            "record_template",
            "record_template_fallback",
        )
        assert output_dict["t_source"][0] in (
            "record_stpq",
            "record_template",
            "record_template_fallback",
        )

    def test_human_unified_preset_record_delineation(self):
        cfg = ProcessCycleConfig.for_human_unified()
        assert cfg.record_delineation is True
        assert cfg.record_fiducial_smoothing is True
        assert cfg.record_delineation_refresh_features is True

    def test_refresh_updates_t_shape_fields(self):
        from pyhearts.processing.cycle_feature_refresh import (
            refresh_cycles_after_timing_update,
        )

        fs = 250.0
        sig, r_peaks = _synthetic_record(fs=fs, n_beats=10)
        cfg = replace(
            ProcessCycleConfig(),
            record_delineation_refresh_features=True,
            record_delineation_refresh_shape=True,
            lite_mode=False,
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
            "T_global_center_idx": [float(r) + 0.32 * fs for r in r_peaks],
            "R_center_idx": [float(pre_r)] * n,
            "P_center_idx": [np.nan] * n,
            "T_center_idx": [float(pre_r) + 0.32 * fs for r in r_peaks],
            "R_center_voltage": [1.0] * n,
            "P_center_voltage": [np.nan] * n,
            "T_center_voltage": [-0.2] * n,
            "R_center_ms": [0.0] * n,
            "P_center_ms": [np.nan] * n,
            "T_center_ms": [320.0] * n,
        }
        for comp in "PQRST":
            for suffix in (
                "le_idx",
                "ri_idx",
                "le_ms",
                "ri_ms",
                "global_le_idx",
                "global_ri_idx",
                "duration_ms",
                "rise_ms",
                "decay_ms",
                "rdsm",
                "sharpness",
                "voltage_integral_uv_ms",
            ):
                key = f"{comp}_{suffix}"
                if key not in output_dict:
                    output_dict[key] = [np.nan] * n
        for key in (
            "PR_interval_ms",
            "QT_interval_ms",
            "RR_interval_ms",
            "ST_segment_ms",
        ):
            output_dict[key] = [np.nan] * n

        stats = refresh_cycles_after_timing_update(
            output_dict,
            epochs_df,
            cycles,
            fs,
            cfg,
            modified_cycles=[0, 1],
        )
        assert stats["cycles_refreshed"] == 2
        assert np.isfinite(output_dict["T_global_center_idx"][0])
        assert np.isfinite(output_dict["T_center_voltage"][0])
