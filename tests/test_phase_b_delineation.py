"""Tests for Phase B P/T delineation signal and STPQ template."""

import numpy as np
import pytest
from dataclasses import replace

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.delineation_signal import (
    median_baseline_removal,
    prepare_record_delineation_signal,
    savgol_search_segment,
)
from pyhearts.processing.record_delineation import (
    build_record_beat_template,
    build_stpq_beat_template,
    delineate_record_template,
)


def _synthetic_stpq_record(fs: float = 250.0, n_beats: int = 15) -> tuple[np.ndarray, np.ndarray]:
    rr = int(0.8 * fs)
    length = rr * (n_beats + 2)
    sig = np.zeros(length, dtype=float)
    r_peaks = []
    for i in range(1, n_beats + 1):
        r = i * rr
        r_peaks.append(r)
        sig[r] += 1.2
        sig[r + int(0.04 * fs)] -= 0.35
        sig[r + int(0.28 * fs)] -= 0.25
        sig[r + int(-0.14 * fs)] += 0.12
    return sig, np.asarray(r_peaks, dtype=int)


class TestDelineationSignal:
    def test_median_baseline_removes_slow_trend(self):
        fs = 250.0
        n = int(5 * fs)
        t = np.arange(n) / fs
        ecg = 0.5 * np.sin(2 * np.pi * 0.2 * t) + 0.02 * np.random.default_rng(0).standard_normal(n)
        out = median_baseline_removal(ecg, fs, 1.0, 2.0)
        assert out.shape == ecg.shape
        assert np.std(out) < np.std(ecg)

    def test_prepare_record_signal_median_mode(self):
        fs = 250.0
        ecg = np.sin(np.linspace(0, 4 * np.pi, int(2 * fs)))
        cfg = ProcessCycleConfig(delineation_baseline_method="median_record")
        out = prepare_record_delineation_signal(ecg, fs, cfg)
        assert out.shape == ecg.shape

    def test_bandpass_delineation_differs_from_median_only(self):
        fs = 250.0
        rng = np.random.default_rng(2)
        ecg = np.sin(np.linspace(0, 20 * np.pi, int(4 * fs))) + 0.05 * rng.standard_normal(int(4 * fs))
        base_cfg = ProcessCycleConfig(
            delineation_baseline_method="median_record",
            delineation_bandpass=False,
        )
        bp_cfg = replace(base_cfg, delineation_bandpass=True)
        out_base = prepare_record_delineation_signal(ecg, fs, base_cfg)
        out_bp = prepare_record_delineation_signal(ecg, fs, bp_cfg)
        assert not np.allclose(out_base, out_bp)

    def test_savgol_preserves_length(self):
        seg = np.random.default_rng(1).standard_normal(80)
        cfg = ProcessCycleConfig(p_t_search_savgol=True, p_t_savgol_window_ms=25.0)
        out = savgol_search_segment(seg, 250.0, cfg)
        assert len(out) == len(seg)


class TestStpqTemplate:
    def test_stpq_template_builds_and_delineates(self):
        fs = 250.0
        sig, r_peaks = _synthetic_stpq_record(fs=fs)
        cfg = replace(
            ProcessCycleConfig(),
            record_template_anchor="s_to_q",
            record_delineation_min_beats=5,
            delineation_baseline_method="median_record",
            p_t_threshold_mode="template",
        )
        raw = build_stpq_beat_template(sig, r_peaks, fs, cfg)
        assert raw.valid
        assert raw.template_anchor == "s_to_q"
        assert raw.t_landmark_idx is not None
        tmpl = delineate_record_template(raw, fs, cfg)
        assert tmpl.t_landmark_source in (
            "early_peak",
            "plateau_apex",
            "rising_edge",
            "isoelectric",
        )

    def test_build_record_dispatch(self):
        fs = 250.0
        sig, r_peaks = _synthetic_stpq_record(fs=fs, n_beats=12)
        cfg_r = ProcessCycleConfig(record_template_anchor="r_centered")
        cfg_s = replace(cfg_r, record_template_anchor="s_to_q", record_delineation_min_beats=5)
        t_r = build_record_beat_template(sig, r_peaks, fs, cfg_r)
        t_s = build_record_beat_template(sig, r_peaks, fs, cfg_s)
        assert t_r.template_anchor == "r_centered"
        assert t_s.template_anchor == "s_to_q"

    def test_human_unified_production_preset(self):
        cfg = ProcessCycleConfig.for_human_unified()
        assert cfg.version == "human-unified"
        assert cfg.record_delineation is True
        assert cfg.record_template_anchor == "s_to_q"
        assert cfg.p_t_threshold_mode == "template"
        assert cfg.delineation_baseline_method == "median_record"
