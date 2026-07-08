"""Tests for R-peak Phase A post-processing."""

import numpy as np
import pytest

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.rpeak import (
    _enforce_minimum_peak_interval,
    _local_kurtosis,
    _postprocess_r_peaks_phase_a,
    _refine_r_peaks_trp,
    _reject_low_kurtosis_sharpest_peak,
    _reject_low_kurtosis_r_peaks,
    _should_apply_kurtosis_filter,
    r_peak_detection,
    r_peak_detection_funnel_stats,
)


class TestPhaseAPostProcessing:
    def test_enforce_minimum_interval_keeps_sharper_peak(self):
        fs = 250.0
        n = 500
        t = np.arange(n) / fs
        ecg = np.zeros(n)
        # Sharp R-like deflection
        ecg[100] = 2.0
        ecg[99] = ecg[101] = -0.5
        # Broad bump 120 ms later (T-like)
        for i in range(130, 145):
            ecg[i] = 0.4 * np.sin(np.pi * (i - 130) / 14)

        peaks = np.array([100, 130], dtype=int)
        min_gap = int(0.4 * fs)  # 400 ms
        out = _enforce_minimum_peak_interval(peaks, ecg, min_gap, 20)
        assert out.size == 1
        assert out[0] == 100

    def test_kurtosis_rejection_drops_broad_peak(self):
        fs = 250.0
        n = 2000
        ecg = np.zeros(n)
        r_indices = [200, 450, 700, 950]
        for r in r_indices:
            ecg[r] = 2.0
            ecg[r - 1] = ecg[r + 1] = -0.8
        # Broad low-kurtosis false peak between beats
        false_r = 820
        for i in range(false_r - 15, false_r + 16):
            ecg[i] = 1.2 * np.exp(-0.5 * ((i - false_r) / 10.0) ** 2)

        peaks = np.array(r_indices + [false_r], dtype=int)
        cfg = ProcessCycleConfig(
            r_kurtosis_rejection=True,
            r_kurtosis_reference_mode="legacy_median",
            r_kurtosis_apply_only_if_oversegmented=False,
        )
        filtered = _reject_low_kurtosis_r_peaks(ecg, peaks, fs, cfg)
        assert false_r not in filtered
        assert len(filtered) >= 4

    def test_trp_refines_to_local_extremum(self):
        fs = 250.0
        raw = np.zeros(200)
        true_r = 80
        raw[true_r] = 3.0
        raw[true_r - 1] = 1.0
        raw[true_r + 1] = 1.0
        coarse = true_r - 3
        peaks = np.array([coarse], dtype=int)
        polarity = np.array([1.0])
        refined = _refine_r_peaks_trp(raw, peaks, polarity, fs, half_window_ms=30.0)
        assert refined[0] == true_r

    def test_postprocess_pipeline_preserves_real_r_peaks(self, simple_ecg_signal, sampling_rate):
        cfg = ProcessCycleConfig()
        peaks = r_peak_detection(
            simple_ecg_signal,
            sampling_rate,
            cfg=cfg,
        )
        assert len(peaks) >= 5

    def test_phase_a_can_disable_refrac_gate_only(self, simple_ecg_signal, sampling_rate):
        cfg = ProcessCycleConfig(
            r_post_detection_refrac_enabled=False,
            r_kurtosis_rejection=True,
            r_trp_on_input_signal=True,
        )
        peaks = r_peak_detection(
            simple_ecg_signal,
            sampling_rate,
            cfg=cfg,
        )
        assert len(peaks) >= 5

    def test_phase_a_can_be_disabled(self, simple_ecg_signal, sampling_rate):
        cfg = ProcessCycleConfig(
            r_post_detection_refrac_enabled=False,
            r_kurtosis_rejection=False,
            r_trp_on_input_signal=False,
            r_miss_beat_rr_factor=1.5,
        )
        peaks = r_peak_detection(
            simple_ecg_signal,
            sampling_rate,
            cfg=cfg,
        )
        assert len(peaks) >= 5

    def test_funnel_stats_keys(self, simple_ecg_signal, sampling_rate):
        cfg = ProcessCycleConfig.for_human()
        stats = r_peak_detection_funnel_stats(
            simple_ecg_signal,
            sampling_rate,
            cfg=cfg,
            raw_ecg=simple_ecg_signal,
        )
        assert stats["n_derivative"] >= stats["n_after_phase_a"]
        assert stats["n_after_phase_a"] == stats["n_after_trp"]
        assert stats["n_after_prp_spacing"] <= stats["n_before_phase_a"]

    def test_sharpest_peak_kurtosis_keeps_sharp_r_not_global_wipe(self):
        fs = 250.0
        n = 3000
        ecg = np.zeros(n)
        rr = int(0.8 * fs)
        r_idx = [200 + i * rr for i in range(8)]
        for r in r_idx:
            ecg[r] = 2.0
            ecg[r - 1] = ecg[r + 1] = -0.9
        for r in r_idx:
            t = r + int(0.35 * fs)
            if t < n:
                for j in range(max(0, t - 12), min(n, t + 13)):
                    ecg[j] = 0.5 * np.exp(-0.5 * ((j - t) / 8.0) ** 2)
        cfg = ProcessCycleConfig(
            r_prp_spacing_enabled=True,
            r_prp_min_interval_ms=400.0,
            r_kurtosis_reference_mode="sharpest_peak",
            r_kurtosis_fraction_of_sharpest=0.45,
            r_kurtosis_apply_only_if_oversegmented=False,
        )
        peaks = np.array(r_idx + [r_idx[0] + int(0.35 * fs)], dtype=int)
        half_w = 20
        k_vals = np.array([_local_kurtosis(ecg, int(p), half_w) for p in peaks])
        kept = _reject_low_kurtosis_sharpest_peak(k_vals, peaks, cfg)
        assert len(kept) >= len(r_idx)
        assert (r_idx[0] + int(0.35 * fs)) not in kept

    def test_kurtosis_skipped_when_not_oversegmented(self):
        fs = 250.0
        ecg = np.zeros(5000)
        ecg[100] = 2.0
        ecg[400] = 2.0
        cfg = ProcessCycleConfig(
            r_kurtosis_rejection=True,
            r_kurtosis_apply_only_if_oversegmented=True,
            r_kurtosis_oversegmented_peaks_per_min=200.0,
        )
        peaks = np.array([100, 400], dtype=int)
        assert not _should_apply_kurtosis_filter(peaks, len(ecg), fs, cfg)

    def test_extremum_refine_mode(self, sampling_rate):
        from pyhearts.processing.peaks import refine_r_peak_near_anchor

        signal = np.zeros(100)
        true_r = 40
        signal[true_r] = 2.0
        refined = refine_r_peak_near_anchor(
            signal,
            anchor_idx=true_r - 4,
            sampling_rate=sampling_rate,
            half_window_ms=30.0,
            refine_mode="extremum",
        )
        assert refined == true_r
