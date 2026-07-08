"""Tests for T-wave search window, QRS removal, config, and benchmark helpers."""

import numpy as np
import pytest

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.derivative_t_detection import compute_t_search_window
from pyhearts.processing.qrs_removal import remove_qrs_sigmoid

from pyhearts.processing.signal_crop import crop_signal
from scripts.benchmark_qtdb_t_sensitivity import filter_gold_to_crop, GoldBeat


class TestComputeTSearchWindow:
    def test_window_extends_with_long_cycle(self):
        fs = 250.0
        n = int(round(1.0 * fs))  # ~1 s cycle
        r = n // 2
        t_start, t_end = compute_t_search_window(r, n, fs, rr_frac=0.55, max_offset_ms=600.0)
        assert t_start >= r + int(round(0.08 * fs))  # at least ~80 ms after R
        assert t_end > r + int(round(0.35 * fs))  # well beyond old 450 ms cap for long cycles
        assert t_end < n

    def test_window_respects_cycle_end(self):
        fs = 500.0
        n = int(round(0.4 * fs))  # short cycle (tachycardia)
        r = n // 3
        t_start, t_end = compute_t_search_window(
            r, n, fs, end_margin_ms=40.0, max_offset_ms=600.0
        )
        assert t_end <= n - 1
        assert t_end - t_start >= int(round(0.1 * fs))


class TestQrsRemoval:
    def test_replaces_qrs_region(self):
        fs = 250.0
        n = 250
        t = np.linspace(0, 1, n)
        signal = 0.1 * np.sin(2 * np.pi * t) + np.zeros(n)
        r = n // 2
        signal[r - 5 : r + 6] = 2.0  # synthetic QRS bump

        cleaned = remove_qrs_sigmoid(signal, r, fs, post_qrs_ms=80.0)
        assert not np.allclose(cleaned[r - 5 : r + 6], signal[r - 5 : r + 6])
        assert np.isfinite(cleaned).all()


class TestBenchmarkCrop:
    def test_crop_fraction(self):
        ecg = np.arange(1000, dtype=float)
        cropped, start, end = crop_signal(ecg, 250.0, fraction=0.15)
        assert start == 0
        assert end == 150
        assert len(cropped) == 150

    def test_crop_duration(self):
        ecg = np.arange(10000, dtype=float)
        cropped, start, end = crop_signal(ecg, 250.0, duration_s=2.0)
        assert start == 0
        assert end == 500
        assert len(cropped) == 500

    def test_filter_gold_to_crop(self):
        gold = [
            GoldBeat("r1", 100, 200),
            GoldBeat("r1", 9000, 9100),
        ]
        kept = filter_gold_to_crop(gold, 0, 5000)
        assert len(kept) == 1
        assert kept[0].t_sample == 200

    def test_crop_from_end(self):
        sig = np.arange(1000, dtype=float)
        cropped, start, end = crop_signal(sig, 250.0, duration_s=1.0, from_end=True)
        assert end == 1000
        assert start == 750
        assert len(cropped) == 250


class TestTWaveConfig:
    def test_default_t_amp_ratio_relaxed(self):
        cfg = ProcessCycleConfig()
        assert cfg.amp_min_ratio["T"] == 0.02

    def test_qrs_removal_enabled_by_default(self):
        cfg = ProcessCycleConfig()
        assert cfg.t_wave_use_qrs_removal is True

    def test_human_preset_t_settings(self):
        cfg = ProcessCycleConfig.for_human()
        assert cfg.t_wave_use_qrs_removal is True
        assert cfg.amp_min_ratio["T"] == 0.020
        assert cfg.snr_mad_multiplier["T"] == 0.8

    def test_lite_mode_default_off(self):
        cfg = ProcessCycleConfig()
        assert cfg.lite_mode is False

    def test_phase2_t_fusion_defaults(self):
        cfg = ProcessCycleConfig()
        assert cfg.t_wave_use_record_prior is True
        assert cfg.t_wave_use_secondary_detector is True


class TestTWaveFusion:
    def test_estimate_record_rt_prior(self):
        from pyhearts.processing.t_wave_fusion import estimate_record_rt_prior

        r = np.array([100, 300, 500, 700, 900], dtype=float)
        t = np.array([180, 380, 580, 780, np.nan], dtype=float)
        prior = estimate_record_rt_prior(r, t, min_beats=3, sampling_rate=250.0)
        assert prior.valid
        assert prior.median_rt_samples == 80.0

    def test_detect_t_amplitude_peak(self):
        from pyhearts.processing.t_wave_fusion import detect_t_amplitude_peak

        sig = np.zeros(200)
        sig[120:140] = 0.5
        idx, h = detect_t_amplitude_peak(sig, 100, 180, inverted=False)
        assert idx is not None
        assert h == pytest.approx(0.5)
