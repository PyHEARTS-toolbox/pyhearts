"""Tests for Gaussian ECG reconstruction from morphology features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyhearts import PyHEARTS, ReconstructedECG, reconstruct_ecg
from pyhearts.processing.gaussian import gaussian_function
from pyhearts.processing.reconstruct import reconstruct_cycle, resolve_wave_params


def _two_beat_features() -> pd.DataFrame:
    """Known P/Q/R/S/T Gaussians at two R locations 400 samples apart."""
    rows = []
    for cycle, r_mu in enumerate((200.0, 600.0)):
        rows.append(
            {
                "P_gauss_center": 80.0,
                "P_center_idx": 80.0,
                "P_global_center_idx": r_mu - 80.0,
                "P_gauss_height": 0.15,
                "P_gauss_stdev_samples": 10.0,
                "P_global_le_idx": r_mu - 100.0,
                "P_global_ri_idx": r_mu - 60.0,
                "Q_gauss_center": 140.0,
                "Q_center_idx": 140.0,
                "Q_global_center_idx": r_mu - 20.0,
                "Q_gauss_height": -0.10,
                "Q_gauss_stdev_samples": 6.0,
                "R_gauss_center": 160.0,
                "R_center_idx": 160.0,
                "R_global_center_idx": r_mu,
                "R_gauss_height": 1.00,
                "R_gauss_stdev_samples": 5.0,
                "S_gauss_center": 175.0,
                "S_center_idx": 175.0,
                "S_global_center_idx": r_mu + 15.0,
                "S_gauss_height": -0.20,
                "S_gauss_stdev_samples": 6.0,
                "T_gauss_center": 250.0,
                "T_center_idx": 250.0,
                "T_global_center_idx": r_mu + 90.0,
                "T_gaussian_global_center_idx": r_mu + 90.0,
                "T_gauss_height": 0.30,
                "T_gauss_stdev_samples": 18.0,
                "rmse": 0.04,
            }
        )
    return pd.DataFrame(rows)


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


def test_reconstruct_places_beats_on_sample_index():
    features = _two_beat_features()
    recon = reconstruct_ecg(features, sampling_rate=500.0, add_noise=False, n_samples=900)

    assert isinstance(recon, ReconstructedECG)
    assert recon.index.shape == recon.gaussian.shape == (900,)
    np.testing.assert_allclose(recon.time_ms[500], 1000.0)
    r_peaks = [
        int(np.argmax(np.abs(recon.components["R"][i0:i1]))) + i0
        for i0, i1 in ((150, 250), (550, 650))
    ]
    assert r_peaks[0] == pytest.approx(200, abs=1)
    assert r_peaks[1] == pytest.approx(600, abs=1)
    assert r_peaks[1] - r_peaks[0] == pytest.approx(400, abs=1)


def test_reconstruct_uses_gaussian_t_not_record_t():
    features = _two_beat_features()
    features.loc[0, "T_global_center_idx"] = 50.0  # record-T pulled far left
    recon = reconstruct_ecg(features, sampling_rate=500.0, add_noise=False, n_samples=400)
    t_peak = int(np.argmax(recon.components["T"][:400]))
    assert t_peak == pytest.approx(290, abs=2)


def test_reconstruct_cycle_matches_gaussian_function():
    row = _two_beat_features().iloc[0]
    n = 320
    xs = np.arange(n, dtype=float)
    expected = np.zeros(n)
    for mu, h, s in (
        (80.0, 0.15, 10.0),
        (140.0, -0.10, 6.0),
        (160.0, 1.00, 5.0),
        (175.0, -0.20, 6.0),
        (250.0, 0.30, 18.0),
    ):
        expected += gaussian_function(xs, mu, h, s)
    got = reconstruct_cycle(row, n)
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-8)


def test_residual_noise_recovers_original():
    features = _two_beat_features()
    fs = 500.0
    n = 900
    recon_clean = reconstruct_ecg(features, fs, add_noise=False, n_samples=n)
    rng = np.random.default_rng(0)
    original = recon_clean.gaussian + rng.normal(0.0, 0.05, size=n)

    recon = reconstruct_ecg(features, fs, original=original, add_noise=True, n_samples=n)
    np.testing.assert_allclose(recon.signal, original, atol=1e-12)
    np.testing.assert_allclose(recon.gaussian + recon.noise, original, atol=1e-12)
    assert recon.extras["noise_source"] == "residual"
    assert np.std(recon.noise) == pytest.approx(0.05, rel=0.15)


def test_isoelectric_noise_keeps_baseline_jitter():
    features = _two_beat_features()
    fs = 500.0
    n = 900
    clean = reconstruct_ecg(features, fs, add_noise=False, n_samples=n).gaussian
    rng = np.random.default_rng(1)
    original = clean + rng.normal(0.0, 0.03, size=n)

    recon = reconstruct_ecg(
        features, fs, original=original, noise_mode="isoelectric", n_samples=n
    )
    # Baseline (far from either R) should keep original jitter.
    assert np.std(recon.noise[:80]) == pytest.approx(0.03, rel=0.4)
    # Wave support is not a copy of the full residual.
    r_slice = recon.noise[190:210]
    assert not np.allclose(r_slice, (original - clean)[190:210])


def test_rmse_noise_is_reproducible():
    features = _two_beat_features()
    a = reconstruct_ecg(features, 500.0, noise_mode="rmse", n_samples=900, rng=7)
    b = reconstruct_ecg(features, 500.0, noise_mode="rmse", n_samples=900, rng=7)
    np.testing.assert_allclose(a.noise, b.noise)
    assert np.std(a.noise) > 0


def test_sigma_falls_back_to_fwhm_and_rise_decay():
    row = pd.Series(
        {
            "R_gauss_center": 100.0,
            "R_center_idx": 100.0,
            "R_global_center_idx": 500.0,
            "R_gauss_height": 1.0,
            "R_gauss_fwhm_samples": 2.3548 * 8.0,
        }
    )
    mu, height, sigma, *_ = resolve_wave_params(row, "R", 500.0)
    assert mu == pytest.approx(500.0)
    assert height == pytest.approx(1.0)
    assert sigma == pytest.approx(8.0, rel=1e-3)

    row_ms = pd.Series(
        {
            "R_center_ms": 1000.0,
            "R_center_voltage": 0.8,
            "R_rise_ms": 20.0,
            "R_decay_ms": 20.0,
        }
    )
    mu, height, sigma, *_ = resolve_wave_params(row_ms, "R", 500.0)
    assert mu == pytest.approx(500.0)
    assert height == pytest.approx(0.8)
    assert sigma == pytest.approx(5.0)  # (20+20)/4 ms → 5 samples at 500 Hz


def test_cycles_table_maps_relative_center_to_global_index():
    features = pd.DataFrame(
        {
            "R_gauss_center": [10.0],
            "R_center_idx": [10.0],
            "R_global_center_idx": [9999.0],  # decoy if cycles are ignored
            "R_gauss_height": [1.0],
            "R_gauss_stdev_samples": [4.0],
        }
    )
    cycles = pd.DataFrame(
        {
            "index": np.arange(100, 130),
            "signal_y": np.zeros(30),
            "cycle": np.zeros(30, dtype=int),
        }
    )
    recon = reconstruct_ecg(
        features, 500.0, cycles=cycles, add_noise=False, n_samples=150
    )
    peak = int(np.argmax(recon.components["R"]))
    assert peak == pytest.approx(110, abs=1)


def test_analyzer_reconstruct_after_analyze_ecg():
    fs = 500.0
    rng = np.random.default_rng(0)
    ecg = _deterministic_ecg(fs) + rng.normal(0.0, 0.02, int(fs * 10.0))
    analyzer = PyHEARTS(sampling_rate=fs, species="human")
    features, cycles = analyzer.analyze_ecg(ecg)
    assert not features.empty

    recon = analyzer.reconstruct_ecg(original=ecg)
    assert recon.gaussian.size == ecg.size
    np.testing.assert_allclose(recon.signal, ecg, atol=1e-10)

    r_centers = pd.to_numeric(features["R_global_center_idx"], errors="coerce")
    r_centers = r_centers[np.isfinite(r_centers)]
    assert len(r_centers) >= 1
    r_trace = recon.components["R"]
    for center in r_centers:
        i = int(round(float(center)))
        lo, hi = max(0, i - 25), min(r_trace.size, i + 26)
        peak = int(np.argmax(r_trace[lo:hi])) + lo
        assert peak == pytest.approx(i, abs=15)


def test_reconstruct_ecg_requires_analyze_first():
    analyzer = PyHEARTS(sampling_rate=500.0, species="human")
    with pytest.raises(RuntimeError, match="analyze_ecg"):
        analyzer.reconstruct_ecg()
