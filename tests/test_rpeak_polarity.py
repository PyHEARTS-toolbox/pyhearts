"""Tests for R-peak polarity / inverted-QRS handling."""

from __future__ import annotations

import numpy as np

from pyhearts._morphology.config import ProcessCycleConfig
from pyhearts._morphology.processing.rpeak import (
    detect_signal_polarity,
    r_peak_detection,
)


def _synthetic_upright(fs: float = 500.0, beats: int = 20) -> tuple[np.ndarray, np.ndarray]:
    duration = 25.0
    t = np.arange(0, duration, 1.0 / fs)
    signal = np.zeros_like(t)
    rr = 1.0
    r_times = np.arange(0.8, duration - 0.8, rr)
    for r_time in r_times:
        signal += 1.0 * np.exp(-0.5 * ((t - r_time) / 0.012) ** 2)
        signal += 0.15 * np.exp(-0.5 * ((t - (r_time - 0.16)) / 0.02) ** 2)
        signal += -0.12 * np.exp(-0.5 * ((t - (r_time - 0.04)) / 0.01) ** 2)
        signal += -0.2 * np.exp(-0.5 * ((t - (r_time + 0.04)) / 0.012) ** 2)
        signal += 0.3 * np.exp(-0.5 * ((t - (r_time + 0.22)) / 0.04) ** 2)
    r_peaks = np.asarray([int(round(rt * fs)) for rt in r_times], dtype=int)
    return signal, r_peaks[:beats]


def _match_rate(ref: np.ndarray, det: np.ndarray, tol: int) -> float:
    if ref.size == 0:
        return float("nan")
    used = np.zeros(det.size, dtype=bool)
    n = 0
    for r in ref:
        if det.size == 0:
            break
        d = np.abs(det - r)
        d[used] = tol + 1
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            n += 1
    return n / ref.size


def test_detect_signal_polarity_normal_and_inverted():
    fs = 500.0
    upright, _ = _synthetic_upright(fs)
    assert detect_signal_polarity(upright, fs) is False
    assert detect_signal_polarity(-upright, fs) is True


def test_r_peak_detection_recovers_inverted_lead():
    fs = 500.0
    upright, ref = _synthetic_upright(fs)
    inverted = -upright
    cfg = ProcessCycleConfig.for_human()
    tol = int(round(40.0 * fs / 1000.0))

    peaks_up = r_peak_detection(upright, fs, cfg=cfg)
    peaks_inv = r_peak_detection(inverted, fs, cfg=cfg)

    assert _match_rate(ref, peaks_up, tol) >= 0.95
    assert _match_rate(ref, peaks_inv, tol) >= 0.95
    assert detect_signal_polarity(inverted, fs, min_refrac_ms=cfg.rpeak_min_refrac_ms)


def test_r_peak_auto_polarity_can_be_disabled():
    fs = 500.0
    upright, ref = _synthetic_upright(fs)
    inverted = -upright
    cfg = ProcessCycleConfig.for_human()
    tol = int(round(20.0 * fs / 1000.0))  # tighter than Q/S offset (~40 ms)

    peaks_off = r_peak_detection(inverted, fs, cfg=cfg, auto_polarity=False)
    peaks_on = r_peak_detection(inverted, fs, cfg=cfg, auto_polarity=True)
    assert _match_rate(ref, peaks_on, tol) > _match_rate(ref, peaks_off, tol)
    assert _match_rate(ref, peaks_off, tol) < 0.5
