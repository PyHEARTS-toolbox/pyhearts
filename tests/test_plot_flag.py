"""Ensure plot=False never creates or shows matplotlib figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from pyhearts import PyHEARTS


def _deterministic_ecg(fs: float = 500.0, duration: float = 8.0) -> np.ndarray:
    times = np.linspace(0.0, duration, int(fs * duration))
    signal = np.zeros_like(times)
    for r_time in np.arange(0.5, duration - 0.5, 1.0):
        signal += 0.15 * np.exp(-((times - (r_time - 0.16)) ** 2) / (2 * 0.02**2))
        signal -= 0.10 * np.exp(-((times - (r_time - 0.04)) ** 2) / (2 * 0.01**2))
        signal += 1.00 * np.exp(-((times - r_time) ** 2) / (2 * 0.01**2))
        signal -= 0.20 * np.exp(-((times - (r_time + 0.04)) ** 2) / (2 * 0.01**2))
        signal += 0.30 * np.exp(-((times - (r_time + 0.25)) ** 2) / (2 * 0.04**2))
    return signal


def test_plot_false_does_not_render_figures(monkeypatch):
    calls = {"figure": 0, "subplots": 0, "show": 0}
    real_figure = plt.figure
    real_subplots = plt.subplots
    real_show = plt.show

    def counting_figure(*args, **kwargs):
        calls["figure"] += 1
        return real_figure(*args, **kwargs)

    def counting_subplots(*args, **kwargs):
        calls["subplots"] += 1
        return real_subplots(*args, **kwargs)

    def counting_show(*args, **kwargs):
        calls["show"] += 1
        return real_show(*args, **kwargs)

    monkeypatch.setattr(plt, "figure", counting_figure)
    monkeypatch.setattr(plt, "subplots", counting_subplots)
    monkeypatch.setattr(plt, "show", counting_show)

    fs = 500.0
    analyzer = PyHEARTS(sampling_rate=fs, plot=False, verbose=False, species="human")
    features, _cycles = analyzer.analyze_ecg(_deterministic_ecg(fs=fs))

    assert len(features) > 0
    assert calls == {"figure": 0, "subplots": 0, "show": 0}
