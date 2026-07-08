"""Tests for template-prior window projection (phase 1)."""

from __future__ import annotations

import numpy as np

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.template_prior_windows import (
    global_window_to_cycle_relative,
)


def test_global_window_to_cycle_relative_maps_inclusive_global_bounds():
    bounds = global_window_to_cycle_relative(1050, 1120, cycle_start_global=1000, cycle_len=200)
    assert bounds == (50, 121)


def test_global_window_to_cycle_relative_returns_none_when_too_narrow():
    assert global_window_to_cycle_relative(1005, 1006, 1000, 50) is None


def test_compute_template_prior_windows_on_synthetic_record():
    from pyhearts.processing.template_prior_windows import compute_template_prior_windows

    fs = 250.0
    n = int(12 * fs)
    t = np.arange(n, dtype=float) / fs
    ecg = np.zeros(n)
    r_peaks = []
    for i, phase in enumerate(np.linspace(0.5, 11.0, 10)):
        r = int((phase + 0.12) * fs)
        p = int((phase - 0.08) * fs)
        tw = int((phase + 0.32) * fs)
        if 0 <= p < n:
            ecg[p : p + 5] += 0.15
        if 0 <= r < n:
            ecg[r - 2 : r + 3] += [ -0.1, -0.2, 1.0, -0.3, -0.1 ]
            r_peaks.append(r)
        if 0 <= tw < n:
            ecg[tw : tw + 8] += 0.25 * np.hanning(8)

    cfg = ProcessCycleConfig.for_human_unified_template_prior_phase1()
    cycles = list(range(len(r_peaks)))
    tmpl, windows = compute_template_prior_windows(
        ecg,
        np.asarray(r_peaks, dtype=int),
        cycles,
        fs,
        cfg,
    )
    assert tmpl is not None
    if tmpl.valid and len(r_peaks) > cfg.record_delineation_min_beats:
        assert len(windows) >= 1
        w0 = windows[0]
        assert w0.t_hi >= w0.t_lo
        assert w0.t_hi - w0.t_lo >= 2
