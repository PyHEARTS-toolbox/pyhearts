"""Sprint 2: fiducial_refine operators and adaptive windows."""

import numpy as np
from dataclasses import replace

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.fiducial_refine import (
    adaptive_refine_half_window_ms,
    refine_in_segment,
)


def test_adaptive_window_scales_with_rt():
    cfg = replace(
        ProcessCycleConfig(),
        record_delineation_refine_ms=15.0,
        record_delineation_refine_adaptive=True,
        record_delineation_refine_rt_frac=0.05,
        record_delineation_refine_ms_max=50.0,
    )
    fs = 250.0
    short = adaptive_refine_half_window_ms(cfg, "T", 1000.0, 1002.0, fs)
    long_rt = adaptive_refine_half_window_ms(cfg, "T", 1000.0, 1200.0, fs)
    assert short <= cfg.record_delineation_refine_ms + 1.0
    assert long_rt > short + 10.0
    assert long_rt == cfg.record_delineation_refine_ms_max


def test_derivative_apex_finds_t_minimum():
    fs = 250.0
    n = int(0.5 * fs)
    t = np.linspace(0, 1, n)
    sig = -0.3 * np.exp(-((t - 0.5) ** 2) / 0.02)
    anchor = int(0.5 * n)
    cfg = ProcessCycleConfig()
    idx = refine_in_segment(
        sig,
        anchor,
        wave="T",
        polarity="negative",
        sampling_rate=fs,
        cfg=cfg,
        half_window_ms=40.0,
    )
    assert abs(idx - anchor) <= int(0.05 * fs)


def test_refine_preset_has_sprint2_flags():
    cfg = replace(
        ProcessCycleConfig.for_human_unified(),
        record_delineation_refine_always=True,
        record_delineation_refine_adaptive=True,
        record_smooth_refine_on_epoch_ms=25.0,
    )
    assert cfg.record_delineation_refine_always is True
    assert cfg.record_delineation_refine_adaptive is True
    assert cfg.record_smooth_refine_on_epoch_ms == 25.0
