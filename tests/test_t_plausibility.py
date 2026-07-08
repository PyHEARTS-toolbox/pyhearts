"""Tests for RT plausibility gate."""

import numpy as np

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.t_plausibility import (
    apply_rt_plausibility_gate,
    check_t_peak_dominance,
    validate_p_pr_interval,
)


def test_rejects_implausible_rt_delay():
    fs = 250.0
    cfg = ProcessCycleConfig(t_rt_plausibility_gate=True, t_rt_bounds_ms=(120.0, 500.0))
    n = 10
    r = np.array([i * int(0.8 * fs) for i in range(n)], dtype=float)
    t = r + 0.32 * fs
    t[5] = r[5] + 0.02 * fs
    out = {
        "R_global_center_idx": list(r),
        "T_global_center_idx": list(t),
        "T_center_idx": list(t),
        "T_center_voltage": [0.1] * n,
        "T_center_ms": list(t / fs * 1000),
        "t_source": ["per_cycle"] * n,
        "t_confidence": ["high"] * n,
    }
    stats = apply_rt_plausibility_gate(out, fs, cfg)
    assert stats["rejected_bounds"] >= 1 or stats["rejected_outlier"] >= 1
    assert not np.isfinite(out["T_global_center_idx"][5])
    assert out["t_source"][5] is None
    assert out["t_confidence"][5] is None


def test_validate_p_pr_interval_rejects_too_close_to_r():
    cfg = ProcessCycleConfig(p_pr_interval_validation=True)
    assert not validate_p_pr_interval(95, 100, 250.0, cfg)
    assert validate_p_pr_interval(50, 100, 250.0, cfg)


def test_check_t_peak_dominance_rejects_secondary_bump():
    fs = 250.0
    cfg = ProcessCycleConfig(t_morphology_dominance_check=True, t_dominance_min_fraction=0.3)
    sig = np.zeros(200)
    lo, hi = 50, 150
    sig[lo:hi] = -0.1
    true_t = 80
    sig[true_t] = -0.8
    wrong = 120
    sig[wrong] = -0.2
    assert not check_t_peak_dominance(sig, wrong, lo, hi, 1, fs, cfg)
    assert check_t_peak_dominance(sig, true_t, lo, hi, 1, fs, cfg)
