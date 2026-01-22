"""
Additional unit tests covering exported helpers in `pyhearts.processing`.

These are focused "smoke" tests: verify basic behavior and guardrails without
requiring real ECG datasets.
"""

from dataclasses import replace

import numpy as np

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing import (
    assess_signal_quality,
    calc_bounds_skewed,
    calc_wavelet_dynamic_offset,
    find_peaks,
    gate_by_local_mad,
    skewed_gaussian_function,
    validate_cycle_physiology,
    validate_intervals_physiological,
    validate_peak_temporal_order,
)
from pyhearts.processing.snrgate import half_fwhm_samples


def test_half_fwhm_samples_returns_positive(simple_ecg_signal):
    idx = int(np.argmax(simple_ecg_signal))
    half = half_fwhm_samples(simple_ecg_signal, idx)
    assert isinstance(half, int)
    assert half >= 1


def test_gate_by_local_mad_accepts_strong_peak():
    # Small noise + one strong peak
    rng = np.random.default_rng(0)
    seg = rng.normal(0.0, 0.01, 200)
    seg[100] += 1.0

    keep, rel_idx, height = gate_by_local_mad(seg, sampling_rate=1000.0, comp="T")
    assert keep is True
    assert rel_idx is not None and 0 <= rel_idx < len(seg)
    assert height is not None and np.isfinite(height)


def test_find_peaks_derivative_mode_finds_peak(sample_epoch_df, sampling_rate):
    sig = sample_epoch_df["signal"].to_numpy()
    xs = np.arange(sig.size) / float(sampling_rate)

    idx, amp, center = find_peaks(
        signal=sig,
        xs=xs,
        start_idx=0,
        end_idx=sig.size,
        mode="max",
        verbose=False,
        use_derivative=True,
    )
    assert idx is not None
    assert amp is not None
    assert center is not None


def test_validate_peak_temporal_order_flags_out_of_order():
    peak_data = {
        "P": {"center_idx": 10},
        "R": {"center_idx": 50},
        "T": {"center_idx": 40},  # out of order (T before R)
    }
    ok, errors = validate_peak_temporal_order(peak_data, verbose=False)
    assert ok is False
    assert len(errors) >= 1


def test_validate_intervals_physiological_flags_out_of_range():
    intervals = {"QT_interval_ms": 1000.0}  # above max 750
    ok, errors = validate_intervals_physiological(intervals, sampling_rate=1000.0, verbose=False)
    assert ok is False
    assert any("QT_interval_ms" in e for e in errors)


def test_validate_cycle_physiology_combines_checks():
    peak_data = {"R": {"center_idx": 10}, "P": {"center_idx": 20}}  # out of order
    intervals = {"RR_interval_ms": 100.0}  # below min 300
    ok, errors = validate_cycle_physiology(peak_data, intervals, sampling_rate=1000.0, verbose=False)
    assert ok is False
    assert "peak_ordering" in errors and "intervals" in errors
    assert errors["peak_ordering"] or errors["intervals"]


def test_assess_signal_quality_short_signal_rejected():
    ok, metrics, reason = assess_signal_quality(np.zeros(50), sampling_rate=1000.0)
    assert ok is False
    assert metrics == {}
    assert "short" in reason.lower()


def test_assess_signal_quality_clean_signal_acceptable(simple_ecg_signal, sampling_rate):
    ok, metrics, reason = assess_signal_quality(simple_ecg_signal, sampling_rate=float(sampling_rate))
    assert isinstance(ok, bool)
    assert isinstance(metrics, dict)
    assert isinstance(reason, str)


def test_calc_bounds_skewed_includes_alpha():
    lo, hi = calc_bounds_skewed(center=100, height=1.0, std=10, alpha=0.5, bound_factor=0.2)
    assert len(lo) == 4
    assert len(hi) == 4
    assert lo[3] < hi[3]


def test_skewed_gaussian_function_outputs_finite():
    xs = np.linspace(-5, 5, 101)
    ys = skewed_gaussian_function(xs, 0.0, 1.0, 1.0, 0.0)  # alpha=0
    assert ys.shape == xs.shape
    assert np.isfinite(ys).all()


def test_calc_wavelet_dynamic_offset_returns_int(simple_ecg_signal, sampling_rate):
    cfg = replace(ProcessCycleConfig(), wavelet_name="db6", wavelet_detail_level=3)
    offset, rL, rR, q, s = calc_wavelet_dynamic_offset(
        ecg_signal=simple_ecg_signal,
        sampling_rate=float(sampling_rate),
        expected_max_energy=1.0,
        plot=False,
        cfg=cfg,
    )
    assert isinstance(offset, int)
    assert rL is None and rR is None and q is None and s is None


