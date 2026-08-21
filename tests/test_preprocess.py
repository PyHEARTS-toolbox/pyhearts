"""Tests for ECG preprocessing hardening."""

import numpy as np
import pytest

from pyhearts import PyHEARTS
from pyhearts._morphology.processing.preprocess import preprocess_ecg


FS = 500.0


def _ecg(n: int = 2000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    return (
        0.8 * np.sin(2 * np.pi * 1.2 * t)
        + 0.2 * np.sin(2 * np.pi * 0.25 * t)
        + 0.02 * rng.normal(size=n)
    )


def test_readme_recipe_returns_1d():
    analyzer = PyHEARTS(sampling_rate=FS, species="human")
    out = analyzer.preprocess_signal(
        _ecg(),
        highpass_cutoff=0.5,
        lowpass_cutoff=50.0,
        filter_order=4,
        notch_frequency=50.0,
        quality_factor=30.0,
    )
    assert isinstance(out, np.ndarray)
    assert out.ndim == 1
    assert out.shape == (2000,)
    assert np.isfinite(out).all()


def test_column_and_row_vectors_ravel():
    ecg = _ecg(500)
    col = preprocess_ecg(
        ecg.reshape(-1, 1),
        FS,
        highpass_cutoff=0.5,
        lowpass_cutoff=40.0,
        filter_order=4,
    )
    row = preprocess_ecg(
        ecg.reshape(1, -1),
        FS,
        highpass_cutoff=0.5,
        lowpass_cutoff=40.0,
        filter_order=4,
    )
    assert col.shape == (500,)
    assert row.shape == (500,)


def test_multilead_array_raises():
    with pytest.raises(ValueError, match="1-D"):
        preprocess_ecg(np.zeros((100, 2)), FS)


def test_unpaired_band_cutoff_raises():
    ecg = _ecg(200)
    with pytest.raises(ValueError, match="filter_order is required"):
        preprocess_ecg(ecg, FS, highpass_cutoff=0.5)
    with pytest.raises(ValueError, match="filter_order is required"):
        preprocess_ecg(ecg, FS, lowpass_cutoff=40.0)
    with pytest.raises(ValueError, match="neither highpass_cutoff nor lowpass_cutoff"):
        preprocess_ecg(ecg, FS, filter_order=4)


def test_unpaired_notch_raises():
    ecg = _ecg(200)
    with pytest.raises(ValueError, match="quality_factor is required"):
        preprocess_ecg(ecg, FS, notch_frequency=50.0)
    with pytest.raises(ValueError, match="notch_frequency was not provided"):
        preprocess_ecg(ecg, FS, quality_factor=30.0)


def test_sparse_nans_interpolated():
    ecg = _ecg(1000)
    ecg = ecg.copy()
    ecg[10:15] = np.nan
    out = preprocess_ecg(ecg, FS, highpass_cutoff=0.5, lowpass_cutoff=40.0, filter_order=4)
    assert np.isfinite(out).all()


def test_too_many_nans_raises():
    ecg = _ecg(1000)
    ecg = ecg.copy()
    ecg[:50] = np.nan  # 5% > default 1%
    with pytest.raises(ValueError, match="NaN fraction"):
        preprocess_ecg(ecg, FS)


def test_cutoff_above_nyquist_raises():
    with pytest.raises(ValueError, match="fs/2"):
        preprocess_ecg(_ecg(200), FS, highpass_cutoff=300.0, filter_order=4)


def test_errors_surface_not_none():
    """Short filtered signals must raise (not return None)."""
    with pytest.raises(ValueError):
        preprocess_ecg(
            np.ones(5),
            FS,
            highpass_cutoff=0.5,
            lowpass_cutoff=40.0,
            filter_order=4,
        )
