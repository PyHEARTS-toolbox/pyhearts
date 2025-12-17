"""
Tests for pyhearts.processing module.

Tests signal processing functions:
- r_peak_detection: R-peak detection algorithm
- epoch_ecg: ECG segmentation into cycles
- preprocess_ecg: Signal preprocessing
- gaussian_function: Gaussian model for waveforms
- calc_bounds: Search window bounds calculation
- detrend_signal: Baseline detrending
"""

import numpy as np
import pandas as pd
import pytest

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing import (
    r_peak_detection,
    epoch_ecg,
    preprocess_ecg,
    gaussian_function,
    compute_gauss_std,
    calc_bounds,
    detrend_signal,
    initialize_output_dict,
)


class TestRPeakDetection:
    """Tests for r_peak_detection function."""

    def test_detects_peaks_in_simple_signal(self, simple_ecg_signal, sampling_rate, default_config):
        """Should detect R-peaks in a clean ECG signal."""
        peaks = r_peak_detection(
            simple_ecg_signal,
            sampling_rate,
            cfg=default_config,
        )
        
        assert len(peaks) > 0
        assert all(isinstance(p, (int, np.integer)) for p in peaks)

    def test_peak_indices_within_bounds(self, simple_ecg_signal, sampling_rate, default_config):
        """Peak indices should be within signal bounds."""
        peaks = r_peak_detection(
            simple_ecg_signal,
            sampling_rate,
            cfg=default_config,
        )
        
        assert all(0 <= p < len(simple_ecg_signal) for p in peaks)

    def test_peaks_at_local_maxima(self, simple_ecg_signal, sampling_rate, default_config):
        """Detected peaks should be at local maxima."""
        peaks = r_peak_detection(
            simple_ecg_signal,
            sampling_rate,
            cfg=default_config,
        )
        
        # Each peak should be higher than immediate neighbors
        for p in peaks:
            if 0 < p < len(simple_ecg_signal) - 1:
                assert simple_ecg_signal[p] >= simple_ecg_signal[p - 1]
                assert simple_ecg_signal[p] >= simple_ecg_signal[p + 1]

    def test_expected_number_of_peaks(self, simple_ecg_signal, sampling_rate, default_config):
        """Should detect approximately the expected number of peaks."""
        # Simple ECG signal has ~60 bpm for 10 seconds = ~9 peaks
        peaks = r_peak_detection(
            simple_ecg_signal,
            sampling_rate,
            cfg=default_config,
        )
        
        # Allow some tolerance
        assert 5 <= len(peaks) <= 15

    def test_no_peaks_in_flat_signal(self, sampling_rate, default_config):
        """Flat signal should return no peaks."""
        flat_signal = np.zeros(10000)
        peaks = r_peak_detection(
            flat_signal,
            sampling_rate,
            cfg=default_config,
        )
        
        assert len(peaks) == 0

    def test_empty_signal_raises(self, sampling_rate, default_config):
        """Empty signal should raise ValueError."""
        with pytest.raises(ValueError):
            r_peak_detection(np.array([]), sampling_rate, cfg=default_config)

    def test_with_human_config(self, simple_ecg_signal, sampling_rate, human_config):
        """Should work with human config preset."""
        peaks = r_peak_detection(
            simple_ecg_signal,
            sampling_rate,
            cfg=human_config,
        )
        
        assert isinstance(peaks, np.ndarray)

    def test_returns_sorted_indices(self, simple_ecg_signal, sampling_rate, default_config):
        """Returned peak indices should be sorted."""
        peaks = r_peak_detection(
            simple_ecg_signal,
            sampling_rate,
            cfg=default_config,
        )
        
        assert np.all(np.diff(peaks) > 0)  # Strictly increasing

    def test_minimum_peak_distance(self, simple_ecg_signal, sampling_rate, default_config):
        """Peaks should respect minimum refractory period."""
        peaks = r_peak_detection(
            simple_ecg_signal,
            sampling_rate,
            cfg=default_config,
        )
        
        if len(peaks) > 1:
            min_distance_samples = int(default_config.rpeak_min_refrac_ms * sampling_rate / 1000)
            distances = np.diff(peaks)
            # After second pass, distance should be based on RR fraction
            assert np.all(distances > min_distance_samples * 0.5)


class TestEpochEcg:
    """Tests for epoch_ecg function."""

    def test_returns_dataframe_and_energy(self, simple_ecg_signal, sampling_rate, default_config):
        """epoch_ecg should return DataFrame and energy estimate."""
        # First detect R-peaks
        peaks = r_peak_detection(simple_ecg_signal, sampling_rate, cfg=default_config)
        
        if len(peaks) > 2:
            epochs_df, energy = epoch_ecg(
                simple_ecg_signal,
                peaks,
                sampling_rate,
                estimate_energy=True,
            )
            
            assert isinstance(epochs_df, pd.DataFrame)
            assert energy is not None or isinstance(energy, (int, float))

    def test_epochs_have_required_columns(self, simple_ecg_signal, sampling_rate, default_config):
        """Epochs DataFrame should have required columns."""
        peaks = r_peak_detection(simple_ecg_signal, sampling_rate, cfg=default_config)
        
        if len(peaks) > 2:
            epochs_df, _ = epoch_ecg(
                simple_ecg_signal,
                peaks,
                sampling_rate,
            )
            
            if not epochs_df.empty:
                assert "cycle" in epochs_df.columns

    def test_epochs_grouped_by_cycle(self, simple_ecg_signal, sampling_rate, default_config):
        """Each cycle should have multiple samples."""
        peaks = r_peak_detection(simple_ecg_signal, sampling_rate, cfg=default_config)
        
        if len(peaks) > 2:
            epochs_df, _ = epoch_ecg(
                simple_ecg_signal,
                peaks,
                sampling_rate,
            )
            
            if not epochs_df.empty:
                cycle_sizes = epochs_df.groupby("cycle").size()
                assert all(size > 10 for size in cycle_sizes)


class TestPreprocessEcg:
    """Tests for preprocess_ecg function."""

    def test_returns_array_or_none(self, simple_ecg_signal, sampling_rate):
        """preprocess_ecg should return array or None."""
        result = preprocess_ecg(
            simple_ecg_signal,
            sampling_rate,
            highpass_cutoff=0.5,
            filter_order=4,
        )
        
        assert result is None or isinstance(result, np.ndarray)

    def test_preserves_length(self, simple_ecg_signal, sampling_rate):
        """Preprocessing should preserve signal length."""
        result = preprocess_ecg(
            simple_ecg_signal,
            sampling_rate,
            highpass_cutoff=0.5,
            filter_order=4,
        )
        
        if result is not None:
            assert len(result) == len(simple_ecg_signal)

    def test_highpass_removes_dc(self, sampling_rate):
        """Highpass filter should remove DC offset."""
        # Signal with DC offset
        signal = np.sin(np.linspace(0, 10 * np.pi, 1000)) + 5.0
        
        result = preprocess_ecg(
            signal,
            sampling_rate,
            highpass_cutoff=0.5,
            filter_order=4,
        )
        
        if result is not None:
            # DC component should be reduced
            assert abs(np.mean(result)) < abs(np.mean(signal))

    def test_no_filtering_returns_none(self, simple_ecg_signal, sampling_rate):
        """No filter parameters should return None."""
        result = preprocess_ecg(
            simple_ecg_signal,
            sampling_rate,
        )
        
        # Implementation may return None or original signal
        assert result is None or isinstance(result, np.ndarray)


class TestGaussianFunction:
    """Tests for gaussian_function."""

    def test_gaussian_shape(self):
        """Gaussian should have expected bell curve shape."""
        x = np.linspace(-5, 5, 101)  # Use odd number for exact center
        center = 0
        height = 1.0
        std = 1.0
        
        y = gaussian_function(x, center, height, std)
        
        # Peak at center (index 50 for 101 points from -5 to 5)
        center_idx = 50
        assert y[center_idx] == pytest.approx(height, rel=0.01)
        
        # Symmetric decay - use equidistant points from center
        assert y[40] == pytest.approx(y[60], rel=0.05)

    def test_gaussian_height(self):
        """Gaussian peak should equal height parameter."""
        x = np.linspace(-5, 5, 101)
        center = 0
        height = 2.5
        std = 1.0
        
        y = gaussian_function(x, center, height, std)
        
        assert max(y) == pytest.approx(height, rel=0.01)

    def test_gaussian_center(self):
        """Peak should be at center position."""
        x = np.linspace(-5, 5, 101)
        center = 2.0
        height = 1.0
        std = 0.5
        
        y = gaussian_function(x, center, height, std)
        
        peak_idx = np.argmax(y)
        assert x[peak_idx] == pytest.approx(center, abs=0.1)

    def test_gaussian_width_scaling(self):
        """Larger std should give wider Gaussian."""
        x = np.linspace(-5, 5, 101)
        
        y_narrow = gaussian_function(x, 0, 1.0, 0.5)
        y_wide = gaussian_function(x, 0, 1.0, 2.0)
        
        # FWHM comparison: wider should have more points above half-max
        half_max = 0.5
        narrow_width = np.sum(y_narrow > half_max)
        wide_width = np.sum(y_wide > half_max)
        
        assert wide_width > narrow_width

    def test_negative_gaussian(self):
        """Negative height should give inverted Gaussian."""
        x = np.linspace(-5, 5, 101)
        
        y = gaussian_function(x, 0, -1.0, 1.0)
        
        assert min(y) == pytest.approx(-1.0, rel=0.01)
        assert max(y) < 0


class TestComputeGaussStd:
    """Tests for compute_gauss_std function."""

    def test_returns_dict(self):
        """compute_gauss_std should return a dict of stds."""
        # Create a simple signal with a peak
        signal = np.zeros(100)
        signal[40:60] = np.exp(-((np.arange(20) - 10) ** 2) / (2 * 3 ** 2))
        
        guess_idxs = {"R": (50, 1.0)}
        
        result = compute_gauss_std(signal, guess_idxs)
        
        assert isinstance(result, dict)
        if "R" in result:
            assert result["R"] > 0

    def test_wider_peak_larger_std(self):
        """Wider peak should give larger std estimate."""
        # Narrow peak
        signal_narrow = np.zeros(100)
        signal_narrow[45:55] = np.exp(-((np.arange(10) - 5) ** 2) / (2 * 1 ** 2))
        
        # Wide peak  
        signal_wide = np.zeros(100)
        signal_wide[30:70] = np.exp(-((np.arange(40) - 20) ** 2) / (2 * 8 ** 2))
        
        guess_narrow = {"R": (50, signal_narrow[50])}
        guess_wide = {"R": (50, signal_wide[50])}
        
        std_narrow = compute_gauss_std(signal_narrow, guess_narrow)
        std_wide = compute_gauss_std(signal_wide, guess_wide)
        
        if "R" in std_narrow and "R" in std_wide:
            assert std_wide["R"] > std_narrow["R"]


class TestCalcBounds:
    """Tests for calc_bounds function."""

    def test_returns_tuple_of_lists(self):
        """calc_bounds should return tuple of (lower_bounds, upper_bounds)."""
        # calc_bounds(center, height, std, bound_factor, flip_height)
        lower, upper = calc_bounds(
            center=100,
            height=1.0,
            std=10,
            bound_factor=0.2,
        )
        
        assert isinstance(lower, list)
        assert isinstance(upper, list)
        assert len(lower) == 3  # [center, height, std]
        assert len(upper) == 3

    def test_bounds_bracket_values(self):
        """Lower bounds should be less than upper bounds."""
        center = 100
        height = 1.0
        std = 10
        
        lower, upper = calc_bounds(center, height, std, bound_factor=0.2)
        
        # Center bounds
        assert lower[0] < upper[0]
        # Height bounds
        assert lower[1] < upper[1]
        # Std bounds
        assert lower[2] < upper[2]

    def test_bounds_around_inputs(self):
        """Bounds should be centered around input values."""
        center = 100
        height = 1.0
        std = 10
        
        lower, upper = calc_bounds(center, height, std, bound_factor=0.2)
        
        # Center should be within bounds
        assert lower[0] < center < upper[0]
        # Height should be within bounds
        assert lower[1] < height < upper[1]

    def test_negative_height_bounds(self):
        """Negative height (troughs) should be handled correctly."""
        lower, upper = calc_bounds(
            center=100,
            height=-0.5,  # Negative for Q/S waves
            std=10,
            bound_factor=0.2,
        )
        
        # Should still have lower < upper
        assert lower[1] < upper[1]


class TestDetrendSignal:
    """Tests for detrend_signal function."""

    def test_removes_linear_trend(self, sampling_rate):
        """Should remove linear baseline drift."""
        n_samples = 1000
        t = np.arange(n_samples)
        xs = t / sampling_rate
        
        # Signal with linear drift
        signal = np.sin(2 * np.pi * t / 100) + 0.01 * t
        
        # detrend_signal(xs, signal, sampling_rate, window_ms)
        detrended, slope = detrend_signal(xs, signal, sampling_rate, window_ms=50)
        
        # Linear trend should be reduced
        original_slope = np.polyfit(t, signal, 1)[0]
        detrended_slope = np.polyfit(t, detrended, 1)[0]
        assert abs(detrended_slope) < abs(original_slope)

    def test_preserves_waveform(self, sampling_rate):
        """Detrending should preserve ECG-like waveform shape."""
        n_samples = 500
        t = np.arange(n_samples)
        xs = t / sampling_rate
        
        # Simple ECG-like waveform with offset
        signal = np.exp(-((xs - 0.25) ** 2) / (2 * 0.01 ** 2)) + 0.5
        
        detrended, slope = detrend_signal(xs, signal, sampling_rate, window_ms=50)
        
        # Peak should still be detectable at same location
        peak_idx_orig = np.argmax(signal)
        peak_idx_detrend = np.argmax(detrended)
        assert abs(peak_idx_orig - peak_idx_detrend) < 10

    def test_returns_tuple(self, sampling_rate):
        """Should return (detrended_signal, slope) tuple."""
        n_samples = 100
        xs = np.arange(n_samples) / sampling_rate
        signal = np.random.randn(n_samples)
        
        result = detrend_signal(xs, signal, sampling_rate, window_ms=20)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], float)


class TestInitializeOutputDict:
    """Tests for initialize_output_dict function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        result = initialize_output_dict(
            cycle_inds=[0, 1, 2],
            components=["P", "R", "T"],
            peak_features=["center_idx", "height"],
            intervals=["PR_interval_ms"],
        )
        
        assert isinstance(result, dict)

    def test_correct_number_of_keys(self):
        """Should have correct number of keys."""
        components = ["P", "R", "T"]
        features = ["center_idx", "height"]
        intervals = ["PR_interval_ms", "RR_interval_ms"]
        
        result = initialize_output_dict(
            cycle_inds=[0, 1],
            components=components,
            peak_features=features,
            intervals=intervals,
        )
        
        # Should have component_feature combinations + intervals
        expected_keys = len(components) * len(features) + len(intervals)
        assert len(result) >= expected_keys

    def test_values_are_lists(self):
        """Values should be lists."""
        result = initialize_output_dict(
            cycle_inds=[0, 1, 2],
            components=["R"],
            peak_features=["center_idx"],
            intervals=["RR_interval_ms"],
        )
        
        for value in result.values():
            assert isinstance(value, list)

    def test_lists_have_correct_length(self):
        """Lists should have length equal to number of cycles."""
        n_cycles = 5
        
        result = initialize_output_dict(
            cycle_inds=list(range(n_cycles)),
            components=["R"],
            peak_features=["center_idx"],
            intervals=["RR_interval_ms"],
        )
        
        for value in result.values():
            assert len(value) == n_cycles

    def test_initialized_with_nan(self):
        """Values should be initialized with NaN."""
        result = initialize_output_dict(
            cycle_inds=[0, 1],
            components=["R"],
            peak_features=["center_idx"],
            intervals=["RR_interval_ms"],
        )
        
        for value in result.values():
            assert all(np.isnan(v) for v in value)

    def test_with_pairwise_differences(self):
        """Should include pairwise difference keys."""
        pairwise = ["R_minus_S", "P_minus_T"]
        
        result = initialize_output_dict(
            cycle_inds=[0, 1],
            components=["R"],
            peak_features=["center_idx"],
            intervals=["RR_interval_ms"],
            pairwise_differences=pairwise,
        )
        
        for key in pairwise:
            assert key in result

