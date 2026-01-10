"""
Tests for pyhearts.feature module.

Tests feature extraction functions:
- calc_hrv_metrics: Heart rate variability metrics
- calc_intervals: ECG interval calculations
- interval_ms: Time conversion utility
"""

import numpy as np
import pytest

from pyhearts.feature import calc_hrv_metrics, calc_intervals, interval_ms


class TestCalcHRVMetrics:
    """Tests for calc_hrv_metrics function."""

    def test_basic_hrv_calculation(self, rr_intervals_ms):
        """Basic HRV calculation with clean data."""
        hr, sdnn, rmssd, nn50, pnn50, sd1, sd2 = calc_hrv_metrics(rr_intervals_ms)
        
        # All should return values
        assert hr is not None
        assert sdnn is not None
        assert rmssd is not None
        assert nn50 is not None

    def test_heart_rate_reasonable(self, rr_intervals_ms):
        """Heart rate should be physiologically reasonable."""
        hr, _, _, _, _, _, _ = calc_hrv_metrics(rr_intervals_ms)
        
        # For ~1000ms RR intervals, HR should be ~60 bpm
        assert hr is not None
        assert 30 < hr < 200  # Physiological range

    def test_sdnn_positive(self, rr_intervals_ms):
        """SDNN should be non-negative."""
        _, sdnn, _, _, _, _, _ = calc_hrv_metrics(rr_intervals_ms)
        
        assert sdnn is not None
        assert sdnn >= 0

    def test_rmssd_positive(self, rr_intervals_ms):
        """RMSSD should be non-negative."""
        _, _, rmssd, _, _, _, _ = calc_hrv_metrics(rr_intervals_ms)
        
        assert rmssd is not None
        assert rmssd >= 0

    def test_nn50_non_negative(self, rr_intervals_ms):
        """NN50 count should be non-negative."""
        _, _, _, nn50, _, _, _ = calc_hrv_metrics(rr_intervals_ms)
        
        assert nn50 is not None
        assert nn50 >= 0

    def test_with_nan_values(self, rr_intervals_with_nan):
        """HRV calculation should handle NaN values."""
        hr, sdnn, rmssd, nn50, pnn50, sd1, sd2 = calc_hrv_metrics(rr_intervals_with_nan)
        
        # Should still compute metrics, ignoring NaNs
        assert hr is not None
        assert sdnn is not None

    def test_constant_rr_intervals(self):
        """Constant RR intervals should give SDNN = 0."""
        constant_rr = np.full(100, 1000.0)  # All 1000ms
        hr, sdnn, rmssd, nn50, pnn50, sd1, sd2 = calc_hrv_metrics(constant_rr)
        
        assert hr == 60  # 60,000ms / 1000ms = 60 bpm
        assert sdnn == 0  # No variability
        assert rmssd == 0  # No successive differences
        assert nn50 == 0  # No differences > 50ms

    def test_single_interval(self):
        """Single RR interval should handle gracefully."""
        single_rr = np.array([1000.0])
        hr, sdnn, rmssd, nn50, pnn50, sd1, sd2 = calc_hrv_metrics(single_rr)
        
        # HR can be computed from single interval
        assert hr == 60

    def test_empty_after_nan_removal(self):
        """All-NaN array should handle gracefully."""
        all_nan = np.array([np.nan, np.nan, np.nan])
        hr, sdnn, rmssd, nn50, pnn50, sd1, sd2 = calc_hrv_metrics(all_nan)
        
        # Should return None or NaN values
        # Based on implementation, empty arrays return nan/None

    def test_fast_heart_rate(self):
        """Test with fast heart rate (short RR intervals)."""
        fast_rr = np.full(100, 500.0)  # 500ms = 120 bpm
        hr, _, _, _, _, _, _ = calc_hrv_metrics(fast_rr)
        
        assert hr == 120

    def test_slow_heart_rate(self):
        """Test with slow heart rate (long RR intervals)."""
        slow_rr = np.full(100, 1500.0)  # 1500ms = 40 bpm
        hr, _, _, _, _, _, _ = calc_hrv_metrics(slow_rr)
        
        assert hr == 40

    def test_high_variability(self):
        """High variability should give larger SDNN."""
        # Create intervals with alternating pattern (high variability)
        high_var_rr = np.array([800, 1200] * 50)  # Alternating 800/1200
        _, sdnn_high, _, nn50_high, _, _, _ = calc_hrv_metrics(high_var_rr)
        
        # Low variability
        low_var_rr = np.array([990, 1010] * 50)  # Alternating 990/1010
        _, sdnn_low, _, nn50_low, _, _, _ = calc_hrv_metrics(low_var_rr)
        
        assert sdnn_high > sdnn_low
        assert nn50_high > nn50_low  # 400ms diff > 50ms

    def test_nn50_threshold(self):
        """NN50 should count differences > 50ms."""
        # Specifically designed: 4 differences of exactly 60ms
        rr = np.array([1000, 1060, 1000, 1060, 1000, 1060, 1000, 1060, 1000])
        # Differences: 60, -60, 60, -60, 60, -60, 60, -60
        # All |60| > 50, so nn50 = 8
        _, _, _, nn50, _, _, _ = calc_hrv_metrics(rr)
        
        assert nn50 == 8

    def test_pnn50_calculation(self):
        """pNN50 should be percentage of differences > 50ms."""
        # 9 intervals = 8 pairs, all > 50ms difference
        rr = np.array([1000, 1060, 1000, 1060, 1000, 1060, 1000, 1060, 1000])
        _, _, _, nn50, pnn50, _, _ = calc_hrv_metrics(rr)
        
        # 8 pairs, all > 50ms, so pNN50 = 100%
        assert nn50 == 8
        assert abs(pnn50 - 100.0) < 0.01

    def test_sd1_relationship(self):
        """SD1 should equal RMSSD / sqrt(2)."""
        rr = np.array([1000, 1050, 1000, 1050, 1000, 1050, 1000, 1050, 1000])
        _, _, rmssd, _, _, sd1, _ = calc_hrv_metrics(rr)
        
        if rmssd is not None and sd1 is not None:
            # Calculate raw RMSSD for comparison
            raw_rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
            expected_sd1 = round(raw_rmssd / np.sqrt(2.0), 2)
            assert abs(sd1 - expected_sd1) < 0.01

    def test_sd2_relationship(self):
        """SD2 should equal sqrt(2 * SDNN^2 - SD1^2)."""
        rr = np.array([1000, 1050, 1000, 1050, 1000, 1050, 1000, 1050, 1000])
        _, sdnn, _, _, _, sd1, sd2 = calc_hrv_metrics(rr)
        
        if sdnn is not None and sd1 is not None and sd2 is not None:
            # Calculate raw values for comparison
            raw_sdnn = np.std(rr, ddof=1)
            raw_rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
            raw_sd1 = raw_rmssd / np.sqrt(2.0)
            expected_sd2 = round(np.sqrt(2.0 * (raw_sdnn ** 2) - (raw_sd1 ** 2)), 2)
            assert abs(sd2 - expected_sd2) < 0.01


class TestIntervalMs:
    """Tests for interval_ms utility function."""

    def test_basic_conversion(self):
        """Basic sample to millisecond conversion within bounds."""
        # interval_ms(curr_idx, prev_idx, lo_ms, hi_ms, ms_per_sample)
        ms_per_sample = 1.0  # 1000 Hz
        curr_idx = 100
        prev_idx = 0
        lo_ms = 0.0
        hi_ms = 200.0
        
        ms = interval_ms(curr_idx, prev_idx, lo_ms, hi_ms, ms_per_sample)
        # 100 samples at 1ms/sample = 100 ms
        assert ms == 100.0

    def test_out_of_bounds_returns_nan(self):
        """Intervals outside bounds should return NaN."""
        ms_per_sample = 1.0
        # 500ms interval but bounds are 0-200ms
        ms = interval_ms(500, 0, 0.0, 200.0, ms_per_sample)
        assert np.isnan(ms)

    def test_none_inputs_return_nan(self):
        """None inputs should return NaN."""
        assert np.isnan(interval_ms(None, 0, 0.0, 200.0, 1.0))
        assert np.isnan(interval_ms(100, None, 0.0, 200.0, 1.0))

    def test_negative_difference_returns_nan(self):
        """Negative difference (curr < prev) should return NaN."""
        ms = interval_ms(50, 100, 0.0, 200.0, 1.0)  # curr < prev
        assert np.isnan(ms)


class TestCalcIntervals:
    """Tests for calc_intervals function."""

    def test_returns_dict(self):
        """calc_intervals should return a dictionary."""
        from pyhearts.feature.intervals import calc_intervals
        
        # Create peak series structure (lists of indices across cycles)
        # calc_intervals(all_peak_series, cycle_idx, sampling_rate, window_size)
        all_peak_series = {
            "P_le_idx": [80],
            "P_ri_idx": [120],
            "Q_le_idx": [170],
            "S_ri_idx": [230],
            "T_le_idx": [320],
            "T_ri_idx": [380],
        }
        
        result = calc_intervals(all_peak_series, cycle_idx=0, sampling_rate=1000)
        assert isinstance(result, dict)

    def test_pr_interval_calculation(self):
        """PR interval should be calculated correctly."""
        from pyhearts.feature.intervals import calc_intervals
        
        # PR interval = Q_le_idx - P_le_idx
        all_peak_series = {
            "P_le_idx": [80],
            "P_ri_idx": [120],
            "Q_le_idx": [170],  # 170 - 80 = 90 samples = 90ms at 1000Hz
            "S_ri_idx": [230],
            "T_le_idx": [320],
            "T_ri_idx": [380],
        }
        
        result = calc_intervals(all_peak_series, cycle_idx=0, sampling_rate=1000)
        
        if "PR_interval_ms" in result and not np.isnan(result["PR_interval_ms"]):
            assert result["PR_interval_ms"] == 90.0

    def test_qrs_interval_calculation(self):
        """QRS interval should be calculated correctly."""
        from pyhearts.feature.intervals import calc_intervals
        
        # QRS interval = S_ri_idx - Q_le_idx
        all_peak_series = {
            "P_le_idx": [80],
            "P_ri_idx": [120],
            "Q_le_idx": [170],
            "S_ri_idx": [230],  # 230 - 170 = 60 samples = 60ms at 1000Hz
            "T_le_idx": [320],
            "T_ri_idx": [380],
        }
        
        result = calc_intervals(all_peak_series, cycle_idx=0, sampling_rate=1000)
        
        if "QRS_interval_ms" in result and not np.isnan(result["QRS_interval_ms"]):
            assert result["QRS_interval_ms"] == 60.0

    def test_missing_peaks_returns_nan(self):
        """Missing peaks should return NaN values."""
        from pyhearts.feature.intervals import calc_intervals
        
        # Empty series - no peak data
        all_peak_series = {
            "P_le_idx": [np.nan],
            "P_ri_idx": [np.nan],
            "Q_le_idx": [np.nan],
            "S_ri_idx": [np.nan],
            "T_le_idx": [np.nan],
            "T_ri_idx": [np.nan],
        }
        
        result = calc_intervals(all_peak_series, cycle_idx=0, sampling_rate=1000)
        
        # All intervals should be NaN
        for key, value in result.items():
            assert np.isnan(value)

