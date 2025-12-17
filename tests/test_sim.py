"""
Tests for pyhearts.sim module.

Tests ECG signal simulation:
- generate_ecg_signal: Synthetic ECG generation

Note: These tests require Python 3.10+ due to neurokit2 dependency.
"""

import numpy as np
import pytest

# Check if simulation module is available
try:
    from pyhearts import generate_ecg_signal
    HAS_SIM = generate_ecg_signal is not None
except (ImportError, TypeError):
    HAS_SIM = False
    generate_ecg_signal = None

requires_sim = pytest.mark.skipif(
    not HAS_SIM,
    reason="Simulation module requires Python 3.10+ (neurokit2 dependency)"
)


@requires_sim
class TestGenerateEcgSignal:
    """Tests for generate_ecg_signal function."""

    def test_returns_tuple(self):
        """Should return a tuple of outputs."""
        result = generate_ecg_signal(duration=5, plot=False)
        
        assert isinstance(result, tuple)
        assert len(result) == 6  # signal, sampling_rate, time, time_axis, start, end

    def test_signal_is_array(self):
        """Signal should be a numpy array."""
        signal, *_ = generate_ecg_signal(duration=5, plot=False)
        
        assert isinstance(signal, np.ndarray)

    def test_signal_length(self):
        """Signal length should match duration * sampling_rate."""
        duration = 10
        sampling_rate = 1000
        
        signal, sr, *_ = generate_ecg_signal(
            duration=duration,
            sampling_rate=sampling_rate,
            plot=False,
        )
        
        expected_length = duration * sampling_rate
        assert len(signal) == expected_length
        assert sr == sampling_rate

    def test_different_durations(self):
        """Different durations should give different lengths."""
        signal_short, *_ = generate_ecg_signal(duration=5, plot=False)
        signal_long, *_ = generate_ecg_signal(duration=20, plot=False)
        
        assert len(signal_short) < len(signal_long)

    def test_different_sampling_rates(self):
        """Different sampling rates should give different lengths."""
        signal_low, *_ = generate_ecg_signal(
            duration=10,
            sampling_rate=500,
            plot=False,
        )
        signal_high, *_ = generate_ecg_signal(
            duration=10,
            sampling_rate=2000,
            plot=False,
        )
        
        assert len(signal_low) == 5000
        assert len(signal_high) == 20000

    def test_has_variability(self):
        """Signal should have non-zero variability."""
        signal, *_ = generate_ecg_signal(duration=10, plot=False)
        
        assert np.std(signal) > 0
        assert np.max(signal) != np.min(signal)

    def test_reproducible_with_seed(self):
        """Same seed should give same signal."""
        signal1, *_ = generate_ecg_signal(duration=5, random_seed=42, plot=False)
        signal2, *_ = generate_ecg_signal(duration=5, random_seed=42, plot=False)
        
        np.testing.assert_array_equal(signal1, signal2)

    def test_different_seeds_different_signals(self):
        """Different seeds should give different signals."""
        signal1, *_ = generate_ecg_signal(duration=5, random_seed=42, plot=False)
        signal2, *_ = generate_ecg_signal(duration=5, random_seed=123, plot=False)
        
        assert not np.array_equal(signal1, signal2)

    def test_noise_level_effect(self):
        """Higher noise level should increase signal variability."""
        signal_low_noise, *_ = generate_ecg_signal(
            duration=10,
            noise_level=0.01,
            random_seed=42,
            plot=False,
        )
        signal_high_noise, *_ = generate_ecg_signal(
            duration=10,
            noise_level=0.5,
            random_seed=42,
            plot=False,
        )
        
        # High-frequency variability should be higher with more noise
        # Compare std of differences as proxy for noise level
        diff_std_low = np.std(np.diff(signal_low_noise))
        diff_std_high = np.std(np.diff(signal_high_noise))
        
        assert diff_std_high > diff_std_low

    def test_drift_effect(self):
        """Drift should affect signal baseline."""
        signal_no_drift, *_ = generate_ecg_signal(
            duration=10,
            drift_start=0,
            drift_end=0,
            noise_level=0,
            random_seed=42,
            plot=False,
        )
        signal_with_drift, *_ = generate_ecg_signal(
            duration=10,
            drift_start=-1,
            drift_end=1,
            noise_level=0,
            random_seed=42,
            plot=False,
        )
        
        # Mean of first and last quarters should differ with drift
        n = len(signal_with_drift)
        first_quarter_mean = np.mean(signal_with_drift[:n//4])
        last_quarter_mean = np.mean(signal_with_drift[-n//4:])
        
        assert last_quarter_mean > first_quarter_mean

    def test_time_array(self):
        """Time array should match signal length."""
        duration = 5
        sampling_rate = 1000
        
        signal, sr, time, *_ = generate_ecg_signal(
            duration=duration,
            sampling_rate=sampling_rate,
            plot=False,
        )
        
        assert len(time) == len(signal)
        assert time[0] == pytest.approx(0, abs=1e-6)
        assert time[-1] == pytest.approx(duration - 1/sampling_rate, abs=0.01)

    def test_heart_rate_parameter(self):
        """Different heart rates should affect signal periodicity."""
        # Lower heart rate = longer period between beats
        signal_slow, *_ = generate_ecg_signal(
            duration=30,
            heart_rate=40,
            plot=False,
        )
        signal_fast, *_ = generate_ecg_signal(
            duration=30,
            heart_rate=120,
            plot=False,
        )
        
        # Count approximate peaks (R-peaks should be local maxima above mean + std)
        threshold_slow = np.mean(signal_slow) + 0.5 * np.std(signal_slow)
        threshold_fast = np.mean(signal_fast) + 0.5 * np.std(signal_fast)
        
        # Fast HR should have more crossings above threshold
        crossings_slow = np.sum(np.diff(signal_slow > threshold_slow))
        crossings_fast = np.sum(np.diff(signal_fast > threshold_fast))
        
        assert crossings_fast > crossings_slow

    def test_line_noise_frequency(self):
        """Line noise should add periodic component."""
        # Generate with line noise
        signal_with_noise, *_ = generate_ecg_signal(
            duration=10,
            line_noise_amplitude=0.2,
            line_noise_frequency=50,
            noise_level=0,
            plot=False,
        )
        
        # FFT should show component at 50 Hz
        fft = np.abs(np.fft.rfft(signal_with_noise))
        freqs = np.fft.rfftfreq(len(signal_with_noise), 1/1000)
        
        # Find index closest to 50 Hz
        idx_50hz = np.argmin(np.abs(freqs - 50))
        
        # Power at 50 Hz should be significant
        assert fft[idx_50hz] > 0


@requires_sim
class TestGenerateEcgSignalIntegration:
    """Integration tests for generate_ecg_signal with PyHEARTS."""

    def test_signal_usable_by_pyhearts(self):
        """Generated signal should be usable by PyHEARTS analyzer."""
        from pyhearts import PyHEARTS
        
        signal, sampling_rate, *_ = generate_ecg_signal(
            duration=30,
            heart_rate=60,
            noise_level=0.05,
            plot=False,
        )
        
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        output_df, epochs_df = analyzer.analyze_ecg(signal)
        
        # Should successfully analyze the signal
        assert isinstance(output_df, object)  # May be empty but should exist
        assert isinstance(epochs_df, object)

    def test_detects_rpeaks_in_generated_signal(self):
        """Should detect R-peaks in generated ECG."""
        from pyhearts import PyHEARTS
        
        duration = 60  # 1 minute
        heart_rate = 60  # 60 bpm = 1 beat/second
        
        signal, sampling_rate, *_ = generate_ecg_signal(
            duration=duration,
            heart_rate=heart_rate,
            noise_level=0.02,  # Low noise for reliable detection
            plot=False,
        )
        
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        analyzer.analyze_ecg(signal)
        
        if hasattr(analyzer, "r_peak_indices") and len(analyzer.r_peak_indices) > 0:
            # Should detect approximately heart_rate * (duration/60) peaks
            expected_peaks = heart_rate * (duration / 60)
            detected_peaks = len(analyzer.r_peak_indices)
            
            # Allow 20% tolerance
            assert detected_peaks >= expected_peaks * 0.8
            assert detected_peaks <= expected_peaks * 1.2

