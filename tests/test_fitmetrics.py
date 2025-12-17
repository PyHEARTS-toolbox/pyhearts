"""
Tests for pyhearts.fitmetrics module.

Tests fit quality metrics:
- calc_r_squared: Coefficient of determination
- calc_rmse: Root mean squared error
"""

import numpy as np
import pytest

from pyhearts.fitmetrics import calc_r_squared, calc_rmse


class TestCalcRSquared:
    """Tests for calc_r_squared function."""

    def test_perfect_fit(self):
        """Identical arrays should give R² = 1.0."""
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fit = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r_squared = calc_r_squared(sig, fit)
        assert np.isclose(r_squared, 1.0, atol=1e-10)

    def test_perfect_linear_relationship(self):
        """Perfectly correlated arrays (scaled) should give R² = 1.0."""
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fit = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # sig * 2
        r_squared = calc_r_squared(sig, fit)
        assert np.isclose(r_squared, 1.0, atol=1e-10)

    def test_no_correlation(self):
        """Uncorrelated arrays should give R² close to 0."""
        np.random.seed(42)
        sig = np.random.randn(1000)
        fit = np.random.randn(1000)
        r_squared = calc_r_squared(sig, fit)
        assert r_squared < 0.1  # Should be close to 0 for uncorrelated data

    def test_negative_correlation(self):
        """Negatively correlated arrays should still give positive R²."""
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fit = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Perfectly negatively correlated
        r_squared = calc_r_squared(sig, fit)
        assert np.isclose(r_squared, 1.0, atol=1e-10)

    def test_partial_correlation(self):
        """Partially correlated arrays should give intermediate R²."""
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fit = np.array([1.1, 1.9, 3.2, 3.8, 5.1])  # Slight deviations
        r_squared = calc_r_squared(sig, fit)
        assert 0.9 < r_squared < 1.0

    def test_r_squared_range(self):
        """R² should always be between 0 and 1."""
        np.random.seed(42)
        for _ in range(10):
            sig = np.random.randn(50)
            fit = np.random.randn(50)
            r_squared = calc_r_squared(sig, fit)
            assert 0.0 <= r_squared <= 1.0

    def test_with_offset(self):
        """Linear relationship with offset should give R² = 1.0."""
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fit = sig + 10  # Constant offset
        r_squared = calc_r_squared(sig, fit)
        assert np.isclose(r_squared, 1.0, atol=1e-10)

    def test_sinusoidal_fit(self):
        """Test with realistic signal-like data."""
        t = np.linspace(0, 2 * np.pi, 100)
        sig = np.sin(t)
        fit = np.sin(t) + 0.1 * np.random.randn(100)  # Slightly noisy fit
        np.random.seed(42)
        r_squared = calc_r_squared(sig, fit)
        assert r_squared > 0.8  # Should still be good fit


class TestCalcRMSE:
    """Tests for calc_rmse function."""

    def test_perfect_fit(self):
        """Identical arrays should give RMSE = 0."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rmse = calc_rmse(observed, predicted)
        assert np.isclose(rmse, 0.0, atol=1e-10)

    def test_constant_error(self):
        """Constant error should give RMSE = |error|."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([2.0, 3.0, 4.0, 5.0, 6.0])  # All off by 1
        rmse = calc_rmse(observed, predicted)
        assert np.isclose(rmse, 1.0, atol=1e-10)

    def test_variable_error(self):
        """Test RMSE calculation with variable errors."""
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.0, 2.0, 4.0, 4.0])  # Errors: 0, 0, 1, 0
        # MSE = (0 + 0 + 1 + 0) / 4 = 0.25
        # RMSE = sqrt(0.25) = 0.5
        rmse = calc_rmse(observed, predicted)
        assert np.isclose(rmse, 0.5, atol=1e-10)

    def test_rmse_always_positive(self):
        """RMSE should always be non-negative."""
        np.random.seed(42)
        for _ in range(10):
            observed = np.random.randn(50)
            predicted = np.random.randn(50)
            rmse = calc_rmse(observed, predicted)
            assert rmse >= 0.0

    def test_symmetric(self):
        """RMSE should be symmetric (order shouldn't matter)."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        rmse1 = calc_rmse(observed, predicted)
        rmse2 = calc_rmse(predicted, observed)
        assert np.isclose(rmse1, rmse2, atol=1e-10)

    def test_scale_sensitivity(self):
        """RMSE should scale with error magnitude."""
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted_small = observed + 0.1  # Small error
        predicted_large = observed + 1.0  # Larger error
        
        rmse_small = calc_rmse(observed, predicted_small)
        rmse_large = calc_rmse(observed, predicted_large)
        
        assert rmse_small < rmse_large
        assert np.isclose(rmse_large / rmse_small, 10.0, atol=1e-10)

    def test_known_value(self):
        """Test with known RMSE calculation."""
        observed = np.array([3.0, -0.5, 2.0, 7.0])
        predicted = np.array([2.5, 0.0, 2.0, 8.0])
        # Errors: 0.5, -0.5, 0, -1
        # Squared errors: 0.25, 0.25, 0, 1
        # MSE = 1.5 / 4 = 0.375
        # RMSE = sqrt(0.375) ≈ 0.612
        rmse = calc_rmse(observed, predicted)
        expected = np.sqrt(0.375)
        assert np.isclose(rmse, expected, atol=1e-10)

    def test_single_element(self):
        """RMSE should work with single element arrays."""
        observed = np.array([5.0])
        predicted = np.array([3.0])
        rmse = calc_rmse(observed, predicted)
        assert np.isclose(rmse, 2.0, atol=1e-10)


class TestFitMetricsIntegration:
    """Integration tests combining R² and RMSE."""

    def test_good_fit_metrics(self):
        """Good fit should have high R² and low RMSE."""
        np.random.seed(42)
        observed = np.sin(np.linspace(0, 4 * np.pi, 100))
        noise = 0.05 * np.random.randn(100)
        predicted = observed + noise
        
        r_squared = calc_r_squared(observed, predicted)
        rmse = calc_rmse(observed, predicted)
        
        assert r_squared > 0.95
        assert rmse < 0.1

    def test_poor_fit_metrics(self):
        """Poor fit should have low R² and high RMSE."""
        np.random.seed(42)
        observed = np.sin(np.linspace(0, 4 * np.pi, 100))
        predicted = np.cos(np.linspace(0, 4 * np.pi, 100))  # Wrong phase
        
        r_squared = calc_r_squared(observed, predicted)
        rmse = calc_rmse(observed, predicted)
        
        # Phase-shifted sine/cosine have R² = 0 at 90° shift
        assert r_squared < 0.5

    def test_ecg_like_waveform(self):
        """Test with ECG-like waveform."""
        t = np.linspace(0, 1, 500)
        
        # Simulated ECG cycle
        observed = (
            1.0 * np.exp(-((t - 0.5) ** 2) / (2 * 0.01 ** 2)) +  # R
            0.2 * np.exp(-((t - 0.3) ** 2) / (2 * 0.02 ** 2)) +  # P
            0.3 * np.exp(-((t - 0.7) ** 2) / (2 * 0.03 ** 2))    # T
        )
        
        # Fitted version with small errors
        np.random.seed(42)
        predicted = observed + 0.02 * np.random.randn(500)
        
        r_squared = calc_r_squared(observed, predicted)
        rmse = calc_rmse(observed, predicted)
        
        assert r_squared > 0.95  # Good but not perfect due to noise
        assert rmse < 0.05

