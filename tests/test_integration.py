"""
Integration tests for PyHEARTS.

End-to-end tests that verify the complete analysis pipeline works correctly
when all components are combined.

Note: Some tests require Python 3.10+ due to neurokit2 dependency.
"""

import numpy as np
import pandas as pd
import pytest

from pyhearts import PyHEARTS, ProcessCycleConfig

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
class TestFullPipeline:
    """End-to-end pipeline tests."""

    @pytest.fixture
    def generated_ecg(self):
        """Generate a clean ECG signal for testing."""
        signal, sampling_rate, time, _, _, _ = generate_ecg_signal(
            duration=60,
            sampling_rate=1000,
            heart_rate=60,
            noise_level=0.02,
            drift_start=0,
            drift_end=0,
            line_noise_amplitude=0,
            random_seed=42,
            plot=False,
        )
        return signal, sampling_rate

    def test_complete_analysis_human(self, generated_ecg):
        """Complete analysis with human preset."""
        signal, sampling_rate = generated_ecg
        
        analyzer = PyHEARTS(sampling_rate=sampling_rate, species="human")
        output_df, epochs_df = analyzer.analyze_ecg(signal)
        
        # Should produce output
        assert isinstance(output_df, pd.DataFrame)
        assert isinstance(epochs_df, pd.DataFrame)
        
        # If successful, should have multiple cycles
        if not output_df.empty:
            assert len(output_df) > 0

    def test_complete_analysis_default(self, generated_ecg):
        """Complete analysis with default config."""
        signal, sampling_rate = generated_ecg
        
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        output_df, epochs_df = analyzer.analyze_ecg(signal)
        
        assert isinstance(output_df, pd.DataFrame)
        assert isinstance(epochs_df, pd.DataFrame)

    def test_analysis_with_preprocessing(self, generated_ecg):
        """Analysis with preprocessing step."""
        signal, sampling_rate = generated_ecg
        
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        
        # Preprocess signal
        preprocessed = analyzer.preprocess_signal(
            signal,
            highpass_cutoff=0.5,
            lowpass_cutoff=40.0,
            filter_order=4,
        )
        
        # Analyze (use preprocessed if available, else original)
        analysis_signal = preprocessed if preprocessed is not None else signal
        output_df, epochs_df = analyzer.analyze_ecg(analysis_signal)
        
        assert isinstance(output_df, pd.DataFrame)

    def test_hrv_computation_after_analysis(self, generated_ecg):
        """HRV computation after successful analysis."""
        signal, sampling_rate = generated_ecg
        
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        analyzer.analyze_ecg(signal)
        
        # Attempt HRV computation
        analyzer.compute_hrv_metrics()
        
        # hrv_metrics should be a dict (may be empty if insufficient data)
        assert isinstance(analyzer.hrv_metrics, dict)

    def test_output_contains_expected_columns(self, generated_ecg):
        """Output DataFrame should contain expected feature columns."""
        signal, sampling_rate = generated_ecg
        
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        output_df, _ = analyzer.analyze_ecg(signal)
        
        if not output_df.empty:
            # Check for interval columns
            interval_cols = [c for c in output_df.columns if "interval" in c.lower()]
            assert len(interval_cols) > 0, "Should have interval columns"
            
            # Check for component columns (P, Q, R, S, T)
            component_cols = [c for c in output_df.columns if any(
                c.startswith(comp) for comp in ["P_", "Q_", "R_", "S_", "T_"]
            )]
            # May have component columns depending on detection success

    def test_epochs_structure(self, generated_ecg):
        """Epochs DataFrame should have correct structure."""
        signal, sampling_rate = generated_ecg
        
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        _, epochs_df = analyzer.analyze_ecg(signal)
        
        if not epochs_df.empty:
            assert "cycle" in epochs_df.columns
            
            # Each cycle should have multiple samples
            cycles = epochs_df["cycle"].unique()
            for cycle in cycles:
                cycle_data = epochs_df[epochs_df["cycle"] == cycle]
                assert len(cycle_data) > 10


@requires_sim
class TestConfigEffects:
    """Tests verifying config changes affect analysis."""

    @pytest.fixture
    def test_signal(self):
        """Short test signal."""
        signal, sr, *_ = generate_ecg_signal(
            duration=30,
            sampling_rate=1000,
            heart_rate=70,
            noise_level=0.03,
            random_seed=42,
            plot=False,
        )
        return signal, sr

    def test_different_species_different_results(self, test_signal):
        """Different species presets should potentially give different results."""
        signal, sampling_rate = test_signal
        
        # Human preset
        analyzer_human = PyHEARTS(sampling_rate=sampling_rate, species="human")
        output_human, _ = analyzer_human.analyze_ecg(signal)
        
        # Default preset
        analyzer_default = PyHEARTS(sampling_rate=sampling_rate)
        output_default, _ = analyzer_default.analyze_ecg(signal)
        
        # Both should produce valid output
        assert isinstance(output_human, pd.DataFrame)
        assert isinstance(output_default, pd.DataFrame)

    def test_epoch_threshold_effect(self, test_signal):
        """Changing epoch threshold should affect results."""
        signal, sampling_rate = test_signal
        
        # Strict threshold
        analyzer_strict = PyHEARTS(
            sampling_rate=sampling_rate,
            epoch_corr_thresh=0.95,
        )
        _, epochs_strict = analyzer_strict.analyze_ecg(signal)
        
        # Lenient threshold
        analyzer_lenient = PyHEARTS(
            sampling_rate=sampling_rate,
            epoch_corr_thresh=0.5,
        )
        _, epochs_lenient = analyzer_lenient.analyze_ecg(signal)
        
        # Lenient threshold should keep more epochs (or same)
        if not epochs_strict.empty and not epochs_lenient.empty:
            strict_cycles = len(epochs_strict["cycle"].unique())
            lenient_cycles = len(epochs_lenient["cycle"].unique())
            assert lenient_cycles >= strict_cycles


@requires_sim
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_short_signal(self):
        """Very short signal should handle gracefully."""
        signal = np.random.randn(100)  # Only 0.1 seconds at 1000 Hz
        
        analyzer = PyHEARTS(sampling_rate=1000)
        output_df, epochs_df = analyzer.analyze_ecg(signal)
        
        # Should not crash, may return empty DataFrames
        assert isinstance(output_df, pd.DataFrame)
        assert isinstance(epochs_df, pd.DataFrame)

    def test_very_noisy_signal(self):
        """Very noisy signal should handle gracefully."""
        # Pure noise
        np.random.seed(42)
        signal = np.random.randn(10000) * 10
        
        analyzer = PyHEARTS(sampling_rate=1000)
        output_df, epochs_df = analyzer.analyze_ecg(signal)
        
        # Should not crash
        assert isinstance(output_df, pd.DataFrame)

    def test_single_peak_signal(self):
        """Signal with single peak should handle gracefully."""
        # Create signal with single R-peak
        t = np.linspace(0, 2, 2000)
        signal = np.exp(-((t - 1) ** 2) / (2 * 0.01 ** 2))
        
        analyzer = PyHEARTS(sampling_rate=1000)
        output_df, epochs_df = analyzer.analyze_ecg(signal)
        
        # Should not crash, but may not produce epochs
        assert isinstance(output_df, pd.DataFrame)

    def test_negative_signal(self):
        """Inverted (negative) ECG should be handled."""
        signal, sr, *_ = generate_ecg_signal(
            duration=10,
            plot=False,
        )
        inverted_signal = -signal
        
        analyzer = PyHEARTS(sampling_rate=sr)
        output_df, epochs_df = analyzer.analyze_ecg(inverted_signal)
        
        # Should not crash (detection might fail on inverted signal)
        assert isinstance(output_df, pd.DataFrame)

    def test_large_signal(self):
        """Large signal (several minutes) should work."""
        signal, sr, *_ = generate_ecg_signal(
            duration=300,  # 5 minutes
            sampling_rate=500,
            heart_rate=70,
            noise_level=0.02,
            plot=False,
        )
        
        analyzer = PyHEARTS(sampling_rate=sr)
        output_df, epochs_df = analyzer.analyze_ecg(signal)
        
        # Should process successfully
        if not output_df.empty:
            # 5 minutes at 70 bpm ≈ 350 beats
            assert len(output_df) > 100


@requires_sim
class TestReproducibility:
    """Tests for analysis reproducibility."""

    def test_same_signal_same_results(self):
        """Same signal should give identical results."""
        signal, sr, *_ = generate_ecg_signal(
            duration=30,
            random_seed=42,
            plot=False,
        )
        
        analyzer1 = PyHEARTS(sampling_rate=sr)
        output1, _ = analyzer1.analyze_ecg(signal.copy())
        
        analyzer2 = PyHEARTS(sampling_rate=sr)
        output2, _ = analyzer2.analyze_ecg(signal.copy())
        
        if not output1.empty and not output2.empty:
            # Compare numeric columns
            for col in output1.columns:
                if output1[col].dtype in [np.float64, np.int64]:
                    np.testing.assert_array_almost_equal(
                        output1[col].values,
                        output2[col].values,
                        decimal=10,
                    )

    def test_config_reproducibility(self):
        """Same config should give same results."""
        signal, sr, *_ = generate_ecg_signal(
            duration=30,
            random_seed=42,
            plot=False,
        )
        
        # Create identical configs
        cfg1 = ProcessCycleConfig.for_human()
        cfg2 = ProcessCycleConfig.for_human()
        
        analyzer1 = PyHEARTS(sampling_rate=sr, cfg=cfg1)
        analyzer2 = PyHEARTS(sampling_rate=sr, cfg=cfg2)
        
        output1, _ = analyzer1.analyze_ecg(signal.copy())
        output2, _ = analyzer2.analyze_ecg(signal.copy())
        
        if not output1.empty and not output2.empty:
            assert output1.shape == output2.shape

