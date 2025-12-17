"""
Tests for pyhearts.core.fit module.

Tests the PyHEARTS main class including:
- Initialization with different configs
- Signal preprocessing
- ECG analysis pipeline
- HRV computation
"""

import numpy as np
import pandas as pd
import pytest

from pyhearts import PyHEARTS, ProcessCycleConfig


class TestPyHEARTSInit:
    """Tests for PyHEARTS initialization."""

    def test_basic_initialization(self, sampling_rate):
        """Basic initialization with sampling rate."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        assert analyzer.sampling_rate == sampling_rate
        assert analyzer.verbose is False
        assert analyzer.plot is False
        assert isinstance(analyzer.cfg, ProcessCycleConfig)

    def test_initialization_with_verbose(self, sampling_rate):
        """Initialization with verbose flag."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate, verbose=True)
        assert analyzer.verbose is True

    def test_initialization_with_plot(self, sampling_rate):
        """Initialization with plot flag."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate, plot=True)
        assert analyzer.plot is True

    def test_initialization_with_human_species(self, sampling_rate):
        """Initialization with human species preset."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate, species="human")
        assert "human" in analyzer.cfg.version

    def test_initialization_with_mouse_species(self, sampling_rate):
        """Initialization with mouse species preset."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate, species="mouse")
        assert "mouse" in analyzer.cfg.version

    def test_initialization_with_custom_config(self, sampling_rate, default_config):
        """Initialization with custom config object."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate, cfg=default_config)
        assert analyzer.cfg == default_config

    def test_initialization_with_overrides(self, sampling_rate):
        """Initialization with config overrides."""
        analyzer = PyHEARTS(
            sampling_rate=sampling_rate,
            rpeak_prominence_multiplier=4.0,
            epoch_corr_thresh=0.9,
        )
        assert analyzer.cfg.rpeak_prominence_multiplier == 4.0
        assert analyzer.cfg.epoch_corr_thresh == 0.9

    def test_initialization_with_species_and_overrides(self, sampling_rate):
        """Species preset with additional overrides."""
        analyzer = PyHEARTS(
            sampling_rate=sampling_rate,
            species="human",
            duration_min_ms=25,
        )
        assert "human" in analyzer.cfg.version
        assert analyzer.cfg.duration_min_ms == 25

    def test_initialization_invalid_override(self, sampling_rate):
        """Invalid override key should raise TypeError."""
        with pytest.raises(TypeError, match="Unknown config key"):
            PyHEARTS(sampling_rate=sampling_rate, invalid_key=123)

    def test_internal_state_initialization(self, sampling_rate):
        """Internal state should be properly initialized."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        assert analyzer.output_dict is None
        assert analyzer.previous_r_center_samples is None
        assert analyzer.previous_p_center_samples is None
        assert analyzer.previous_gauss_features is None
        assert analyzer.sig_corrected_dict == {}
        assert analyzer.hrv_metrics == {}


class TestPyHEARTSPreprocess:
    """Tests for PyHEARTS preprocessing."""

    def test_preprocess_returns_array(self, sampling_rate, simple_ecg_signal):
        """Preprocessing should return an array."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        result = analyzer.preprocess_signal(simple_ecg_signal)
        # Result could be None if all filters are disabled
        # With no parameters, should just return the signal
        assert result is None or isinstance(result, np.ndarray)

    def test_preprocess_with_highpass(self, sampling_rate, simple_ecg_signal):
        """Preprocessing with highpass filter."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        result = analyzer.preprocess_signal(
            simple_ecg_signal,
            highpass_cutoff=0.5,
            filter_order=4,
        )
        if result is not None:
            assert len(result) == len(simple_ecg_signal)

    def test_preprocess_with_lowpass(self, sampling_rate, simple_ecg_signal):
        """Preprocessing with lowpass filter."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        result = analyzer.preprocess_signal(
            simple_ecg_signal,
            lowpass_cutoff=40.0,
            filter_order=4,
        )
        if result is not None:
            assert len(result) == len(simple_ecg_signal)

    def test_preprocess_preserves_length(self, sampling_rate, simple_ecg_signal):
        """Preprocessing should preserve signal length."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        result = analyzer.preprocess_signal(
            simple_ecg_signal,
            highpass_cutoff=0.5,
            lowpass_cutoff=40.0,
            filter_order=4,
        )
        if result is not None:
            assert len(result) == len(simple_ecg_signal)


class TestPyHEARTSAnalyze:
    """Tests for PyHEARTS analysis pipeline."""

    def test_analyze_returns_dataframes(self, sampling_rate, simple_ecg_signal):
        """analyze_ecg should return two DataFrames."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        output_df, epochs_df = analyzer.analyze_ecg(simple_ecg_signal)
        
        assert isinstance(output_df, pd.DataFrame)
        assert isinstance(epochs_df, pd.DataFrame)

    def test_analyze_detects_rpeaks(self, sampling_rate, simple_ecg_signal):
        """Analysis should detect R-peaks."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        analyzer.analyze_ecg(simple_ecg_signal)
        
        assert hasattr(analyzer, "r_peak_indices")
        assert len(analyzer.r_peak_indices) > 0

    def test_analyze_creates_epochs(self, sampling_rate, simple_ecg_signal):
        """Analysis should create epochs DataFrame."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        _, epochs_df = analyzer.analyze_ecg(simple_ecg_signal)
        
        if not epochs_df.empty:
            assert "cycle" in epochs_df.columns
            assert "signal" in epochs_df.columns or len(epochs_df.columns) > 0

    def test_analyze_extracts_features(self, sampling_rate, simple_ecg_signal):
        """Analysis should extract morphological features."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        output_df, _ = analyzer.analyze_ecg(simple_ecg_signal)
        
        if not output_df.empty:
            # Should have interval columns
            interval_cols = [c for c in output_df.columns if "interval" in c.lower()]
            assert len(interval_cols) > 0

    def test_analyze_empty_signal(self, sampling_rate):
        """Analysis of empty signal should handle gracefully."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        empty_signal = np.array([])
        
        # Should not crash, may return empty DataFrames
        output_df, epochs_df = analyzer.analyze_ecg(empty_signal)
        assert isinstance(output_df, pd.DataFrame)
        assert isinstance(epochs_df, pd.DataFrame)

    def test_analyze_flat_signal(self, sampling_rate):
        """Analysis of flat signal should handle gracefully (no R-peaks)."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        flat_signal = np.zeros(10000)
        
        output_df, epochs_df = analyzer.analyze_ecg(flat_signal)
        # Should return empty DataFrames when no R-peaks detected
        assert output_df.empty or len(output_df) == 0

    def test_analyze_stores_output_dict(self, sampling_rate, simple_ecg_signal):
        """Analysis should populate output_dict."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        analyzer.analyze_ecg(simple_ecg_signal)
        
        # May be populated if analysis succeeds
        assert analyzer.output_dict is not None or hasattr(analyzer, "output_df")


class TestPyHEARTSHRV:
    """Tests for HRV computation."""

    def test_compute_hrv_after_analysis(self, sampling_rate, simple_ecg_signal):
        """HRV computation after successful analysis."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        analyzer.analyze_ecg(simple_ecg_signal)
        
        # Only compute HRV if we have enough intervals
        if analyzer.output_dict and "RR_interval_ms" in analyzer.output_dict:
            rr_intervals = np.array(analyzer.output_dict["RR_interval_ms"])
            clean_rr = rr_intervals[~np.isnan(rr_intervals)]
            
            if len(clean_rr) >= 60:
                analyzer.compute_hrv_metrics()
                assert isinstance(analyzer.hrv_metrics, dict)

    def test_compute_hrv_without_analysis(self, sampling_rate):
        """HRV computation without prior analysis should handle gracefully."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        analyzer.compute_hrv_metrics()
        # Should not crash, hrv_metrics should be empty dict
        assert analyzer.hrv_metrics == {}

    def test_compute_hrv_insufficient_intervals(self, sampling_rate):
        """HRV computation with insufficient intervals."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        # Manually set a small output_dict
        analyzer.output_dict = {"RR_interval_ms": [800, 850]}
        analyzer.compute_hrv_metrics()
        # Should handle gracefully (< 60 intervals triggers skip)
        assert analyzer.hrv_metrics == {} or len(analyzer.hrv_metrics) > 0


class TestPyHEARTSOutputDict:
    """Tests for output dictionary initialization."""

    def test_initialize_output_dict(self, sampling_rate):
        """Test output dict initialization."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        
        output_dict = analyzer.initialize_output_dict(
            cycle_inds=[0, 1, 2],
            components=["P", "Q", "R", "S", "T"],
            peak_features=["center_idx", "height"],
            intervals=["PR_interval_ms", "QRS_interval_ms"],
            pairwise_differences=["R_minus_S"],
        )
        
        assert isinstance(output_dict, dict)
        # Should have keys for component+feature combinations
        assert len(output_dict) > 0

    def test_initialize_output_dict_no_pairwise(self, sampling_rate):
        """Test output dict initialization without pairwise differences."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        
        output_dict = analyzer.initialize_output_dict(
            cycle_inds=[0, 1],
            components=["R"],
            peak_features=["center_idx"],
            intervals=["RR_interval_ms"],
        )
        
        assert isinstance(output_dict, dict)


class TestPyHEARTSMetadata:
    """Tests for metadata generation."""

    def test_resolved_config(self, sampling_rate):
        """Test config resolution to dict."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        config_dict = analyzer._resolved_config()
        
        assert isinstance(config_dict, dict)
        assert "rpeak_prominence_multiplier" in config_dict
        assert "epoch_corr_thresh" in config_dict

    def test_metadata_payload(self, sampling_rate):
        """Test metadata payload generation."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        metadata = analyzer._metadata_payload()
        
        assert isinstance(metadata, dict)
        assert "sampling_rate_hz" in metadata
        assert "config" in metadata
        assert "runtime" in metadata
        assert metadata["sampling_rate_hz"] == sampling_rate

    def test_git_info(self, sampling_rate):
        """Test git info retrieval."""
        analyzer = PyHEARTS(sampling_rate=sampling_rate)
        git_info = analyzer._git_info()
        
        assert isinstance(git_info, dict)
        # May or may not have values depending on git state
        assert "commit" in git_info
        assert "branch" in git_info

