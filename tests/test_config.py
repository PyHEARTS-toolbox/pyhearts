"""
Tests for pyhearts.config module.

Tests ProcessCycleConfig dataclass including:
- Default instantiation
- Species presets (human, mouse)
- Validation logic
- Field overrides
"""

import pytest
from dataclasses import replace

from pyhearts.config import ProcessCycleConfig


class TestProcessCycleConfigDefaults:
    """Test default ProcessCycleConfig instantiation."""

    def test_default_instantiation(self):
        """Default config should instantiate without errors."""
        cfg = ProcessCycleConfig()
        assert cfg is not None

    def test_default_values(self):
        """Check key default values are set correctly."""
        cfg = ProcessCycleConfig()
        
        # R-peak detection defaults
        assert cfg.rpeak_prominence_multiplier == 3.0
        assert cfg.rpeak_min_refrac_ms == 100.0
        assert cfg.rpeak_rr_frac_second_pass == 0.55
        assert cfg.rpeak_bpm_bounds == (40.0, 900.0)
        
        # Epoching defaults
        assert cfg.epoch_corr_thresh == 0.80
        assert cfg.epoch_var_thresh == 5.0
        
        # Wavelet defaults
        assert cfg.wavelet_name == "db6"
        assert cfg.wavelet_detail_level == 3
        
        # Version tag
        assert cfg.version == "v1"

    def test_frozen_dataclass(self):
        """Config should be frozen (immutable)."""
        cfg = ProcessCycleConfig()
        with pytest.raises(AttributeError):
            cfg.rpeak_prominence_multiplier = 5.0

    def test_default_amp_min_ratio(self):
        """Default amplitude ratios should be set."""
        cfg = ProcessCycleConfig()
        assert "P" in cfg.amp_min_ratio
        assert "T" in cfg.amp_min_ratio
        assert "Q" in cfg.amp_min_ratio
        assert "S" in cfg.amp_min_ratio
        assert all(0 <= v < 1 for v in cfg.amp_min_ratio.values())

    def test_default_snr_settings(self):
        """SNR gate settings should be initialized."""
        cfg = ProcessCycleConfig()
        assert "P" in cfg.snr_mad_multiplier
        assert "T" in cfg.snr_mad_multiplier
        assert cfg.snr_mad_multiplier["P"] > 0
        assert cfg.snr_mad_multiplier["T"] > 0


class TestProcessCycleConfigPresets:
    """Test species-specific presets."""

    def test_human_preset_instantiation(self):
        """Human preset should instantiate without errors."""
        cfg = ProcessCycleConfig.for_human()
        assert cfg is not None
        assert "human" in cfg.version

    def test_mouse_preset_instantiation(self):
        """Mouse preset should instantiate without errors."""
        cfg = ProcessCycleConfig.for_mouse()
        assert cfg is not None
        assert "mouse" in cfg.version

    def test_human_preset_rr_bounds(self):
        """Human preset should have appropriate RR bounds."""
        cfg = ProcessCycleConfig.for_human()
        lo, hi = cfg.rr_bounds_ms
        # Human: ~30-200 bpm → 300-2000ms RR
        assert lo >= 250  # At least 240 bpm max
        assert hi <= 2500  # At least 24 bpm min

    def test_mouse_preset_rr_bounds(self):
        """Mouse preset should have faster RR bounds."""
        cfg = ProcessCycleConfig.for_mouse()
        lo, hi = cfg.rr_bounds_ms
        # Mouse: ~240-900 bpm → 67-250ms RR
        assert lo < 150  # Faster than human
        assert hi <= 300  # Much faster max HR

    def test_human_preset_duration_min(self):
        """Human preset should have larger minimum duration."""
        cfg = ProcessCycleConfig.for_human()
        assert cfg.duration_min_ms >= 15  # Human waves are wider

    def test_mouse_preset_duration_min(self):
        """Mouse preset should have smaller minimum duration."""
        cfg = ProcessCycleConfig.for_mouse()
        assert cfg.duration_min_ms < 10  # Mouse waves are narrower

    def test_presets_differ_from_default(self):
        """Presets should differ from default in key ways."""
        default = ProcessCycleConfig()
        human = ProcessCycleConfig.for_human()
        mouse = ProcessCycleConfig.for_mouse()
        
        # Presets should have different RR bounds
        assert human.rr_bounds_ms != default.rr_bounds_ms
        assert mouse.rr_bounds_ms != default.rr_bounds_ms
        assert human.rr_bounds_ms != mouse.rr_bounds_ms


class TestProcessCycleConfigValidation:
    """Test configuration validation logic."""

    def test_invalid_epoch_corr_thresh_high(self):
        """epoch_corr_thresh > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="epoch_corr_thresh"):
            replace(ProcessCycleConfig(), epoch_corr_thresh=1.5)

    def test_invalid_epoch_corr_thresh_low(self):
        """epoch_corr_thresh < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="epoch_corr_thresh"):
            replace(ProcessCycleConfig(), epoch_corr_thresh=-0.1)

    def test_invalid_epoch_var_thresh(self):
        """epoch_var_thresh <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="epoch_var_thresh"):
            replace(ProcessCycleConfig(), epoch_var_thresh=0)
        with pytest.raises(ValueError, match="epoch_var_thresh"):
            replace(ProcessCycleConfig(), epoch_var_thresh=-1)

    def test_invalid_bound_factor_high(self):
        """bound_factor >= 1 should raise ValueError."""
        with pytest.raises(ValueError, match="bound_factor"):
            replace(ProcessCycleConfig(), bound_factor=1.0)

    def test_invalid_bound_factor_low(self):
        """bound_factor <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="bound_factor"):
            replace(ProcessCycleConfig(), bound_factor=0.0)

    def test_invalid_maxfev(self):
        """maxfev <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="maxfev"):
            replace(ProcessCycleConfig(), maxfev=0)

    def test_invalid_wavelet_offset_order(self):
        """wavelet_base_offset_ms >= wavelet_max_offset_ms should raise."""
        with pytest.raises(ValueError, match="wavelet"):
            replace(ProcessCycleConfig(), wavelet_base_offset_ms=100, wavelet_max_offset_ms=50)

    def test_invalid_rr_bounds(self):
        """Invalid RR bounds should raise ValueError."""
        with pytest.raises(ValueError, match="rr_bounds_ms"):
            replace(ProcessCycleConfig(), rr_bounds_ms=(1000, 500))  # lo > hi
        with pytest.raises(ValueError, match="rr_bounds_ms"):
            replace(ProcessCycleConfig(), rr_bounds_ms=(-100, 1000))  # negative

    def test_invalid_threshold_fraction(self):
        """threshold_fraction outside (0,1) should raise."""
        with pytest.raises(ValueError, match="threshold_fraction"):
            replace(ProcessCycleConfig(), threshold_fraction=0)
        with pytest.raises(ValueError, match="threshold_fraction"):
            replace(ProcessCycleConfig(), threshold_fraction=1.0)

    def test_invalid_sharp_stat(self):
        """Invalid sharp_stat option should raise."""
        with pytest.raises(ValueError, match="sharp_stat"):
            replace(ProcessCycleConfig(), sharp_stat="invalid")

    def test_invalid_sharp_amp_norm(self):
        """Invalid sharp_amp_norm option should raise."""
        with pytest.raises(ValueError, match="sharp_amp_norm"):
            replace(ProcessCycleConfig(), sharp_amp_norm="invalid")

    def test_invalid_shape_diff_mode(self):
        """Invalid shape_diff_mode should raise."""
        with pytest.raises(ValueError, match="shape_diff_mode"):
            replace(ProcessCycleConfig(), shape_diff_mode="invalid")

    def test_invalid_savgol_window_even(self):
        """Even savgol_window_pts should raise."""
        with pytest.raises(ValueError, match="savgol_window_pts"):
            replace(ProcessCycleConfig(), savgol_window_pts=6)

    def test_invalid_savgol_window_small(self):
        """savgol_window_pts < 3 should raise."""
        with pytest.raises(ValueError, match="savgol_window_pts"):
            replace(ProcessCycleConfig(), savgol_window_pts=2)

    def test_invalid_savgol_polyorder(self):
        """savgol_polyorder >= savgol_window_pts should raise."""
        with pytest.raises(ValueError, match="savgol_polyorder"):
            replace(ProcessCycleConfig(), savgol_window_pts=7, savgol_polyorder=8)

    def test_invalid_rpeak_bpm_bounds(self):
        """Invalid rpeak_bpm_bounds should raise."""
        with pytest.raises(ValueError, match="rpeak_bpm_bounds"):
            replace(ProcessCycleConfig(), rpeak_bpm_bounds=(200, 100))  # lo > hi

    def test_invalid_rpeak_prominence_multiplier(self):
        """rpeak_prominence_multiplier <= 0 should raise."""
        with pytest.raises(ValueError, match="rpeak_prominence_multiplier"):
            replace(ProcessCycleConfig(), rpeak_prominence_multiplier=0)


class TestProcessCycleConfigOverrides:
    """Test field-level overrides."""

    def test_single_field_override(self):
        """Single field override should work."""
        cfg = replace(ProcessCycleConfig(), rpeak_prominence_multiplier=4.0)
        assert cfg.rpeak_prominence_multiplier == 4.0

    def test_multiple_field_override(self):
        """Multiple field overrides should work."""
        cfg = replace(
            ProcessCycleConfig(),
            rpeak_prominence_multiplier=4.0,
            epoch_corr_thresh=0.9,
            maxfev=5000,
        )
        assert cfg.rpeak_prominence_multiplier == 4.0
        assert cfg.epoch_corr_thresh == 0.9
        assert cfg.maxfev == 5000

    def test_preset_with_override(self):
        """Overriding preset values should work."""
        base = ProcessCycleConfig.for_human()
        cfg = replace(base, duration_min_ms=30)
        assert cfg.duration_min_ms == 30
        assert "human" in cfg.version  # Other preset values preserved

    def test_valid_amp_min_ratio_override(self):
        """Valid amp_min_ratio override should work."""
        cfg = replace(
            ProcessCycleConfig(),
            amp_min_ratio={"P": 0.05, "T": 0.08, "Q": 0.03, "S": 0.03}
        )
        assert cfg.amp_min_ratio["P"] == 0.05

    def test_invalid_amp_min_ratio_key(self):
        """Invalid key in amp_min_ratio should raise."""
        with pytest.raises(ValueError, match="amp_min_ratio"):
            replace(
                ProcessCycleConfig(),
                amp_min_ratio={"X": 0.05, "T": 0.08, "Q": 0.03, "S": 0.03}
            )

    def test_invalid_amp_min_ratio_value(self):
        """Value >= 1 in amp_min_ratio should raise."""
        with pytest.raises(ValueError, match="amp_min_ratio"):
            replace(
                ProcessCycleConfig(),
                amp_min_ratio={"P": 1.5, "T": 0.08, "Q": 0.03, "S": 0.03}
            )

