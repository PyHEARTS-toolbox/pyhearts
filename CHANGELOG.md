# Changelog

All notable changes to PyHEARTS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **QRS Removal for T-wave Detection**: New `qrs_removal.py` module implementing established QRS removal method
  - `remove_qrs_sigmoid()`: Replaces QRS complexes with sigmoid functions to reduce interference
  - `remove_baseline_wander()`: Two-stage baseline removal (750ms + 2000ms windows) for cleaner T-wave detection
  - Integrated into derivative-based peak detection pipeline when `t_wave_use_qrs_removal=True`
- **T-wave Detection Evaluation Tool**: `test_t_wave_accuracy.py` for comprehensive T-wave detection benchmarking against ECGpuwave and QTDB annotations

### Changed
- **T-wave Detection Improvements**:
  - QRS removal now applied before T-wave detection in derivative-based mode (mimics established delineation approach)
  - SNR gate now uses filtered/QRS-removed signal consistently for accurate SNR calculations
  - Relaxed T-wave SNR threshold from 1.5× to 0.8× MAD in `for_human()` preset for improved detection
  - Fixed precomputed peak mapping: now uses `index` column (global sample indices) instead of `signal_x` (relative time) for correct cycle-to-R-peak mapping
  - Precomputed T-wave onset/offset indices now properly preserved in `peak_data` and propagated to `output_df`
- **P and T Peak Detection Robustness**:
  - Fixed "too many values to unpack" error when Gaussian fitting arrays have unexpected shapes
  - Fixed incorrect return value (None) to proper 5-tuple format in `process_cycle()` error handling
  - Improved array unpacking to handle cases where `params_per_peak` may differ from expected (3 vs 4 columns)
- Renamed `plts/` module to `plots/` for clarity
- Renamed `objs/` module to `core/` for clarity  
- Renamed `feature/HRV.py` to `feature/hrv.py` to follow Python naming conventions

### Fixed
- **Critical Bug**: Precomputed T-wave annotations (onset/offset) were detected but not appearing in `output_df`
  - Root cause: Mapping from R-peak indices to cycle indices used wrong column (`signal_x` instead of `index`)
  - Impact: 0% T-wave detection rate in evaluation despite 100% internal detection
  - Resolution: Fixed mapping logic to use global sample indices, achieving 99.2% T-wave extraction rate
- **Critical Bug**: "too many values to unpack (expected 3)" error during cycle processing
  - Root cause: Transposing arrays with 4 columns when only 3 values expected
  - Impact: Many cycles failing to process, preventing T-wave extraction
  - Resolution: Changed to direct column indexing instead of transposition, with proper shape validation
- **SNR Gate Signal Mismatch**: SNR gate was using original signal while peaks detected in filtered signal
  - Fixed to use consistent signal source (filtered/QRS-removed) for both detection and SNR calculation
- **Return Value Consistency**: `process_cycle()` now always returns proper 5-tuple format instead of `None` in error cases

### Configuration
- Added `t_wave_use_qrs_removal: bool = True` to `ProcessCycleConfig` (enabled by default in `for_human()` preset)
- Updated `for_human()` preset: `snr_mad_multiplier` for T-waves reduced from 1.5 to 0.8

### Validation
- T-wave detection evaluation on QTDB dataset:
  - Detection rate improved from 0% to 99.2% for sel16265 after mapping fix
  - Overall: 1,761 PyHEARTS T-waves detected across 10 QTDB records
  - Peak timing accuracy: 0.85 ms mean error, 2.24 ms MAE (excellent)
  - Onset/offset biases identified: +41.6 ms (late) and -33.2 ms (early) - future improvement target

## [1.3.0] - 2025-01-XX

### Added
- **Signal quality assessment**: New `assess_signal_quality()` function in `pyhearts.processing.quality` module
  - Evaluates SNR, amplitude range, and baseline wander
  - Integrated into `PyHEARTS.analyze_ecg()` with configurable thresholds
  - Logs warnings for low-quality signals while allowing processing to continue

### Changed
- **P-wave detection improvements**:
  - Increased P-wave SNR threshold from 1.6 to 2.2 (37.5% increase) to reduce false positives
  - Narrowed P-wave search window from 160ms to 120ms for more precise detection
  - Improved search logic to focus on physiologically relevant pre-QRS region
- **R-peak detection improvements**:
  - Increased R-peak prominence multiplier from 2.25 to 2.5 (11% increase) to reduce false positives
  - Enhanced amplitude filtering and gap-filling logic for better precision
  - Improved notch filter Q factor from 30.0 to 35.0 for better power line noise rejection
- **Peak timing accuracy**:
  - Replaced fixed ±10 sample adjustment window with adaptive window based on Gaussian half-FWHM
  - Window capped at ±20 samples for physiologically relevant peak refinement
  - Applied consistently to all wave types (P, Q, R, S, T) in both estimation paths
- **T-wave detection tuning**:
  - Adjusted T-wave SNR threshold from 1.8 to 1.5 for improved detection while maintaining false positive control
- **Polarity detection**:
  - Improved logic to handle edge cases (e.g., negative baseline with positive R-peaks)
  - Stricter thresholds (1.2x/1.3x) for inversion detection to reduce false positives
  - Conservative fallback: assumes normal polarity when uncertain
  - Primary amplitude check: substantial positive peaks (>0.5 mV) override other conditions

### Fixed
- Fixed incorrect polarity detection for signals with negative baseline but positive R-peaks
- Improved peak refinement to place peaks at true local extrema, not just Gaussian centers

### Configuration
- Updated `for_human()` preset with optimized parameters based on QTDB benchmark analysis
- Version identifier: `v1.3-human-qtdb-improved`

### Validation
- All improvements validated on simulated signals with no performance degradation
- Maintained 100% detection rate on normal and inverted test signals
- Improved timing accuracy (mean offset: -0.44 ms)

## [1.0.0] - 2025-12-17

### Added
- Initial release of PyHEARTS
- `PyHEARTS` main class for ECG analysis
- `ProcessCycleConfig` dataclass for typed configuration
- Species presets: `for_human()` and `for_mouse()`
- Beat-by-beat morphological feature extraction (139 features)
- P, Q, R, S, T wave detection and Gaussian fitting
- Heart rate variability (HRV) metrics: SDNN, RMSSD, NN50
- Interval calculations: PR, QRS, QT, ST, RR, PP
- Shape feature extraction: duration, rise/decay, sharpness
- ECG signal simulation via `generate_ecg_signal()`
- Visualization functions in `plots/` module
- Fit quality metrics: R², RMSE
- Reproducibility metadata (git info, config hash, code SHA)
- Comprehensive input validation in configuration

### Processing Pipeline
- R-peak detection with two-pass prominence-based algorithm
- Epoch segmentation with correlation and variance thresholds
- Wavelet-based dynamic offset calculation for QRS bounds
- Local baseline detrending
- SNR gating for P and T wave detection

[Unreleased]: https://github.com/voytek-lab/pyhearts/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/voytek-lab/pyhearts/releases/tag/v1.0.0

