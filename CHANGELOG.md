# Changelog

All notable changes to PyHEARTS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Renamed `plts/` module to `plots/` for clarity
- Renamed `objs/` module to `core/` for clarity  
- Renamed `feature/HRV.py` to `feature/hrv.py` to follow Python naming conventions

### Added
- `pyproject.toml` for modern Python packaging (PEP 517/518)
- `.gitignore` file
- `CHANGELOG.md` to track version history
- `py.typed` marker for PEP 561 typed package support

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

