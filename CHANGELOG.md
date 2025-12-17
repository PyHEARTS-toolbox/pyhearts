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

