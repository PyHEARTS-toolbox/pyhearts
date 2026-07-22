# Changelog

All notable changes to PyHEARTS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- `compute_hrv_metrics` now builds RR intervals from detected R peaks
  (pre-epoch / pre-morphology), physiology-gated by `rr_bounds_ms`, instead of
  using retained-cycle `RR_interval_ms`. Morphology feature-table RR is unchanged.

## [1.0.0] - 2026-07-22

First public release of the PyHEARTS toolbox.

### Added
- Installable package with `PyHEARTS` as the public analyzer.
- Morphology core for R/P detection, cycle segmentation, and symmetric
  Gaussian fitting of P/Q/R/S/T waves.
- Record-level record-level T stage at the end of `analyze_ecg`.
- Explicit `T_gaussian_global_center_idx` alongside record-T `T_global_center_idx`.
- Reproducible CSV + metadata export via `save_output` (package version plus
  pipeline tag and resolved configs).
- Optional extras for NeuroKit2 simulation (`sim`) and WFDB I/O (`wfdb`).
- Example notebooks under `examples/`.
- Unit, smoke, and pipeline regression tests; CI packaging check.

### Validation
- SPH normal-repeat subset: 802 recordings, median R² 0.9688, 83.83% of
  fitted cycles above R² 0.9, and 98.06% T availability.
- QTDB manual subset: 300 expert beats, 99.67% R sensitivity, 80.94% P
  sensitivity, and 69.00% T sensitivity within ±40 ms.
