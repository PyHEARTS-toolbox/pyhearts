# Changelog

All notable changes to PyHEARTS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Track `docs/OUTPUT_VARIABLES.md` as the canonical per-cycle / HRV feature
  set (names, families, and calculations).
- `reconstruct_ecg`: rebuild a global ECG timeseries from per-beat Gaussian
  morphology (P/Q/R/S/T μ, height, σ, with voltage / FWHM / rise–decay
  fallbacks), placing components on sample index (and `time_ms`). Residual
  noise is taken from the original recording when provided, or synthesized
  from per-cycle `rmse`. Also available as `analyzer.reconstruct_ecg()`.
  Example walkthrough: `examples/reconstruct_ecg.ipynb`.
- Morphology R-peak auto-polarity: inverted QRS / lead polarity is detected
  (`detect_signal_polarity`) and the analysis trace is uprighted before peak
  finding and morphology / record-T. Disable with `rpeak_auto_polarity=False`.
- Held-out validation protocol under `validation/`: frozen public-default
  config (`config_frozen_v1.json`), LUDB-only primary scoring, and post-lock
  sensitivity sweep. Optional paper-era morphology arm lockfile included.
  Primary LUDB (200 records, lead II): unconditional Se @ ±40 ms R/P/T =
  85.58% / 79.59% / 58.59%.
- Mouse morphology T search: compact post-S window (`t_ignore_wavelet_guard`,
  `t_end_margin_ms`), reseed when prior fit lacks T (`t_reseed_if_missing`),
  and height-above-baseline SNR gating (`t_height_above_baseline`). Enabled
  only in `ProcessCycleConfig.for_mouse()` (`version=v1-fitbounds-clip-mouse-t`);
  human defaults unchanged.
- Record-T merge fusion (human_unified): restore Gaussian T on record-T miss
  (`record_t_fallback_gaussian_on_miss`) and prefer Gaussian when it is ≥20 ms
  later (`record_t_prefer_later_gaussian_ms`). Lifts LUDB lead-II T Se@40 from
  ~62% to ~75%+ without changing the record-T search itself.

### Fixed
- Morphology Gaussian warm-start: if a prior beat is missing any of P/Q/R/S/T,
  re-run full peak search (`reseed_missing_components`, default on) so a miss
  on beat 0 cannot lock that wave out of the rest of the recording. Mouse T
  reseed (`t_reseed_if_missing`) is unchanged.
- Harden `preprocess_ecg` / `preprocess_signal`: coerce column/row vectors to
  1-D, require paired filter args (`filter_order` with band cutoffs;
  `quality_factor` with `notch_frequency`), interpolate sparse NaNs
  (≤ `max_nan_frac`), and raise on failure instead of printing and returning
  `None`.
- Clip Gaussian `curve_fit` initial guesses into their parameter bounds before
  SciPy `trf` (`clip_guess_to_bounds`). CASE 2 previously passed raw `p0`, so
  tight `bound_factor` values produced mass "Initial guess is outside of
  provided bounds" failures and NaN `R_global_center_idx` exports — falsely
  looking like R-detection failures.

### Changed
- `compute_hrv_metrics` now builds RR intervals from detected R peaks
  (pre-epoch / pre-morphology), physiology-gated by `rr_bounds_ms`, instead of
  using retained-cycle `RR_interval_ms`. Morphology feature-table RR is unchanged.
- Dataset role language: QTDB is development/benchmark (not held-out for
  current defaults); AA/PTB-XL are tuning; SPH/MGH/mouse are paper-era
  held-outs for paper-era parameters only; LUDB is the frozen-config held-out.

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
- Dataset roles: AA/PTB-XL = tuning; QTDB = development/benchmark (incl. Dec
  2024 re-tune); SPH/MGH/mouse = paper-era held-outs for paper-era parameters
  only (no recorded tuning objective/record list); LUDB = frozen-config
  held-out for current public defaults (`validation/`).
- SPH normal-repeat subset: 802 recordings, median R² 0.9688, 83.83% of
  fitted cycles above R² 0.9, and 98.06% T availability.
- QTDB manual subset (development/benchmark): 300 expert beats, 99.67% R
  sensitivity, 80.94% P sensitivity, and 69.00% T sensitivity within ±40 ms.
