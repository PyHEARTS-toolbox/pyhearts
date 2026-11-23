# Changelog

All notable changes to PyHEARTS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-11-23

### Changed
- License: PyHEARTS is now available under the PolyForm Strict License 1.0.0, which permits
  noncommercial use but not modification or redistribution. Version 1.0.0 granted no use without
  written permission. `LICENSE.md` now also states that a patent is pending (SD2024-342-2).
- No code changes. Outputs are identical to 1.0.0.

## [1.0.0] - 2026-11-23

First public release, tagged `v1.0.0`. It includes every change below, made after the
July 2026 pre-release snapshot.

### Fixed (Gaussian height cap)
- Warm-started height bounds (previous height +/- `bound_factor`) could grow 20% per beat without limit.
  On long recordings, R and S (and sometimes P and Q) converged on one center with large opposite
  heights that cancel (for example +212 and -211 mV on a 1 mV QRS), while the fit R^2 stayed near 0.99.
  Before the fix this affected 7.0% of Autonomic Aging beats and 6.1% of MGH/MF beats.
- New `ProcessCycleConfig.max_height_factor` (default 1.5): every Gaussian height is bounded by 1.5 times
  the cycle's peak absolute detrended amplitude (`calc_bounds(..., height_cap=...)`), and a warm start whose
  carried-over height already exceeds the cap is discarded so the cycle is refitted from fresh initial values.
- Interval outputs are unchanged on sampled records (median shifts of 1 ms or less); fitted amplitudes and the
  shape features computed from them change. Regression test: `tests/test_height_cap.py`.

### Fixed (Gaussian SD bounds: standard-2)
- `calc_bounds` no longer rounds the Gaussian SD bounds down to whole samples. With
  `upper = int(1.2 * SD)`, an SD of 4 samples or less could only shrink, so on long
  recordings a warm-started fit trapped P, Q and S waves at SDs of 1 to 5 samples for the
  rest of the recording (34% of Autonomic Aging recordings for P). Those waves were then
  shorter than the 20 ms minimum width and lost their shape features. Bounds are now
  `max(0.8 * SD, 1)` to `max(1.2 * SD, lower + 1)` samples. `STANDARD_PIPELINE_VERSION` is
  now `standard-2`; cohort outputs must be regenerated.

### Removed (unused record-delineation refresh path)
- Removed `apply_record_level_delineation`, `apply_record_fill_missing_t` and the ten
  private helpers only they used from `processing/record_delineation.py`. No code in the
  package called them.
- Removed the modules reachable only through them: `processing/cycle_feature_refresh.py`
  (`refresh_cycles_after_timing_update`, no longer exported from `pyhearts.processing`),
  `pt_detection_mode.py`, `qrs_boundary_detection_v2.py`, `record_wavelet_delineation.py`,
  `t_timing_audit.py` and `waveletoffset.py`.
- Removed the 25 `ProcessCycleConfig` settings read only by that code (for example
  `record_delineation_refresh_features`, `record_delineation_replace_t`,
  `qrs_onset_search_window_ms`, `lite_mode`). Configs that still set them must drop them;
  the validation loaders already ignore unknown keys.
- Outputs are unchanged: 40 PTB-XL records give identical feature tables before and after.

### Changed (P boundaries: boundary-layer-3)
- P boundaries are only placed when the P peak lies within 250 ms before QRS onset
  (`p_max_lead_ms`); a P peak farther back is a T or U wave, or atrial fibrillation/flutter
  activity, and the beat gets no P boundaries (PR and P duration NaN). Each P slope maximum
  must lie within 100 ms of the P peak (`p_half_window_ms`, was 150 ms). Developed on QTDB and
  PTB-XL fold 9. `BOUNDARY_LAYER_VERSION` is now `boundary-layer-3`; `LEGACY_P_PARAMS`
  restores the boundary-layer-2 P rule.

### Changed (QRS boundaries: boundary-layer-2)
- New QRS onset/offset rule (`qrs_rule="noise_chain"`, default). Slope maxima on
  the 2^2 wavelet scale count as part of the QRS only if they exceed a fraction of
  the QRS slope (0.04 before R, 0.10 after R) and 3x the median |W2| within 1.5 s,
  so noise maxima no longer widen the complex. Maxima are chained outward from R
  (gaps up to 40 ms before R, 60 ms after R; search +/-150 ms). QRS-end thresholds
  are 0.5 (last maximum positive) and 0.125 (negative). Developed on QTDB and
  PTB-XL fold 9 only (`validation/qrs_dev_cache.py`, `validation/qrs_dev_eval.py`).
  `BOUNDARY_LAYER_VERSION` is now `boundary-layer-2`; `LEGACY_QRS_PARAMS`
  reproduces boundary-layer-1. Outputs from earlier runs must be regenerated.

### Changed (breaking: output columns)
- Clinical boundaries and intervals now come from a wave-boundary layer
  (`pyhearts.processing.boundary_layer`): multiscale wavelet delineation
  (Martinez et al. 2004 thresholds), QRS as one complex, template-guided P/T
  checks, no imputation from neighboring beats. Sources `PR_interval_ms`,
  `PR_segment_ms`, `QRS_interval_ms`, `ST_segment_ms`, `ST_interval_ms`,
  `QT_interval_ms` and `P_duration_ms` (human / species-agnostic presets).
  Previously these were built from Gaussian 30%-of-height crossings, which sit
  inside P and T waves and outside fitted Q/S waves.
- Gaussian per-wave widths renamed `{P,Q,R,S,T}_duration_ms` →
  `{P,Q,R,S,T}_width_ms` (morphology features, unchanged values).
- `{file_id}_meta.json` records `intervals.source`, the boundary-layer
  version and parameters, and the physiological limits applied.

- R-peak detection for the human preset now uses `rpeak_method="gradient"`,
  a port of NeuroKit2's default detector (MIT License; identical peaks on all
  200 LUDB records). LUDB lead II, +/-40 ms: Se/PPV 0.972/0.970 (was
  0.920/0.920). QTDB (82 records, full beat annotations, +/-150 ms): Se/PPV
  0.959/0.956 (was 0.923/0.929). Mouse and species-agnostic presets keep
  `rpeak_method="prominence"`.
- Human preset (lead II) no longer flips signal polarity
  (`rpeak_auto_polarity=False`): R is always the positive QRS deflection,
  matching clinical convention and cardiologist annotations.
- Human preset searches for R within ±50 ms of the detected QRS fiducial
  (`r_search_half_window_ms=50`). Previously R was the maximum of the whole
  cycle, so a T wave taller than R (or a QRS without a positive deflection)
  was labeled R and every other wave label shifted.
- New column `QRS_fiducial_global_idx`; the boundary layer anchors on it when
  no R is fitted. LUDB (lead II): R-peak Se/PPV 0.990/0.991 at ±40 ms;
  boundary coverage P 98-99%, QRS 96%, T 94%. PTB-XL fold 10 vs 12SL:
  R amplitude r 0.970 (was 0.875), S amplitude r 0.882 (was 0.490).

### Added
- Wave-boundary columns `P/QRS/T_onset_global_idx`, `P/QRS/T_offset_global_idx`.
- `pyhearts.standard.run_standard`: one standardized pipeline for cohort runs
  (lead II only, sampling-rate check, missing-sample exclusion, fixed
  filtering, preprocessing logged in metadata).

### Fixed
- `extract_shape_features`: a Gaussian center fitted outside the cycle window
  raised `IndexError` and discarded every component's shape features for that
  cycle (seen on QTDB sel231). The out-of-window component is now skipped.
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
- Mouse morphology T search: compact post-S window (`t_ignore_wavelet_guard`,
  `t_end_margin_ms`), reseed when prior fit lacks T (`t_reseed_if_missing`),
  and height-above-baseline SNR gating (`t_height_above_baseline`). Enabled
  only in `ProcessCycleConfig.for_mouse()` (`version=v1-fitbounds-clip-mouse-t`);
  human defaults unchanged.
- Record-T merge fusion (human_unified): restore Gaussian T on record-T miss
  (`record_t_fallback_gaussian_on_miss`) and prefer Gaussian when it is ≥20 ms
  later (`record_t_prefer_later_gaussian_ms`). Does not change the record-T
  search itself.

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
  held-outs for paper-era parameters only. Local held-out corpora and the
  `validation/` harness are not shipped.

## [1.0.0-pre] - 2026-07-22

Pre-release snapshot of the PyHEARTS toolbox (not tagged or published).

### Added
- Installable package with `PyHEARTS` as the public analyzer.
- Morphology core for R/P detection, cycle segmentation, and symmetric
  Gaussian fitting of P/Q/R/S/T waves.
- Record-level T stage at the end of `analyze_ecg`.
- Explicit `T_gaussian_global_center_idx` alongside record-T `T_global_center_idx`.
- Reproducible CSV + metadata export via `save_output` (package version plus
  pipeline tag and resolved configs).
- Optional extras for NeuroKit2 simulation (`sim`) and WFDB I/O (`wfdb`).
- Example notebooks under `examples/`.
- Unit, smoke, and pipeline regression tests; CI packaging check.

### Validation
- Dataset roles: AA/PTB-XL = tuning; QTDB = development/benchmark (incl. Dec
  2024 re-tune); SPH/MGH/mouse = paper-era held-outs for paper-era parameters
  only (no recorded tuning objective/record list).
- SPH normal-repeat subset: 802 recordings, median R² 0.9688, 83.83% of
  fitted cycles above R² 0.9, and 98.06% T availability.
- QTDB manual subset (development/benchmark): 300 expert beats, 99.67% R
  sensitivity, 80.94% P sensitivity, and 69.00% T sensitivity within ±40 ms.
