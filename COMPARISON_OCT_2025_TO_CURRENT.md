# Main Differences: PyHEARTS October 2025 vs Current Version

## Module Structure & Organization
- **Module renaming**: `objs/` → `core/`, `plts/` → `plots/`, `HRV.py` → `hrv.py` (Python naming conventions)
- **Processing modules**: Expanded from 13 to 25+ files with specialized detection methods

## P-Wave Detection
- **Derivative-validated method**: New robust P-wave detection method (`p_wave_detection_derivative_validated.py`) with multi-stage validation (amplitude, derivative extrema, sign, noise level, temporal ordering)
- **Multiple detection methods**: Added `p_wave_detection_fixed_window.py` and `p_wave_detection_improved.py` as alternatives
- **Training phase**: New adaptive training phase (`p_training_phase.py`) that learns P-wave characteristics from initial recording segment
- **Fallback methods**: Enhanced fallback detection with parabolic interpolation for sub-sample accuracy

## T-Wave Detection
- **QRS removal**: New `qrs_removal.py` module implementing sigmoid-based QRS replacement to reduce interference with T-wave detection
- **Derivative-based detection**: Enhanced `derivative_t_detection.py` with QRS-removed signal processing
- **Improved accuracy**: T-wave detection accuracy improved through QRS artifact elimination

## R-Peak Detection
- **Multiple algorithms**: Added support for three detection methods:
  - Prominence-based (default, optimized)
  - Pan-Tompkins algorithm (`pan_tompkins.py`)
  - Bandpass energy-based (`bandpass_energy_rpeak.py`)
- **Enhanced dual-polarity handling**: Improved detection for mixed-polarity signals (both inverted and upright R-peaks)
- **Optimized parameters**: R-peak prominence multiplier lowered from 3.0σ to 2.5σ for better recall (based on QTDB benchmark)

## QRS Boundary Detection
- **Derivative-based method**: New `qrs_boundary_detection_v2.py` with improved accuracy over fixed windows
- **Adaptive thresholds**: 8% derivative threshold with 1.5 SD baseline proximity criteria
- **Accommodates broad complexes**: Better handling of bundle branch blocks and wide QRS morphologies

## Feature Extraction
- **ST segment features**: New `st_segment.py` module extracting ST elevation, ST slope, and ST deviation (not included in current methods section per user request)
- **Beat-to-beat variability**: New `variability.py` module computing variability metrics (std, CV, IQR, MAD, range) for priority features across cycles
- **Concavity validation**: Added validation checks to ensure physiologically plausible waveform shapes (positive waves have true maxima, negative waves have true minima)
- **Slope features**: Enhanced shape features with maximum upslope, maximum downslope, and slope asymmetry calculations
- **Additional intervals**: Expanded interval calculations to include PR_segment, ST_segment, and ST_interval separately from PR_interval and QT_interval

## Signal Quality & Preprocessing
- **Signal quality assessment**: New `quality.py` module with automated quality checks (SNR, amplitude range, baseline wander) and configurable warning thresholds
- **Quality warnings**: Automatic generation of quality warnings in metadata without interrupting processing
- **Sampling rate warnings**: Automatic warnings when sampling rate <300 Hz may impair Q/S wave detection

## Configuration & Defaults
- **Optimized thresholds**: Multiple parameter adjustments based on QTDB benchmark analysis (Dec 2024):
  - R-peak prominence: 3.0σ → 2.5σ
  - Epoch correlation threshold: 0.80 → 0.70
  - Epoch variance threshold: 5.0× → 6.0×
  - Second-pass RR fraction: 55% → 50%
- **Enhanced presets**: `for_human()` and `for_mouse()` presets with more refined species-specific parameters
- **Sensitivity modes**: Added `sensitivity` parameter ("standard", "high", "maximum") for R-peak detection tuning

## HRV Metrics
- **Expanded metrics**: HRV calculation (`hrv.py`) now includes additional metrics:
  - pNN50 (percentage of successive RR differences >50 ms)
  - SD1 and SD2 (Poincaré plot metrics)
- **Previous version**: Only SDNN, RMSSD, and NN50

## Processing Improvements
- **Adaptive thresholds**: New `adaptive_threshold.py` module for SNR-adaptive boundary detection
- **Enhanced validation**: Improved peak validation logic with better handling of edge cases
- **Error handling**: Better error recovery and logging throughout the pipeline
- **Reproducibility**: Enhanced metadata tracking with git info, code SHA, and configuration hashing

## Feature Completeness
- **Total features**: Expanded from ~139 features to ~243 features per recording (including HRV and variability metrics)
- **Morphological features**: Enhanced with slope features (max_upslope, max_downslope, slope_asymmetry) and concavity validation
- **Interval features**: Added PR_segment, ST_segment, and ST_interval as separate measurements
- **Direct voltage measurements**: Added center_voltage, le_voltage, and ri_voltage at fiducial points

## Performance & Accuracy
- **QTDB benchmark validation**: Extensive validation on QTDB dataset informing parameter optimization
- **Detection rates**: Improved P-wave detection (97%+ with human preset), T-wave detection (85%+), and R-peak recall (>70% with high sensitivity)
- **Timing accuracy**: Median absolute timing error <8 ms for fiducial points

### Quantitative Performance Improvements (Dec 19, 2025 → Jan 7, 2026)

Based on evaluation of 103 common subjects from the QTDB dataset:

**Detection Rate Improvements:**
- **T-wave**: 47.8% → 90.8% (+89.9% relative improvement) - nearly doubled
- **Q-wave**: 56.6% → 99.8% (+76.3% relative improvement) - nearly doubled  
- **S-wave**: 75.3% → 98.4% (+30.8% relative improvement)
- **P-wave**: 82.3% → 95.2% (+15.7% relative improvement)
- **R-wave**: 94.4% → 99.7% (+5.7% relative improvement)

**Additional Metrics:**
- **+206,734 additional waveform detections** across all subjects
- **+17,813 additional cycles** retained (+19.0% increase)
- **63-97% reduction in variability** (standard deviation) across all wave types
- **All waves now achieve ≥90% mean detection rate**

See `QTDB_PERFORMANCE_IMPROVEMENTS.md` for detailed quantitative analysis.

