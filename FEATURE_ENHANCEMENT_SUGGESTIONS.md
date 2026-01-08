# Feature Enhancement Suggestions for Age and Disease Prediction

This document outlines suggested additional waveform shape parameters that could enhance PyHEARTS' capability for age and disease prediction. These features are organized by category with clinical relevance and implementation guidance.

## Current Feature Summary

PyHEARTS currently extracts **167 features** per ECG recording:
- **150 per-wave morphological features** (30 features × 5 waves: P, Q, R, S, T)
- **8 interval features** (PR, PR segment, QRS, ST segment, ST interval, QT, RR, PP)
- **3 pairwise voltage differences** (R-S, R-P, T-R)
- **2 fit quality metrics** (R², RMSE)
- **4 HRV metrics** (average HR, SDNN, RMSSD, NN50)

---

## Suggested Additional Features

### 1. Advanced Morphological Features

#### 1.1 Skewness and Asymmetry Measures
**Clinical Relevance**: T-wave asymmetry is associated with repolarization abnormalities and arrhythmia risk. P-wave asymmetry can indicate atrial remodeling.

**Suggested Features** (per wave: P, R, T):
- `{Wave}_skewness`: Statistical skewness of the wave segment
  - Calculation: `scipy.stats.skew(segment)`
  - Units: Dimensionless
  - Interpretation: Positive = right-skewed, negative = left-skewed

- `{Wave}_asymmetry_ratio`: Ratio of rise time to decay time (complement to RDSM)
  - Calculation: `decay_ms / rise_ms` (alternative to current `rdsm = rise_ms / duration_ms`)
  - Units: Dimensionless
  - Note: Currently have RDSM, but asymmetry_ratio provides different perspective

- `{Wave}_gauss_skew_alpha`: Skew parameter from skewed Gaussian fit (if enabled)
  - Already supported in `gaussian.py` via `skewed_gaussian_function()`
  - Requires: `cfg.use_skewed_gaussian = True`
  - Units: Dimensionless (alpha parameter)

**Implementation**: Add to `extract_shape_features()` in `shape.py`

---

#### 1.2 Kurtosis (Peakedness)
**Clinical Relevance**: Wave peakedness can indicate conduction velocity changes or myocardial fibrosis.

**Suggested Features** (per wave: P, Q, R, S, T):
- `{Wave}_kurtosis`: Statistical kurtosis of the wave segment
  - Calculation: `scipy.stats.kurtosis(segment)`
  - Units: Dimensionless
  - Interpretation: High kurtosis = sharper peak, low kurtosis = flatter peak

**Implementation**: Add to `extract_shape_features()` in `shape.py`

---

#### 1.3 Curvature and Second Derivative Features
**Clinical Relevance**: Curvature changes indicate inflection points, notching, or conduction delays.

**Suggested Features** (per wave: P, Q, R, S, T):
- `{Wave}_max_curvature`: Maximum absolute curvature in the wave
  - Calculation: `max(abs(second_derivative))` where `second_derivative = diff(first_derivative) / dt`
  - Units: mV/s²
  - Interpretation: Higher curvature = sharper transitions

- `{Wave}_curvature_integral`: Integral of absolute curvature
  - Calculation: `trapezoid(abs(second_derivative), dx=dt)`
  - Units: mV/s
  - Interpretation: Total "bendiness" of the wave

- `{Wave}_inflection_count`: Number of sign changes in second derivative
  - Calculation: Count zero-crossings in second derivative
  - Units: Count (integer)
  - Interpretation: Indicates notching or complex morphology

**Implementation**: Add to `extract_shape_features()` in `shape.py`, compute second derivative similar to `calc_sharpness_derivative()`

---

#### 1.4 Slope and Steepness Measures
**Clinical Relevance**: Slope changes are associated with conduction velocity and repolarization gradients.

**Suggested Features** (per wave: P, Q, R, S, T):
- `{Wave}_max_upslope`: Maximum positive slope (rising phase)
  - Calculation: `max(diff(segment[onset:peak]) / dt)`
  - Units: mV/s
  - Interpretation: Maximum rate of depolarization/repolarization

- `{Wave}_max_downslope`: Maximum negative slope (falling phase)
  - Calculation: `min(diff(segment[peak:offset]) / dt)`
  - Units: mV/s
  - Interpretation: Maximum rate of repolarization/depolarization

- `{Wave}_slope_asymmetry`: Ratio of upslope to downslope magnitudes
  - Calculation: `abs(max_upslope) / abs(max_downslope)`
  - Units: Dimensionless
  - Interpretation: Asymmetry in activation/repolarization rates

- `{Wave}_mean_slope`: Average absolute slope
  - Calculation: `mean(abs(diff(segment) / dt))`
  - Units: mV/s
  - Interpretation: Overall "steepness" of the wave

**Implementation**: Add to `extract_shape_features()` in `shape.py`

---

#### 1.5 Area and Energy Features
**Clinical Relevance**: Area under the curve represents total electrical activity, useful for detecting low-amplitude signals.

**Suggested Features** (per wave: P, Q, R, S, T):
- `{Wave}_area_above_baseline`: Area above baseline (for positive waves)
  - Calculation: `trapezoid(max(segment - baseline, 0), dx=dt)`
  - Units: mV·ms
  - Note: Currently have `voltage_integral_uv_ms` which includes negative areas

- `{Wave}_area_below_baseline`: Area below baseline (for negative waves)
  - Calculation: `trapezoid(min(segment - baseline, 0), dx=dt)`
  - Units: mV·ms

- `{Wave}_energy`: Signal energy (squared integral)
  - Calculation: `trapezoid(segment², dx=dt)`
  - Units: (mV)²·ms
  - Interpretation: Total energy of the wave

- `{Wave}_rms_amplitude`: Root mean square amplitude
  - Calculation: `sqrt(mean(segment²))`
  - Units: mV
  - Interpretation: Effective amplitude considering entire wave

**Implementation**: Add to `extract_shape_features()` in `shape.py`

---

#### 1.6 Notching and Fragmentation Detection
**Clinical Relevance**: QRS notching/fragmentation is associated with myocardial scarring, bundle branch blocks, and arrhythmia risk.

**Suggested Features**:
- `QRS_notch_count`: Number of notches in QRS complex
  - Calculation: Count local maxima/minima in QRS segment beyond expected Q, R, S peaks
  - Units: Count (integer)
  - Interpretation: >1 indicates fragmentation

- `QRS_fragmentation_index`: Measure of QRS complexity
  - Calculation: `(number_of_peaks + number_of_inflections) / QRS_duration_ms`
  - Units: Count/ms
  - Interpretation: Higher = more fragmented

- `P_wave_notch_count`: Number of notches in P-wave (indicates atrial conduction delays)
  - Calculation: Count additional peaks in P-wave segment
  - Units: Count (integer)

- `T_wave_notch_count`: Number of notches in T-wave
  - Calculation: Count additional peaks in T-wave segment
  - Units: Count (integer)

**Implementation**: Add peak detection algorithm to identify secondary peaks within wave boundaries in `extract_shape_features()`

---

### 2. ST Segment Specific Features

**Clinical Relevance**: ST segment changes are critical for detecting ischemia, myocardial infarction, and electrolyte imbalances.

**Suggested Features**:
- `ST_elevation_mv`: ST segment elevation/depression at J-point
  - Calculation: `signal[J_point_idx] - baseline`
  - Units: mV
  - Interpretation: Positive = elevation, negative = depression
  - Note: J-point is typically S-wave offset

- `ST_slope_mv_per_ms`: Slope of ST segment
  - Calculation: `(signal[ST_mid] - signal[J_point]) / (ST_mid_time - J_point_time)`
  - Units: mV/ms
  - Interpretation: Positive = upsloping, negative = downsloping, flat = horizontal

- `ST_area_above_baseline`: Area of ST segment above baseline
  - Calculation: `trapezoid(max(ST_segment - baseline, 0), dx=dt)`
  - Units: mV·ms

- `ST_deviation_max`: Maximum deviation from baseline in ST segment
  - Calculation: `max(abs(ST_segment - baseline))`
  - Units: mV

**Implementation**: Add new function `extract_st_segment_features()` in `shape.py` or `intervals.py`

---

### 3. Advanced Interval Features

#### 3.1 Rate-Corrected Intervals
**Clinical Relevance**: QT interval must be corrected for heart rate (QTc) for clinical interpretation.

**Suggested Features**:
- `QTc_Bazett_ms`: QT interval corrected using Bazett's formula
  - Calculation: `QT_ms / sqrt(RR_ms / 1000)`
  - Units: ms
  - Interpretation: Normal <440ms (males), <460ms (females)

- `QTc_Fridericia_ms`: QT interval corrected using Fridericia's formula
  - Calculation: `QT_ms / (RR_ms / 1000)^(1/3)`
  - Units: ms
  - Interpretation: Alternative correction, often more accurate at high heart rates

- `QTc_Framingham_ms`: QT interval corrected using Framingham formula
  - Calculation: `QT_ms + 0.154 * (1000 - RR_ms)`
  - Units: ms

**Implementation**: Add to `calc_intervals()` in `intervals.py`

---

#### 3.2 Additional Intervals
**Clinical Relevance**: Additional intervals provide more granular timing information.

**Suggested Features**:
- `JT_interval_ms`: Duration from J-point (S offset) to T offset
  - Calculation: `T_ri_idx - S_ri_idx`
  - Units: ms
  - Interpretation: Pure repolarization time (excludes QRS)

- `JTc_ms`: JT interval corrected for heart rate
  - Calculation: `JT_ms / sqrt(RR_ms / 1000)`
  - Units: ms

- `PR_segment_elevation_mv`: PR segment elevation (indicates pericarditis)
  - Calculation: `mean(PR_segment) - baseline`
  - Units: mV

- `TP_interval_ms`: Duration from T offset to next P onset
  - Calculation: `next_P_le_idx - T_ri_idx`
  - Units: ms
  - Interpretation: Diastolic interval

**Implementation**: Add to `calc_intervals()` in `intervals.py`

---

### 4. Beat-to-Beat Variability Features

**Clinical Relevance**: Variability in morphological features across beats is associated with disease states and aging.

**Suggested Features** (computed across all cycles):
For each per-cycle morphological feature, compute:
- `{Feature}_std`: Standard deviation across cycles
- `{Feature}_cv`: Coefficient of variation (std/mean)
- `{Feature}_iqr`: Interquartile range (75th - 25th percentile)
- `{Feature}_mad`: Median absolute deviation
- `{Feature}_range`: Range (max - min)

**Priority Features for Variability**:
- `R_gauss_height_std`, `R_gauss_height_cv`
- `QT_interval_ms_std`, `QT_interval_ms_cv`
- `QRS_interval_ms_std`, `QRS_interval_ms_cv`
- `T_gauss_height_std`, `T_gauss_height_cv`
- `PR_interval_ms_std`, `PR_interval_ms_cv`

**Implementation**: Add new function `compute_beat_to_beat_variability()` in new file `pyhearts/feature/variability.py` or add to `hrv.py`

---

### 5. Frequency Domain Features

**Clinical Relevance**: Frequency domain analysis reveals information about conduction patterns and autonomic function.

**Suggested Features** (per wave or per cycle):
- `{Wave}_dominant_frequency_hz`: Dominant frequency component
  - Calculation: FFT of wave segment, find peak frequency
  - Units: Hz
  - Interpretation: Conduction velocity indicator

- `{Wave}_spectral_centroid_hz`: Weighted mean frequency
  - Calculation: `sum(freq * magnitude) / sum(magnitude)`
  - Units: Hz

- `{Wave}_spectral_bandwidth_hz`: Frequency spread
  - Calculation: Standard deviation of frequency distribution
  - Units: Hz

- `{Wave}_spectral_power_ratio`: Ratio of low to high frequency power
  - Calculation: `power(0-5Hz) / power(5-15Hz)` for P-wave
  - Units: Dimensionless

**Implementation**: Add new function `extract_frequency_features()` in new file `pyhearts/feature/frequency.py`

---

### 6. Advanced HRV Metrics

**Clinical Relevance**: Additional HRV metrics provide more comprehensive autonomic assessment.

**Suggested Features**:
- `pNN50`: Percentage of successive RR differences > 50ms
  - Calculation: `(NN50 / (N-1)) * 100`
  - Units: Percentage
  - Interpretation: Short-term HRV measure

- `SDANN`: Standard deviation of average NN intervals in 5-minute segments
  - Calculation: Requires longer recordings, segment into 5-min windows
  - Units: ms
  - Interpretation: Long-term HRV measure

- `SDNN_index`: Mean of standard deviations of NN intervals in 5-minute segments
  - Calculation: Mean of SDNN for each 5-min segment
  - Units: ms

- `HRV_triangular_index`: Integral of RR interval histogram divided by maximum
  - Calculation: `total_RR_count / max_histogram_bin`
  - Units: Dimensionless

- `TINN`: Triangular interpolation of NN interval histogram
  - Calculation: Base width of triangle fitted to RR histogram
  - Units: ms

**Frequency Domain HRV** (requires longer recordings, ~5+ minutes):
- `LF_power`: Low frequency power (0.04-0.15 Hz)
- `HF_power`: High frequency power (0.15-0.4 Hz)
- `LF_HF_ratio`: Ratio of LF to HF power
- `total_power`: Total spectral power

**Poincaré Plot Features**:
- `SD1`: Short-term variability (perpendicular to line of identity)
- `SD2`: Long-term variability (along line of identity)
- `SD1_SD2_ratio`: Ratio of SD1 to SD2

**Implementation**: Extend `calc_hrv_metrics()` in `hrv.py`

---

### 7. Complex-Level Features

**Clinical Relevance**: Features computed across the entire cardiac cycle provide global morphology information.

**Suggested Features**:
- `QRS_axis_degrees`: Electrical axis of QRS complex
  - Calculation: Requires multi-lead ECG (not applicable for single-lead)
  - Units: Degrees
  - Note: Requires multiple leads, may not be applicable

- `T_axis_degrees`: Electrical axis of T-wave
  - Calculation: Requires multi-lead ECG
  - Units: Degrees

- `QRS_T_angle_degrees`: Angle between QRS and T axes
  - Calculation: `abs(QRS_axis - T_axis)`
  - Units: Degrees
  - Interpretation: Repolarization heterogeneity

- `cycle_area_ratio`: Ratio of positive to negative area in cycle
  - Calculation: `area_above_baseline / abs(area_below_baseline)`
  - Units: Dimensionless

- `cycle_complexity`: Measure of overall cycle complexity
  - Calculation: `(number_of_peaks + number_of_inflections) / cycle_duration_ms`
  - Units: Count/ms

**Implementation**: Add to `extract_shape_features()` or create new function `extract_complex_features()`

---

### 8. Relative and Normalized Features

**Clinical Relevance**: Normalized features reduce inter-individual variability and highlight relative changes.

**Suggested Features**:
- `{Wave}_height_ratio_to_R`: Wave amplitude normalized to R-peak
  - Calculation: `{Wave}_gauss_height / R_gauss_height`
  - Units: Dimensionless
  - Already partially covered by pairwise differences, but explicit ratios useful

- `{Wave}_duration_ratio_to_RR`: Wave duration normalized to RR interval
  - Calculation: `{Wave}_duration_ms / RR_interval_ms`
  - Units: Dimensionless

- `{Interval}_ratio_to_RR`: Interval duration normalized to RR interval
  - Calculation: `{Interval}_ms / RR_interval_ms`
  - Units: Dimensionless

- `{Wave}_area_ratio_to_R`: Wave area normalized to R-peak area
  - Calculation: `{Wave}_voltage_integral / R_voltage_integral`
  - Units: Dimensionless

**Implementation**: Add normalization functions to `extract_shape_features()` and `calc_intervals()`

---

### 9. Signal Quality and Artifact Features

**Clinical Relevance**: Quality metrics help identify reliable vs. noisy cycles for downstream analysis.

**Suggested Features** (per cycle):
- `cycle_snr_db`: Signal-to-noise ratio
  - Calculation: `20 * log10(signal_power / noise_power)`
  - Units: dB
  - Note: Partially covered by existing SNR gates, but explicit metric useful

- `cycle_baseline_wander_mv`: Baseline wander magnitude
  - Calculation: `std(detrended_signal - original_signal)` or similar
  - Units: mV

- `cycle_motion_artifact_score`: Likelihood of motion artifact
  - Calculation: Based on high-frequency content or derivative analysis
  - Units: Dimensionless (0-1)

- `cycle_quality_score`: Overall quality score
  - Calculation: Composite of SNR, baseline wander, artifact detection
  - Units: Dimensionless (0-1)

**Implementation**: Extend existing quality assessment in `processing/quality.py`

---

## Implementation Priority

### High Priority (High Clinical Value, Easy Implementation)
1. **QTc calculations** (Bazett, Fridericia) - Critical for clinical interpretation
2. **Beat-to-beat variability** (std, cv, iqr) for key features - High predictive value
3. **ST segment features** (elevation, slope) - Critical for ischemia detection
4. **Additional HRV metrics** (pNN50, SD1, SD2) - Well-established clinical value
5. **Slope features** (max upslope, max downslope) - Easy to compute, clinically relevant

### Medium Priority (Moderate Clinical Value, Moderate Implementation Complexity)
6. **Curvature features** (max curvature, inflection count) - Useful for notching detection
7. **Kurtosis and skewness** - Statistical shape descriptors
8. **Area features** (area above/below baseline, energy) - Complementary to existing integrals
9. **JT interval** - Additional repolarization measure
10. **Relative/normalized features** - Reduce inter-individual variability

### Lower Priority (Specialized Use Cases, Higher Complexity)
11. **Frequency domain features** - Requires FFT, more complex
12. **Notching/fragmentation detection** - Requires sophisticated peak detection
13. **Complex-level features** - May require multi-lead data
14. **Advanced HRV frequency domain** - Requires longer recordings

---

## Estimated Feature Count Increase

If all suggested features were implemented:
- **Per-wave morphological**: +15-20 features per wave × 5 waves = **+75-100 features**
- **ST segment**: +4 features = **+4 features**
- **Intervals**: +6-8 features = **+8 features**
- **Beat-to-beat variability**: +50-100 features (depending on which base features) = **+50-100 features**
- **HRV**: +10-15 features = **+15 features**
- **Frequency domain**: +15-20 features = **+20 features**
- **Complex-level**: +5-10 features = **+10 features**
- **Quality metrics**: +4-5 features = **+5 features**

**Total potential increase**: **+187-262 features**

**New total**: **354-429 features** (from current 167)

---

## Recommendations

1. **Start with High Priority features** - Implement QTc, variability metrics, ST features, and additional HRV first
2. **Maintain backward compatibility** - Add new features as optional/extended feature set
3. **Add configuration flags** - Allow users to enable/disable feature groups for performance
4. **Document clinical significance** - Update FEATURE_REFERENCE.md with new features
5. **Validate against clinical datasets** - Test new features on age/disease prediction tasks
6. **Consider computational cost** - Some features (frequency domain, variability) may be expensive for large datasets

---

## References

- Bazett HC. An analysis of the time-relations of electrocardiograms. Heart. 1920;7:353-370.
- Fridericia LS. Die Systolendauer im Elektrokardiogramm bei normalen Menschen und bei Herzkranken. Acta Medica Scandinavica. 1920;53:469-486.
- Task Force of the European Society of Cardiology. Heart rate variability: standards of measurement, physiological interpretation, and clinical use. Circulation. 1996;93:1043-1065.
- Tereshchenko LG, et al. Beat-to-beat QT interval variability: novel evidence for repolarization lability in ischemic and nonischemic dilated cardiomyopathy. Circulation. 2008;118:191-199.

