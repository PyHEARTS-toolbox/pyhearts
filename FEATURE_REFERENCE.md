# PyHEARTS Feature Reference

Complete list of all features extracted by PyHEARTS, including definitions and calculation methods.

**Total Features**: 181 per cardiac cycle + 7 HRV metrics + ~45-70 variability metrics = **~240 total features**

---

## Table of Contents

1. [Per-Wave Morphological Features](#per-wave-morphological-features)
2. [Interval Features](#interval-features)
3. [Pairwise Voltage Differences](#pairwise-voltage-differences)
4. [Fit Quality Metrics](#fit-quality-metrics)
5. [Heart Rate Variability (HRV) Metrics](#heart-rate-variability-hrv-metrics)

---

## Per-Wave Morphological Features

Each feature is extracted for all 5 ECG waves: **P, Q, R, S, T**

**Total per wave**: 33 features × 5 waves = **165 features**

### 1. Global Indices (Absolute Sample Indices)

These indices reference positions in the full ECG signal (not cycle-relative).

#### `{Wave}_global_center_idx`
- **Definition**: Absolute sample index of the wave peak center in the full ECG signal
- **Calculation**: Maps cycle-relative `center_idx` to global signal via `xs_samples` array
- **Units**: Sample number (integer)
- **Example**: `R_global_center_idx = 1250` means R-peak is at sample 1250 in the full signal

#### `{Wave}_global_le_idx`
- **Definition**: Absolute sample index of the left edge (onset) of the wave
- **Calculation**: Maps cycle-relative `le_idx` to global signal via `xs_samples` array
- **Units**: Sample number (integer)
- **Note**: Left edge is the onset/start of the wave

#### `{Wave}_global_ri_idx`
- **Definition**: Absolute sample index of the right edge (offset) of the wave
- **Calculation**: Maps cycle-relative `ri_idx` to global signal via `xs_samples` array
- **Units**: Sample number (integer)
- **Note**: Right edge is the offset/end of the wave

---

### 2. Time-Domain Locations (Relative to Cycle)

These times are relative to the start of the cardiac cycle (R-peak centered).

#### `{Wave}_center_ms`
- **Definition**: Time of wave peak center relative to cycle start
- **Calculation**: `center_idx / sampling_rate * 1000.0`
- **Units**: Milliseconds (ms)
- **Note**: For R-wave, this is typically near 0 (cycle is centered on R-peak)

#### `{Wave}_le_ms`
- **Definition**: Time of left edge (onset) relative to cycle start
- **Calculation**: `le_idx / sampling_rate * 1000.0`
- **Units**: Milliseconds (ms)

#### `{Wave}_ri_ms`
- **Definition**: Time of right edge (offset) relative to cycle start
- **Calculation**: `ri_idx / sampling_rate * 1000.0`
- **Units**: Milliseconds (ms)

---

### 3. Local Indices (Within Cycle)

These indices are relative to the start of the current cardiac cycle.

#### `{Wave}_center_idx`
- **Definition**: Sample index of wave peak center within the cycle
- **Calculation**: Detected peak location in detrended cycle signal
- **Units**: Sample number (integer, cycle-relative)
- **Note**: For Q/S waves (negative peaks), refined to actual minimum in the segment

#### `{Wave}_le_idx`
- **Definition**: Sample index of left edge (onset) within the cycle
- **Calculation**: 
  - **Derivative-based method** (if `use_derivative_based_limits=True`): Finds point where signal slope/curvature changes significantly relative to local baseline
  - **Threshold-based method** (default): Finds point where signal crosses `threshold_fraction * (peak_height - baseline)` moving away from peak
- **Units**: Sample number (integer, cycle-relative)

#### `{Wave}_ri_idx`
- **Definition**: Sample index of right edge (offset) within the cycle
- **Calculation**: Same as `le_idx` but searching in opposite direction
- **Units**: Sample number (integer, cycle-relative)

---

### 4. Gaussian Fit Parameters

These parameters describe the Gaussian function fitted to each wave.

#### `{Wave}_gauss_center`
- **Definition**: Gaussian-fitted center position (cycle-relative index)
- **Calculation**: Result from `scipy.optimize.curve_fit()` using `gaussian_function()`
- **Units**: Sample number (float, cycle-relative)
- **Note**: May differ slightly from `center_idx` due to fitting optimization

#### `{Wave}_gauss_height`
- **Definition**: Gaussian-fitted peak amplitude
- **Calculation**: Height parameter from Gaussian fit: `A` in `A * exp(-0.5 * ((x - μ) / σ)²)`
- **Units**: Millivolts (mV)
- **Note**: For Q/S waves (negative), this is the negative amplitude

#### `{Wave}_gauss_stdev_samples`
- **Definition**: Gaussian standard deviation in samples
- **Calculation**: Standard deviation parameter `σ` from Gaussian fit, in sample units
- **Units**: Samples (float)
- **Note**: Describes the width of the Gaussian distribution

#### `{Wave}_gauss_stdev_ms`
- **Definition**: Gaussian standard deviation in milliseconds
- **Calculation**: `gauss_stdev_samples / sampling_rate * 1000.0`
- **Units**: Milliseconds (ms)

#### `{Wave}_gauss_fwhm_samples`
- **Definition**: Full-width at half-maximum (FWHM) in samples
- **Calculation**: `2.0 * sqrt(2.0 * ln(2.0)) * gauss_stdev_samples` ≈ `2.355 * gauss_stdev_samples`
- **Units**: Samples (float)
- **Note**: FWHM is the width of the peak at half its maximum height

#### `{Wave}_gauss_fwhm_ms`
- **Definition**: Full-width at half-maximum in milliseconds
- **Calculation**: `gauss_fwhm_samples / sampling_rate * 1000.0`
- **Units**: Milliseconds (ms)

---

### 5. FWHM-Based Boundary Indices

These boundaries are derived purely from the Gaussian morphology (half-FWHM on each side).

#### `{Wave}_fwhm_le_idx`
- **Definition**: Left edge index based on half-FWHM from Gaussian center
- **Calculation**: `gauss_center - (gauss_fwhm_samples / 2.0)`
- **Units**: Sample number (float, cycle-relative)
- **Note**: Provides morphology-based width measure, independent of threshold detection

#### `{Wave}_fwhm_ri_idx`
- **Definition**: Right edge index based on half-FWHM from Gaussian center
- **Calculation**: `gauss_center + (gauss_fwhm_samples / 2.0)`
- **Units**: Sample number (float, cycle-relative)

#### `{Wave}_fwhm_le_ms`
- **Definition**: Left edge time based on FWHM (milliseconds)
- **Calculation**: `fwhm_le_idx / sampling_rate * 1000.0`
- **Units**: Milliseconds (ms)

#### `{Wave}_fwhm_ri_ms`
- **Definition**: Right edge time based on FWHM (milliseconds)
- **Calculation**: `fwhm_ri_idx / sampling_rate * 1000.0`
- **Units**: Milliseconds (ms)

#### `{Wave}_fwhm_global_le_idx`
- **Definition**: Global sample index of FWHM-based left edge
- **Calculation**: Maps `fwhm_le_idx` to global signal via `xs_samples` array
- **Units**: Sample number (integer)

#### `{Wave}_fwhm_global_ri_idx`
- **Definition**: Global sample index of FWHM-based right edge
- **Calculation**: Maps `fwhm_ri_idx` to global signal via `xs_samples` array
- **Units**: Sample number (integer)

---

### 6. Amplitudes at Key Points

Voltage measurements at specific locations on the wave.

#### `{Wave}_center_voltage`
- **Definition**: Voltage at the wave peak center
- **Calculation**: `sig_detrended[center_idx]`
- **Units**: Millivolts (mV)
- **Note**: Direct measurement from detrended signal at peak location

#### `{Wave}_le_voltage`
- **Definition**: Voltage at the left edge (onset)
- **Calculation**: `sig_detrended[le_idx]`
- **Units**: Millivolts (mV)

#### `{Wave}_ri_voltage`
- **Definition**: Voltage at the right edge (offset)
- **Calculation**: `sig_detrended[ri_idx]`
- **Units**: Millivolts (mV)

---

### 7. Duration, Symmetry, and Sharpness

Morphological characteristics describing wave shape.

#### `{Wave}_duration_ms`
- **Definition**: Total duration of the wave from onset to offset
- **Calculation**: `(ri_idx - le_idx) / sampling_rate * 1000.0`
- **Units**: Milliseconds (ms)
- **Validation**: Must be ≥ `duration_min_ms` (default: 20ms for humans, 2ms for mice)

#### `{Wave}_rise_ms`
- **Definition**: Duration from left edge (onset) to peak center
- **Calculation**: `(center_idx - le_idx) / sampling_rate * 1000.0`
- **Units**: Milliseconds (ms)
- **Note**: Describes the upslope/rising phase of the wave

#### `{Wave}_decay_ms`
- **Definition**: Duration from peak center to right edge (offset)
- **Calculation**: `(ri_idx - center_idx) / sampling_rate * 1000.0`
- **Units**: Milliseconds (ms)
- **Note**: Describes the downslope/decaying phase of the wave

#### `{Wave}_rdsm`
- **Definition**: Rise-Decay Symmetry Measure
- **Calculation**: `rise_samples / duration_samples`
- **Units**: Dimensionless ratio (0-1)
- **Interpretation**:
  - `rdsm = 0.5`: Symmetric wave (equal rise and decay)
  - `rdsm < 0.5`: Faster rise, slower decay
  - `rdsm > 0.5`: Slower rise, faster decay

#### `{Wave}_sharpness`
- **Definition**: Derivative-based sharpness measure, amplitude-normalized
- **Calculation**:
  1. Extract wave segment: `signal[left_idx:right_idx+1]`
  2. Apply Savitzky-Golay smoothing (window=7, polyorder=3) if segment length ≥ 7
  3. Compute derivative: `abs(diff(segment)) / dt` (units: mV/s)
  4. Compute summary statistic:
     - `mean`: Mean of absolute derivatives
     - `median`: Median of absolute derivatives
     - `p95`: 95th percentile of absolute derivatives (default)
  5. Normalize by amplitude:
     - `p2p`: Peak-to-peak amplitude (5th-95th percentile range)
     - `rms`: Root mean square amplitude
     - `mad`: Median absolute deviation
  6. Final: `sharpness = derivative_statistic / amplitude_norm`
- **Units**: Dimensionless (1/s or s⁻¹)
- **Configuration**: Controlled by `cfg.sharp_stat` and `cfg.sharp_amp_norm`
- **Note**: Higher values indicate sharper, more rapid transitions

#### `{Wave}_max_upslope_mv_per_s`
- **Definition**: Maximum positive slope in the rising phase (onset to peak)
- **Calculation**: `max(diff(rising_segment) / dt)` where `rising_segment = signal[le_idx:center_idx+1]`
- **Units**: Millivolts per second (mV/s)
- **Interpretation**: Maximum rate of depolarization (for P, R, T) or repolarization (for Q, S)
- **Clinical Relevance**: Higher values indicate faster activation/repolarization, associated with conduction velocity

#### `{Wave}_max_downslope_mv_per_s`
- **Definition**: Maximum negative slope in the falling phase (peak to offset)
- **Calculation**: `min(diff(decay_segment) / dt)` where `decay_segment = signal[center_idx:ri_idx+1]`
- **Units**: Millivolts per second (mV/s)
- **Interpretation**: Maximum rate of repolarization (for P, R, T) or depolarization (for Q, S)
- **Clinical Relevance**: Higher magnitude values indicate faster repolarization/depolarization

#### `{Wave}_slope_asymmetry`
- **Definition**: Ratio of maximum upslope to maximum downslope magnitudes
- **Calculation**: `abs(max_upslope) / abs(max_downslope)`
- **Units**: Dimensionless ratio
- **Interpretation**:
  - `slope_asymmetry > 1`: Faster rise than decay (asymmetric activation)
  - `slope_asymmetry < 1`: Faster decay than rise (asymmetric repolarization)
  - `slope_asymmetry ≈ 1`: Symmetric slopes
- **Clinical Relevance**: Asymmetry in T-wave slopes is associated with repolarization heterogeneity and arrhythmia risk

---

### 8. Voltage Integral

#### `{Wave}_voltage_integral_uv_ms`
- **Definition**: Area under the wave curve
- **Calculation**:
  1. Extract segment: `signal[left_idx:right_idx+1]`
  2. Integrate using trapezoidal rule: `np.trapezoid(segment, dx=dt)` where `dt = 1/sampling_rate`
  3. Convert units: `(mV·s) × 1e6 = µV·ms`
- **Units**: Microvolt-milliseconds (µV·ms)
- **Note**: Represents the total electrical activity of the wave (voltage-time product)

---

## Interval Features

Timing intervals between different waves or wave boundaries.

**Total**: 11 interval features (8 basic + 3 QTc)

### 1. PR Interval

#### `PR_interval_ms`
- **Definition**: Duration from P-wave onset to QRS complex onset
- **Calculation**: `(Q_le_idx - P_le_idx) / sampling_rate * 1000.0`
- **Units**: Milliseconds (ms)
- **Physiological Range**: 50-500 ms (validated)
- **Clinical Significance**: Measures atrioventricular conduction time
- **Interpolation**: If P or Q onset missing, interpolates from surrounding cycles (window_size=3)

### 2. PR Segment

#### `PR_segment_ms`
- **Definition**: Duration from P-wave offset to QRS complex onset
- **Calculation**: `(Q_le_idx - P_ri_idx) / sampling_rate * 1000.0`
- **Units**: Milliseconds (ms)
- **Physiological Range**: 10-400 ms (validated)
- **Clinical Significance**: Isoelectric segment between P-wave and QRS

### 3. QRS Interval

#### `QRS_interval_ms`
- **Definition**: Duration of the QRS complex (from Q onset to S offset)
- **Calculation**: `(S_ri_idx - Q_le_idx) / sampling_rate * 1000.0`
- **Units**: Milliseconds (ms)
- **Physiological Range**: 20-300 ms (validated)
- **Clinical Significance**: Ventricular depolarization duration
- **Note**: Includes Q, R, and S waves

### 4. ST Segment

#### `ST_segment_ms`
- **Definition**: Duration from S-wave offset to T-wave onset
- **Calculation**: `(T_le_idx - S_ri_idx) / sampling_rate * 1000.0`
- **Units**: Milliseconds (ms)
- **Physiological Range**: 20-500 ms (validated)
- **Clinical Significance**: Isoelectric segment, important for ischemia detection

### 5. ST Interval

#### `ST_interval_ms`
- **Definition**: Duration from S-wave offset to T-wave offset
- **Calculation**: `(T_ri_idx - S_ri_idx) / sampling_rate * 1000.0`
- **Units**: Milliseconds (ms)
- **Physiological Range**: 5-700 ms (validated)
- **Clinical Significance**: Total ST segment + T-wave duration

### 6. QT Interval

#### `QT_interval_ms`
- **Definition**: Duration from QRS complex onset to T-wave offset
- **Calculation**: `(T_ri_idx - Q_le_idx) / sampling_rate * 1000.0`
- **Units**: Milliseconds (ms)
- **Physiological Range**: 200-750 ms (validated)
- **Clinical Significance**: Total ventricular activity (depolarization + repolarization)
- **Note**: Often corrected for heart rate (QTc)

### 7. RR Interval

#### `RR_interval_ms`
- **Definition**: Duration between consecutive R-peaks
- **Calculation**: `(current_R_global_center_idx - previous_R_global_center_idx) / sampling_rate * 1000.0`
- **Units**: Milliseconds (ms)
- **Physiological Range**: Configurable via `cfg.rr_bounds_ms` (default: 60-1800 ms for humans, 80-250 ms for mice)
- **Validation**: Must be within physiological bounds
- **Clinical Significance**: Heart rate variability, rhythm analysis
- **Note**: First cycle has `NaN` (no previous R-peak)

### 8. PP Interval

#### `PP_interval_ms`
- **Definition**: Duration between consecutive P-peaks
- **Calculation**: `(current_P_global_center_idx - previous_P_global_center_idx) / sampling_rate * 1000.0`
- **Units**: Milliseconds (ms)
- **Physiological Range**: Configurable via `cfg.pp_bounds_ms` (default: same as `rr_bounds_ms`)
- **Validation**: Must be within physiological bounds
- **Clinical Significance**: Atrial rate, useful for detecting atrial arrhythmias
- **Note**: First cycle or cycles without P-waves have `NaN`

---

### 9. QTc (Rate-Corrected QT Interval)

The QT interval varies with heart rate, so QTc (corrected QT) standardizes it to a common rate (typically 60 bpm) for clinical interpretation. PyHEARTS provides three correction formulas.

#### `QTc_Bazett_ms`
- **Definition**: QT interval corrected using Bazett's formula
- **Calculation**: `QT_ms / √(RR_ms / 1000)`
- **Units**: Milliseconds (ms)
- **Normal Range**: <440ms (males), <460ms (females)
- **Clinical Significance**: Most commonly used formula in clinical practice
- **Limitations**: Less accurate at very high (>100 bpm) or very low (<50 bpm) heart rates
- **Note**: Returns `NaN` if QT or RR interval is missing or invalid

#### `QTc_Fridericia_ms`
- **Definition**: QT interval corrected using Fridericia's formula
- **Calculation**: `QT_ms / (RR_ms / 1000)^(1/3)`
- **Units**: Milliseconds (ms)
- **Normal Range**: Similar to Bazett (<440-460ms)
- **Clinical Significance**: Often more accurate than Bazett at high heart rates
- **Advantages**: Uses cube root instead of square root, preferred in some research settings
- **Note**: Returns `NaN` if QT or RR interval is missing or invalid

#### `QTc_Framingham_ms`
- **Definition**: QT interval corrected using Framingham formula
- **Calculation**: `QT_ms + 0.154 × (1000 - RR_ms)`
- **Units**: Milliseconds (ms)
- **Normal Range**: Similar to Bazett (<440-460ms)
- **Clinical Significance**: Linear correction formula, alternative to power-based formulas
- **Advantages**: Simple additive approach
- **Note**: Returns `NaN` if QT or RR interval is missing or invalid

**Clinical Relevance of Prolonged QTc**:
- Increased risk of arrhythmias (especially Torsades de Pointes)
- Drug-induced QT prolongation
- Congenital long QT syndrome
- Electrolyte imbalances
- Myocardial ischemia

---

## Pairwise Voltage Differences

Voltage differences between specific wave pairs, useful for morphology analysis.

**Total**: 3 pairwise difference features

### 1. R-S Voltage Difference

#### `R_minus_S_voltage_diff_signed`
- **Definition**: Signed voltage difference between R-peak and S-peak
- **Calculation**: `R_gauss_height - S_gauss_height`
- **Units**: Millivolts (mV)
- **Mode**: `signed` (default) - preserves sign, or `absolute` (if configured)
- **Clinical Significance**: QRS complex amplitude, useful for axis determination
- **Note**: Positive values indicate R > S (normal), negative indicates inverted QRS

### 2. R-P Voltage Difference

#### `R_minus_P_voltage_diff_signed`
- **Definition**: Signed voltage difference between R-peak and P-peak
- **Calculation**: `R_gauss_height - P_gauss_height`
- **Units**: Millivolts (mV)
- **Mode**: `signed` (default) or `absolute`
- **Clinical Significance**: Relative amplitude of ventricular vs. atrial activity

### 3. T-R Voltage Difference

#### `T_minus_R_voltage_diff_signed`
- **Definition**: Signed voltage difference between T-peak and R-peak
- **Calculation**: `T_gauss_height - R_gauss_height`
- **Units**: Millivolts (mV)
- **Mode**: `signed` (default) or `absolute`
- **Clinical Significance**: T-wave amplitude relative to R-peak, useful for detecting T-wave abnormalities

**Configuration**: Pairs can be customized via `cfg.shape_interdeflection_pairs` (default: `[("R", "S"), ("R", "P"), ("T", "R")]`)

---

## Fit Quality Metrics

Measures of how well the Gaussian model fits the actual ECG signal.

**Total**: 2 fit quality features

### 1. R-Squared

#### `r_squared`
- **Definition**: Coefficient of determination (R²) for Gaussian fit
- **Calculation**: 
  ```python
  ss_res = sum((observed - fitted)²)
  ss_tot = sum((observed - mean(observed))²)
  r_squared = 1 - (ss_res / ss_tot)
  ```
- **Units**: Dimensionless (0-1, can be negative for poor fits)
- **Interpretation**:
  - `R² = 1.0`: Perfect fit
  - `R² > 0.9`: Excellent fit (75%+ of cycles achieve this)
  - `R² > 0.7`: Good fit
  - `R² < 0.5`: Poor fit
  - `R² < 0`: Fit worse than horizontal line
- **Note**: Computed for the entire cycle fit (all waves combined)

### 2. Root Mean Square Error

#### `rmse`
- **Definition**: Root mean square error between observed and fitted signal
- **Calculation**: `sqrt(mean((observed - fitted)²))`
- **Units**: Millivolts (mV)
- **Interpretation**: Lower values indicate better fit
- **Note**: Absolute error measure, complementary to R²

---

## Heart Rate Variability (HRV) Metrics

These are computed separately from the per-cycle features, using the RR interval series.

**Total**: 7 HRV metrics (computed via `compute_hrv_metrics()`)

### 1. Average Heart Rate

#### `average_heart_rate`
- **Definition**: Mean heart rate in beats per minute
- **Calculation**: 
  1. Convert RR intervals to heart rate: `HR = 60 / (RR_ms / 1000)`
  2. Compute mean: `average_heart_rate = mean(HR)`
  3. Round to nearest integer
- **Units**: Beats per minute (bpm)
- **Requirements**: Requires ≥60 RR intervals for reliable computation
- **Note**: NaN if insufficient data

### 2. SDNN

#### `sdnn`
- **Definition**: Standard deviation of NN intervals (normal-to-normal intervals)
- **Calculation**: `std(RR_intervals, ddof=1)` where `ddof=1` uses sample standard deviation
- **Units**: Milliseconds (ms)
- **Requirements**: Requires ≥2 RR intervals
- **Clinical Significance**: Overall HRV measure, reflects both short-term and long-term variability
- **Note**: NaN if insufficient data

### 3. RMSSD

#### `rmssd`
- **Definition**: Root mean square of successive differences
- **Calculation**: `sqrt(mean(diff(RR_intervals)²))`
- **Units**: Milliseconds (ms)
- **Requirements**: Requires ≥2 RR intervals
- **Clinical Significance**: Short-term HRV measure, reflects parasympathetic activity
- **Note**: More sensitive to rapid changes than SDNN

### 4. NN50

#### `nn50`
- **Definition**: Number of successive RR interval differences greater than 50 ms
- **Calculation**: `sum(abs(diff(RR_intervals)) > 50)`
- **Units**: Count (integer)
- **Requirements**: Requires ≥2 RR intervals
- **Clinical Significance**: Count of significant beat-to-beat variations
- **Note**: Related to pNN50 (percentage), which would be `NN50 / (N-1) * 100`

### 5. pNN50

#### `pnn50`
- **Definition**: Percentage of successive RR interval differences greater than 50 ms
- **Calculation**: `(NN50 / (N-1)) * 100` where N is the number of RR intervals
- **Units**: Percentage (float, 0-100)
- **Requirements**: Requires ≥2 RR intervals
- **Clinical Significance**: Short-term HRV measure, reflects parasympathetic activity
- **Interpretation**: Higher values indicate greater beat-to-beat variability
- **Note**: More sensitive than NN50 as it normalizes by the number of intervals

### 6. SD1 (Poincaré Plot)

#### `sd1`
- **Definition**: Short-term HRV from Poincaré plot (perpendicular to line of identity)
- **Calculation**: `RMSSD / √2`
- **Units**: Milliseconds (ms)
- **Requirements**: Requires ≥2 RR intervals
- **Clinical Significance**: Reflects short-term variability and parasympathetic activity
- **Interpretation**: Higher values indicate greater short-term HRV
- **Note**: SD1 is mathematically related to RMSSD: SD1 = RMSSD / √2

### 7. SD2 (Poincaré Plot)

#### `sd2`
- **Definition**: Long-term HRV from Poincaré plot (along line of identity)
- **Calculation**: `√(2 × SDNN² - SD1²)`
- **Units**: Milliseconds (ms)
- **Requirements**: Requires ≥2 RR intervals
- **Clinical Significance**: Reflects long-term variability and overall HRV
- **Interpretation**: Higher values indicate greater long-term HRV
- **Note**: SD2 captures variability along the line of identity in the Poincaré plot

**Computation**: Called via `PyHEARTS.compute_hrv_metrics()` after `analyze_ecg()`

**Poincaré Plot Notes**:
- The Poincaré plot is a scatter plot of RR(n) vs RR(n+1)
- SD1 measures width of the cloud (perpendicular to line of identity) = short-term variability
- SD2 measures length of the cloud (along line of identity) = long-term variability
- SD1/SD2 ratio provides additional insight into the balance of short-term vs long-term variability

---

## Beat-to-Beat Variability Metrics

These metrics quantify the variability of morphological features across cardiac cycles, computed from the per-cycle feature series.

**Total**: Variable (5 metrics × N priority features, typically 45-70 metrics depending on detected features)

**Computation**: Called automatically after `analyze_ecg()` via `PyHEARTS.compute_variability_metrics()`, or manually via `PyHEARTS.compute_variability_metrics(priority_features=...)`

### Default Priority Features

Variability is computed for the following features by default:
- **Intervals**: `QT_interval_ms`, `QRS_interval_ms`, `PR_interval_ms`, `RR_interval_ms`
- **QTc**: `QTc_Bazett_ms`, `QTc_Fridericia_ms`
- **Wave Amplitudes**: `R_gauss_height`, `P_gauss_height`, `T_gauss_height`
- **Wave Durations**: `R_duration_ms`, `P_duration_ms`, `T_duration_ms`
- **Segments**: `ST_segment_ms`

### Variability Metrics (per feature)

For each priority feature, the following 5 metrics are computed:

#### `{Feature}_std`
- **Definition**: Standard deviation of the feature across cycles
- **Calculation**: `std(feature_series, ddof=1)` (sample standard deviation)
- **Units**: Same as the feature (ms for intervals, mV for amplitudes, etc.)
- **Interpretation**: Higher values indicate greater beat-to-beat variability
- **Clinical Significance**: Increased variability in QT intervals is associated with arrhythmia risk

#### `{Feature}_cv`
- **Definition**: Coefficient of variation (normalized variability)
- **Calculation**: `std / abs(mean)`
- **Units**: Dimensionless ratio
- **Interpretation**: 
  - CV < 0.1: Low variability
  - CV 0.1-0.3: Moderate variability
  - CV > 0.3: High variability
- **Clinical Significance**: Normalizes variability by mean, useful for comparing features with different scales

#### `{Feature}_iqr`
- **Definition**: Interquartile range (75th percentile - 25th percentile)
- **Calculation**: `percentile(feature_series, 75) - percentile(feature_series, 25)`
- **Units**: Same as the feature
- **Interpretation**: Robust measure of spread, less sensitive to outliers than range
- **Clinical Significance**: IQR provides a robust measure of variability that is less affected by extreme values

#### `{Feature}_mad`
- **Definition**: Median absolute deviation (robust measure of variability)
- **Calculation**: `1.4826 × median(abs(feature_series - median(feature_series)))`
- **Units**: Same as the feature
- **Interpretation**: Robust alternative to standard deviation, resistant to outliers
- **Clinical Significance**: MAD is less sensitive to outliers than standard deviation, useful for noisy signals

#### `{Feature}_range`
- **Definition**: Range (maximum - minimum)
- **Calculation**: `max(feature_series) - min(feature_series)`
- **Units**: Same as the feature
- **Interpretation**: Total spread of values across cycles
- **Clinical Significance**: Range indicates the full extent of beat-to-beat variation

**Requirements**: Requires ≥2 valid (non-NaN) values for the feature across cycles. Returns `NaN` if insufficient data.

**Access**: Variability metrics are stored in `PyHEARTS.variability_metrics` dictionary after calling `analyze_ecg()`.

**Example**:
```python
from pyhearts import PyHEARTS

hearts = PyHEARTS(sampling_rate=250.0, species='human')
output_df, epochs_df = hearts.analyze_ecg(ecg_signal)

# Variability metrics computed automatically
qt_std = hearts.variability_metrics.get('QT_interval_ms_std')
qt_cv = hearts.variability_metrics.get('QT_interval_ms_cv')
r_height_std = hearts.variability_metrics.get('R_gauss_height_std')
```

---

## Feature Summary by Category

### Per-Wave Features (30 features × 5 waves = 150 features)

1. **Global Indices** (3): `global_center_idx`, `global_le_idx`, `global_ri_idx`
2. **Time-Domain** (3): `center_ms`, `le_ms`, `ri_ms`
3. **Local Indices** (3): `center_idx`, `le_idx`, `ri_idx`
4. **Gaussian Parameters** (6): `gauss_center`, `gauss_height`, `gauss_stdev_samples`, `gauss_stdev_ms`, `gauss_fwhm_samples`, `gauss_fwhm_ms`
5. **FWHM Boundaries** (6): `fwhm_le_idx`, `fwhm_ri_idx`, `fwhm_le_ms`, `fwhm_ri_ms`, `fwhm_global_le_idx`, `fwhm_global_ri_idx`
6. **Amplitudes** (3): `center_voltage`, `le_voltage`, `ri_voltage`
7. **Morphology** (8): `duration_ms`, `rise_ms`, `decay_ms`, `rdsm`, `sharpness`, `max_upslope_mv_per_s`, `max_downslope_mv_per_s`, `slope_asymmetry`
8. **Integral** (1): `voltage_integral_uv_ms`

### Interval Features (11 features)

1. `PR_interval_ms`
2. `PR_segment_ms`
3. `QRS_interval_ms`
4. `ST_segment_ms`
5. `ST_interval_ms`
6. `QT_interval_ms`
7. `RR_interval_ms`
8. `PP_interval_ms`
9. `QTc_Bazett_ms` (rate-corrected QT using Bazett's formula)
10. `QTc_Fridericia_ms` (rate-corrected QT using Fridericia's formula)
11. `QTc_Framingham_ms` (rate-corrected QT using Framingham formula)

### Pairwise Differences (3 features)

1. `R_minus_S_voltage_diff_signed`
2. `R_minus_P_voltage_diff_signed`
3. `T_minus_R_voltage_diff_signed`

### Fit Quality (2 features)

1. `r_squared`
2. `rmse`

### HRV Metrics (7 features, computed separately)

1. `average_heart_rate`
2. `sdnn`
3. `rmssd`
4. `nn50`
5. `pnn50`
6. `sd1`
7. `sd2`

### Variability Metrics (~45-70 features, computed separately)

For each priority feature, 5 metrics are computed:
- `{Feature}_std`: Standard deviation
- `{Feature}_cv`: Coefficient of variation
- `{Feature}_iqr`: Interquartile range
- `{Feature}_mad`: Median absolute deviation
- `{Feature}_range`: Range

Default priority features (9-14 features, depending on detection):
- Intervals: `QT_interval_ms`, `QRS_interval_ms`, `PR_interval_ms`, `RR_interval_ms`
- QTc: `QTc_Bazett_ms`, `QTc_Fridericia_ms`
- Amplitudes: `R_gauss_height`, `P_gauss_height`, `T_gauss_height`
- Durations: `R_duration_ms`, `P_duration_ms`, `T_duration_ms`
- Segments: `ST_segment_ms`

---

## Total Feature Count

- **Per-cycle features**: 165 (morphological) + 11 (intervals) + 3 (pairwise) + 2 (fit quality) = **181 features per cycle**
- **HRV metrics**: 7 features (computed once per recording)
- **Variability metrics**: ~45-70 features (5 metrics × 9-14 priority features, computed once per recording)
- **Grand total**: **~240 features** per ECG recording

---

## Notes on Missing Values

- Features are stored as `NaN` if:
  - Wave detection failed
  - Gaussian fitting failed
  - Validation failed (physiological constraints)
  - Required previous cycle data is missing (for RR/PP intervals)
- Interpolation is attempted for interval calculations using surrounding cycles (window_size=3)
- Missing values allow downstream analysis to handle incomplete data gracefully

---

## Feature Naming Convention

All features follow the pattern: `{Wave}_{FeatureName}`

- **Wave**: `P`, `Q`, `R`, `S`, `T`
- **FeatureName**: Descriptive name (e.g., `center_ms`, `gauss_height`)
- **Exceptions**: 
  - Interval features: `{Interval}_interval_ms` or `{Interval}_segment_ms`
  - Pairwise differences: `{Wave1}_minus_{Wave2}_voltage_diff_{mode}`
  - Fit quality: `r_squared`, `rmse` (no wave prefix)

---

## Accessing Features

Features are stored in the `output_df` DataFrame returned by `PyHEARTS.analyze_ecg()`:

```python
from pyhearts import PyHEARTS

hearts = PyHEARTS(sampling_rate=250.0, species='human')
output_df, epochs_df = hearts.analyze_ecg(ecg_signal)

# Access features
r_heights = output_df['R_gauss_height']  # R-peak amplitudes
qt_intervals = output_df['QT_interval_ms']  # QT intervals
pr_intervals = output_df['PR_interval_ms']  # PR intervals

# HRV metrics (computed separately)
hearts.compute_hrv_metrics()
print(hearts.hrv_metrics)  # Dictionary with HRV metrics

# Variability metrics (computed automatically)
print(hearts.variability_metrics)  # Dictionary with variability metrics
```

---

## References

- Feature extraction: `pyhearts/feature/shape.py`, `pyhearts/feature/intervals.py`
- Gaussian fitting: `pyhearts/processing/gaussian.py`
- Interval calculation: `pyhearts/feature/intervals.py`
- HRV computation: `pyhearts/feature/hrv.py`
- Variability computation: `pyhearts/feature/variability.py`
- Main processing: `pyhearts/processing/processcycle.py`

