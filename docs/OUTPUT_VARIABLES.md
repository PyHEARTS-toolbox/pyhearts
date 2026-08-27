# PyHEARTS Output Variables Reference

Canonical feature set for `analyze_ecg` / `{file_id}_pyhearts.csv` and optional
HRV outputs. For threshold and species-specific knobs, see
`ProcessCycleConfig` (`pyhearts.config` / `pyhearts._morphology.config`).

## Feature set overview

`analyze_ecg` returns one row per retained cycle. Named columns (136) plus
the saved index `cycle_index`:

| Family | Count | Columns |
|---|---|---|
| Cycle quality | 3 | `cycle_trend`, `r_squared`, `rmse` |
| Per-wave morphology | 120 | 24 metrics × P, Q, R, S, T (list below) |
| Intervals | 8 | `PR_interval_ms`, `PR_segment_ms`, `QRS_interval_ms`, `ST_segment_ms`, `ST_interval_ms`, `QT_interval_ms`, `PP_interval_ms`, `RR_interval_ms` |
| Pairwise voltages | 3 | `R_minus_S_voltage_diff_signed`, `R_minus_P_voltage_diff_signed`, `T_minus_R_voltage_diff_signed` |
| Record-T extras | 2 | `T_gaussian_global_center_idx`, `t_source` |

Per-wave metrics (each prefixed `P_`, `Q_`, `R_`, `S_`, `T_`):

`global_center_idx`, `global_le_idx`, `global_ri_idx`, `center_ms`, `le_ms`,
`ri_ms`, `center_idx`, `le_idx`, `ri_idx`, `gauss_center`, `gauss_height`,
`gauss_stdev_samples`, `gauss_stdev_ms`, `gauss_fwhm_samples`, `gauss_fwhm_ms`,
`center_voltage`, `le_voltage`, `ri_voltage`, `duration_ms`, `rise_ms`,
`decay_ms`, `rdsm`, `sharpness`, `voltage_integral_uv_ms`

Optional record-level HRV (`compute_hrv_metrics`, ≥60 valid RR intervals):
`average_heart_rate`, `sdnn`, `rmssd`, `nn50`, `rr_source`, `n_r_peaks`,
`n_rr_intervals`. Optional `{file_id}_rr_intervals.csv` is the gated RR
series used for those metrics (`rr_interval_ms`).

## Per-cycle CSV (`*_pyhearts.csv`)

The table lists each column emitted by `{file_id}_pyhearts.csv`, with a
plain-language description and how it is computed.

**Notes**

- `cycle_index` is the DataFrame index (written when saving with
  `index=True`); it is not a named feature column.
- Wave **edges**, **sharpness**, and **voltage integrals** are computed on the
  **fitted** sum of Gaussians. Point voltages (`*_voltage`) are read from the
  **detrended** cycle.
- Edge threshold: `thr = threshold_fraction × gauss_height` (default
  `threshold_fraction = 0.30`). Positive waves search while `fit > thr`;
  negative waves (Q, S) while `fit < thr`. Search is capped by
  `min(shape_search_scale · σ, physiology window)`. Waves are discarded if
  duration &lt; `duration_min_ms` (20 ms human / default, 2 ms mouse), if rise
  or decay ≤ 0, or if concavity checks fail.
- Center refinement: fresh fit → local extremum within ±10 samples on the
  detrended cycle; warm-started (seeded) fit → within ±½ FWHM.
- Public human / species-agnostic runs may overwrite `T_global_center_idx`
  with the record-level T fiducial; the morphology Gaussian center is kept in
  `T_gaussian_global_center_idx`. Other T morphology columns remain from the
  Gaussian pass.

| Feature Name | Description | Calculation |
|---|---|---|
| `cycle_index` | Cycle number (0-based) within the record | Index assigned during epoching; increments per retained R–R cycle |
| `cycle_trend` | Estimated linear trend slope within the cycle | After median baseline subtraction over `detrend_window_ms`, slope from `linregress` through the first and last corrected samples (slope per sample) |
| `r_squared` | Goodness-of-fit (R²) of Gaussian model | Squared Pearson correlation between detrended cycle and fitted sum of Gaussians |
| `rmse` | Root mean square error of fit | RMSE between detrended cycle and fitted waveform |
| `P_global_center_idx` | P wave center (absolute sample index) | `xs_samples[center_idx]` after center refinement |
| `P_global_le_idx` | P wave left edge (absolute sample index) | `xs_samples[le_idx]` from std-guided threshold on fitted waveform |
| `P_global_ri_idx` | P wave right edge (absolute sample index) | `xs_samples[ri_idx]` from std-guided threshold on fitted waveform |
| `P_center_ms` | P wave center time (ms) | `(global_center_idx / fs) × 1000` |
| `P_le_ms` | P wave left-edge time (ms) | `(global_le_idx / fs) × 1000` |
| `P_ri_ms` | P wave right-edge time (ms) | `(global_ri_idx / fs) × 1000` |
| `Q_global_center_idx` | Q wave center (absolute sample index) | `xs_samples[center_idx]` after center refinement |
| `Q_global_le_idx` | Q wave left edge (absolute sample index) | `xs_samples[le_idx]` from std-guided threshold on fitted waveform |
| `Q_global_ri_idx` | Q wave right edge (absolute sample index) | `xs_samples[ri_idx]` from std-guided threshold on fitted waveform |
| `Q_center_ms` | Q wave center time (ms) | `(global_center_idx / fs) × 1000` |
| `Q_le_ms` | Q wave left-edge time (ms) | `(global_le_idx / fs) × 1000` |
| `Q_ri_ms` | Q wave right-edge time (ms) | `(global_ri_idx / fs) × 1000` |
| `R_global_center_idx` | R peak center (absolute sample index) | `xs_samples[center_idx]` after center refinement |
| `R_global_le_idx` | R peak left edge (absolute sample index) | `xs_samples[le_idx]` from std-guided threshold on fitted waveform |
| `R_global_ri_idx` | R peak right edge (absolute sample index) | `xs_samples[ri_idx]` from std-guided threshold on fitted waveform |
| `R_center_ms` | R peak center time (ms) | `(global_center_idx / fs) × 1000` |
| `R_le_ms` | R peak left-edge time (ms) | `(global_le_idx / fs) × 1000` |
| `R_ri_ms` | R peak right-edge time (ms) | `(global_ri_idx / fs) × 1000` |
| `S_global_center_idx` | S wave center (absolute sample index) | `xs_samples[center_idx]` after center refinement |
| `S_global_le_idx` | S wave left edge (absolute sample index) | `xs_samples[le_idx]` from std-guided threshold on fitted waveform |
| `S_global_ri_idx` | S wave right edge (absolute sample index) | `xs_samples[ri_idx]` from std-guided threshold on fitted waveform |
| `S_center_ms` | S wave center time (ms) | `(global_center_idx / fs) × 1000` |
| `S_le_ms` | S wave left-edge time (ms) | `(global_le_idx / fs) × 1000` |
| `S_ri_ms` | S wave right-edge time (ms) | `(global_ri_idx / fs) × 1000` |
| `T_global_center_idx` | T wave center (absolute sample index) | With record-T enabled: assigned record-level T sample (may fall back to Gaussian). Without record-T: `xs_samples[center_idx]` after center refinement |
| `T_global_le_idx` | T wave left edge (absolute sample index) | `xs_samples[le_idx]` from std-guided threshold on fitted waveform |
| `T_global_ri_idx` | T wave right edge (absolute sample index) | `xs_samples[ri_idx]` from std-guided threshold on fitted waveform |
| `T_center_ms` | T wave center time (ms) | `(global_center_idx / fs) × 1000` |
| `T_le_ms` | T wave left-edge time (ms) | `(global_le_idx / fs) × 1000` |
| `T_ri_ms` | T wave right-edge time (ms) | `(global_ri_idx / fs) × 1000` |
| `T_gaussian_global_center_idx` | Morphology Gaussian T center (absolute sample index) | Copy of pre-merge Gaussian `T_global_center_idx`; retained when record-T overwrites `T_global_center_idx` |
| `t_source` | Provenance of `T_global_center_idx` | One of `record_t`, `record_t_miss`, `gaussian_fill`, `gaussian_later`, or `missing` |
| `P_center_idx` | P wave center (relative index) | Rounded fitted μ refined to local extremum within ±10 samples (or ±½ FWHM for seeded) |
| `P_le_idx` | P wave left edge (relative index) | Search left on fitted waveform from P center while `fit > thr`; subject to std/physiology caps and `duration_min_ms` |
| `P_ri_idx` | P wave right edge (relative index) | Search right on fitted waveform from P center while `fit > thr`; subject to std/physiology caps and `duration_min_ms` |
| `P_gauss_center` | P wave Gaussian center (samples) | Fitted μ from `curve_fit` |
| `P_gauss_height` | P wave Gaussian amplitude (mV) | Fitted peak height on detrended signal |
| `P_gauss_stdev_samples` | P wave Gaussian σ (samples) | Fitted σ; initial guess from half-height width / 2.3548, floored at 0.5 samples |
| `P_gauss_stdev_ms` | P wave Gaussian σ (ms) | `σ(samples) × (1000 / fs)` |
| `P_gauss_fwhm_samples` | P wave Gaussian FWHM (samples) | `2√(2 ln 2) · σ` |
| `P_gauss_fwhm_ms` | P wave Gaussian FWHM (ms) | `FWHM(samples) × (1000 / fs)` |
| `P_center_voltage` | P wave amplitude at center (mV) | Detrended signal value at `center_idx` |
| `P_le_voltage` | P wave amplitude at left edge (mV) | Detrended signal at `le_idx` |
| `P_ri_voltage` | P wave amplitude at right edge (mV) | Detrended signal at `ri_idx` |
| `P_duration_ms` | P wave duration (ms) | `(ri_idx − le_idx) × (1000 / fs)`; must exceed species minimum |
| `P_rise_ms` | P wave rise time (ms) | `(center_idx − le_idx) × (1000 / fs)` |
| `P_decay_ms` | P wave decay time (ms) | `(ri_idx − center_idx) × (1000 / fs)` |
| `P_rdsm` | P wave rise-decay symmetry | `(center_idx − le_idx) / (ri_idx − le_idx)` |
| `P_sharpness` | P wave sharpness (normalized) | On fitted segment `[le_idx, ri_idx]` (Savitzky–Golay 7/3 if length ≥ 7): P95(`\|dV/dt\|`) / (P95 − P5 of segment); same default for all waves |
| `P_voltage_integral_uv_ms` | P wave voltage integral (µV·ms) | Trapezoidal integral of **fitted** mV over inclusive `[le_idx, ri_idx]` with `dx = 1/fs`, × 1e6 |
| `Q_center_idx` | Q wave center (relative index) | Rounded fitted μ refined to local extremum within ±10 samples (or ±½ FWHM for seeded) |
| `Q_le_idx` | Q wave left edge (relative index) | Search left on fitted waveform while `fit < thr` (Q negative); subject to caps and `duration_min_ms` |
| `Q_ri_idx` | Q wave right edge (relative index) | Search right on fitted waveform while `fit < thr` (Q negative); subject to caps and `duration_min_ms` |
| `Q_gauss_center` | Q wave Gaussian center (samples) | Fitted μ from `curve_fit` |
| `Q_gauss_height` | Q wave Gaussian amplitude (mV) | Fitted peak height on detrended signal |
| `Q_gauss_stdev_samples` | Q wave Gaussian σ (samples) | Fitted σ; initial guess from half-height width / 2.3548, floored at 0.5 samples |
| `Q_gauss_stdev_ms` | Q wave Gaussian σ (ms) | `σ(samples) × (1000 / fs)` |
| `Q_gauss_fwhm_samples` | Q wave Gaussian FWHM (samples) | `2√(2 ln 2) · σ` |
| `Q_gauss_fwhm_ms` | Q wave Gaussian FWHM (ms) | `FWHM(samples) × (1000 / fs)` |
| `Q_center_voltage` | Q wave amplitude at center (mV) | Detrended signal value at `center_idx` |
| `Q_le_voltage` | Q wave amplitude at left edge (mV) | Detrended signal at `le_idx` |
| `Q_ri_voltage` | Q wave amplitude at right edge (mV) | Detrended signal at `ri_idx` |
| `Q_duration_ms` | Q wave duration (ms) | `(ri_idx − le_idx) × (1000 / fs)`; must exceed species minimum |
| `Q_rise_ms` | Q wave rise time (ms) | `(center_idx − le_idx) × (1000 / fs)` |
| `Q_decay_ms` | Q wave decay time (ms) | `(ri_idx − center_idx) × (1000 / fs)` |
| `Q_rdsm` | Q wave rise-decay symmetry | `(center_idx − le_idx) / (ri_idx − le_idx)` |
| `Q_sharpness` | Q wave sharpness (normalized) | Same as `P_sharpness` (P95 `\|dV/dt\|` / robust P95−P5 on fitted segment) |
| `Q_voltage_integral_uv_ms` | Q wave voltage integral (µV·ms) | Trapezoidal integral of **fitted** mV over inclusive `[le_idx, ri_idx]` with `dx = 1/fs`, × 1e6 |
| `R_center_idx` | R peak center (relative index) | Rounded fitted μ refined to local extremum within ±10 samples (or ±½ FWHM for seeded) |
| `R_le_idx` | R peak left edge (relative index) | Search left on fitted waveform while `fit > thr` (R positive); subject to caps and `duration_min_ms` |
| `R_ri_idx` | R peak right edge (relative index) | Search right on fitted waveform while `fit > thr` (R positive); subject to caps and `duration_min_ms` |
| `R_gauss_center` | R peak Gaussian center (samples) | Fitted μ from `curve_fit` |
| `R_gauss_height` | R peak Gaussian amplitude (mV) | Fitted peak height on detrended signal |
| `R_gauss_stdev_samples` | R peak Gaussian σ (samples) | Fitted σ; initial guess from half-height width / 2.3548, floored at 0.5 samples |
| `R_gauss_stdev_ms` | R peak Gaussian σ (ms) | `σ(samples) × (1000 / fs)` |
| `R_gauss_fwhm_samples` | R peak Gaussian FWHM (samples) | `2√(2 ln 2) · σ` |
| `R_gauss_fwhm_ms` | R peak Gaussian FWHM (ms) | `FWHM(samples) × (1000 / fs)` |
| `R_center_voltage` | R peak amplitude at center (mV) | Detrended signal value at `center_idx` |
| `R_le_voltage` | R peak amplitude at left edge (mV) | Detrended signal at `le_idx` |
| `R_ri_voltage` | R peak amplitude at right edge (mV) | Detrended signal at `ri_idx` |
| `R_duration_ms` | R peak duration (ms) | `(ri_idx − le_idx) × (1000 / fs)`; must exceed species minimum |
| `R_rise_ms` | R peak rise time (ms) | `(center_idx − le_idx) × (1000 / fs)` |
| `R_decay_ms` | R peak decay time (ms) | `(ri_idx − center_idx) × (1000 / fs)` |
| `R_rdsm` | R peak rise-decay symmetry | `(center_idx − le_idx) / (ri_idx − le_idx)` |
| `R_sharpness` | R peak sharpness (normalized) | Same as `P_sharpness` (P95 for all waves; not a 5th-percentile special case) |
| `R_voltage_integral_uv_ms` | R peak voltage integral (µV·ms) | Trapezoidal integral of **fitted** mV over inclusive `[le_idx, ri_idx]` with `dx = 1/fs`, × 1e6 |
| `S_center_idx` | S wave center (relative index) | Rounded fitted μ refined to local extremum within ±10 samples (or ±½ FWHM for seeded) |
| `S_le_idx` | S wave left edge (relative index) | Search left on fitted waveform while `fit < thr` (S negative); subject to caps and `duration_min_ms` |
| `S_ri_idx` | S wave right edge (relative index) | Search right on fitted waveform while `fit < thr` (S negative); subject to caps and `duration_min_ms` |
| `S_gauss_center` | S wave Gaussian center (samples) | Fitted μ from `curve_fit` |
| `S_gauss_height` | S wave Gaussian amplitude (mV) | Fitted peak height on detrended signal |
| `S_gauss_stdev_samples` | S wave Gaussian σ (samples) | Fitted σ; initial guess from half-height width / 2.3548, floored at 0.5 samples |
| `S_gauss_stdev_ms` | S wave Gaussian σ (ms) | `σ(samples) × (1000 / fs)` |
| `S_gauss_fwhm_samples` | S wave Gaussian FWHM (samples) | `2√(2 ln 2) · σ` |
| `S_gauss_fwhm_ms` | S wave Gaussian FWHM (ms) | `FWHM(samples) × (1000 / fs)` |
| `S_center_voltage` | S wave amplitude at center (mV) | Detrended signal value at `center_idx` |
| `S_le_voltage` | S wave amplitude at left edge (mV) | Detrended signal at `le_idx` |
| `S_ri_voltage` | S wave amplitude at right edge (mV) | Detrended signal at `ri_idx` |
| `S_duration_ms` | S wave duration (ms) | `(ri_idx − le_idx) × (1000 / fs)`; must exceed species minimum |
| `S_rise_ms` | S wave rise time (ms) | `(center_idx − le_idx) × (1000 / fs)` |
| `S_decay_ms` | S wave decay time (ms) | `(ri_idx − center_idx) × (1000 / fs)` |
| `S_rdsm` | S wave rise-decay symmetry | `(center_idx − le_idx) / (ri_idx − le_idx)` |
| `S_sharpness` | S wave sharpness (normalized) | Same as `P_sharpness` (P95 `\|dV/dt\|` / robust P95−P5 on fitted segment) |
| `S_voltage_integral_uv_ms` | S wave voltage integral (µV·ms) | Trapezoidal integral of **fitted** mV over inclusive `[le_idx, ri_idx]` with `dx = 1/fs`, × 1e6 |
| `T_center_idx` | T wave center (relative index) | Rounded fitted μ refined to local extremum within ±10 samples (or ±½ FWHM for seeded) |
| `T_le_idx` | T wave left edge (relative index) | Search left on fitted waveform while `fit > thr` (T positive); subject to caps and `duration_min_ms` |
| `T_ri_idx` | T wave right edge (relative index) | Search right on fitted waveform while `fit > thr` (T positive); subject to caps and `duration_min_ms` |
| `T_gauss_center` | T wave Gaussian center (samples) | Fitted μ from `curve_fit` |
| `T_gauss_height` | T wave Gaussian amplitude (mV) | Fitted peak height on detrended signal |
| `T_gauss_stdev_samples` | T wave Gaussian σ (samples) | Fitted σ; initial guess from half-height width / 2.3548, floored at 0.5 samples |
| `T_gauss_stdev_ms` | T wave Gaussian σ (ms) | `σ(samples) × (1000 / fs)` |
| `T_gauss_fwhm_samples` | T wave Gaussian FWHM (samples) | `2√(2 ln 2) · σ` |
| `T_gauss_fwhm_ms` | T wave Gaussian FWHM (ms) | `FWHM(samples) × (1000 / fs)` |
| `T_center_voltage` | T wave amplitude at center (mV) | Detrended signal value at `center_idx` |
| `T_le_voltage` | T wave amplitude at left edge (mV) | Detrended signal at `le_idx` |
| `T_ri_voltage` | T wave amplitude at right edge (mV) | Detrended signal at `ri_idx` |
| `T_duration_ms` | T wave duration (ms) | `(ri_idx − le_idx) × (1000 / fs)`; must exceed species minimum |
| `T_rise_ms` | T wave rise time (ms) | `(center_idx − le_idx) × (1000 / fs)` |
| `T_decay_ms` | T wave decay time (ms) | `(ri_idx − center_idx) × (1000 / fs)` |
| `T_rdsm` | T wave rise-decay symmetry | `(center_idx − le_idx) / (ri_idx − le_idx)` |
| `T_sharpness` | T wave sharpness (normalized) | Same as `P_sharpness` (P95 `\|dV/dt\|` / robust P95−P5 on fitted segment) |
| `T_voltage_integral_uv_ms` | T wave voltage integral (µV·ms) | Trapezoidal integral of **fitted** mV over inclusive `[le_idx, ri_idx]` with `dx = 1/fs`, × 1e6 |
| `PR_interval_ms` | PR interval duration (ms) | `(Q_le − P_le) × (1000 / fs)` on relative indices; missing edges imputed from ±3-cycle mean; gated to physiologic limits |
| `PR_segment_ms` | PR segment duration (ms) | `(Q_le − P_ri) × (1000 / fs)`; imputed / gated as above |
| `QRS_interval_ms` | QRS duration (ms) | `(S_ri − Q_le) × (1000 / fs)`; imputed / gated as above |
| `ST_segment_ms` | ST segment duration (ms) | `(T_le − S_ri) × (1000 / fs)`; imputed / gated as above |
| `ST_interval_ms` | ST interval duration (ms) | `(T_ri − S_ri) × (1000 / fs)`; imputed / gated as above |
| `QT_interval_ms` | QT interval duration (ms) | `(T_ri − Q_le) × (1000 / fs)`; imputed / gated as above |
| `R_minus_S_voltage_diff_signed` | Signed voltage difference R − S | Difference of fitted Gaussian heights: `amp(R) − amp(S)` |
| `R_minus_P_voltage_diff_signed` | Signed voltage difference R − P | Difference of fitted Gaussian heights: `amp(R) − amp(P)` |
| `T_minus_R_voltage_diff_signed` | Signed voltage difference T − R | Difference of fitted Gaussian heights: `amp(T) − amp(R)` |
| `RR_interval_ms` | RR interval between successive morphology R centers (ms) | `Δsamples(R_global_center) × (1000 / fs)`; gated to `rr_bounds_ms` |
| `PP_interval_ms` | PP interval between successive morphology P centers (ms) | `Δsamples(P_global_center) × (1000 / fs)`; gated to `pp_bounds_ms` (defaults to `rr_bounds_ms`) |

## Heart-rate variability CSV (`*_hrv_metrics.csv`)

HRV is **not** part of the per-cycle CSV. Call
`analyzer.compute_hrv_metrics()` then `analyzer.save_hrv_metrics(...)` to write
a one-row `{file_id}_hrv_metrics.csv`.

Intervals come from the **detected R-peak series** (before epoch / morphology
retention), gated by `rr_bounds_ms`. Computation requires ≥ 60 valid RR
intervals. Metric values are rounded to integers.

| Feature Name | Description | Calculation |
|---|---|---|
| `average_heart_rate` | Mean heart rate (bpm) | Instantaneous `HRᵢ = 60 / (RRᵢ / 1000)`; average over valid RR intervals; rounded to int |
| `sdnn` | Standard deviation of NN intervals (ms) | Sample SDNN with `ddof=1`: `sqrt( Σ (RRᵢ − mean(RR))² / (N − 1) )`; rounded to int |
| `rmssd` | Root mean square of successive differences (ms) | `dᵢ = RRᵢ₊₁ − RRᵢ`; `RMSSD = sqrt( mean(dᵢ²) )` (= `1/N` over successive diffs); rounded to int |
| `nn50` | Count of successive \|ΔRR\| &gt; 50 ms | `NN50 = Σ 1(\|RRᵢ₊₁ − RRᵢ\| &gt; 50)` |
| `rr_source` | Provenance of RR series | Always `"r_peaks"` for the public HRV path |
| `n_r_peaks` | Number of detected R peaks | `len(r_peak_indices)` |
| `n_rr_intervals` | Number of gated RR intervals used | Count of RR intervals remaining after `rr_bounds_ms` filtering |

## RR-interval series (`*_rr_intervals.csv`)

Written by `analyzer.save_rr_intervals(...)` after `compute_hrv_metrics()`.
One row per gated RR interval from the detected R-peak series (not
morphology-retained `RR_interval_ms`).

| Feature Name | Description | Calculation |
|---|---|---|
| `rr_interval_ms` | Successive R–R interval (ms) | `Δsamples(r_peak) × (1000 / fs)`, kept only if inside `rr_bounds_ms` |
