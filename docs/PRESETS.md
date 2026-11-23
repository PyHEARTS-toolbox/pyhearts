# PyHEARTS configuration

## Standard pipeline

`pyhearts.standard.run_standard` is the recommended entry point. It applies the
same processing to every recording:

- 0.5 to 50 Hz zero-phase Butterworth band-pass (order 4) and a 50 or 60 Hz
  notch (Q = 30)
- a check that the sampling rate matches the rate documented for the dataset
- linear bridging of missing samples, exclusion of recordings with too many
  missing samples, and removal of beats near missing data
- the human analyzer preset described below

```python
from pyhearts.standard import run_standard

ph, summary = run_standard(ecg, fs=500, expected_fs=500, mains_hz=60,
                           record_id="subject_001", results_dir="results")
```

`summary` and the saved metadata JSON record the pipeline version
(`standard-2`), the filter settings and the missing-data counts.

## Analyzer presets

`PyHEARTS(..., species=...)` selects the preset.

| Setting | Behavior |
|---------|----------|
| `species="human"` | Human morphology preset, record-level T stage and wavelet boundary layer |
| `species=None` | Species-agnostic morphology defaults, record-level T stage and wavelet boundary layer |
| `species="mouse"` | Mouse morphology preset; record-level T stage off; intervals from Gaussian edges |

```python
from pyhearts import PyHEARTS

analyzer = PyHEARTS(sampling_rate=500.0, species="human")
features, cycles = analyzer.analyze_ecg(filtered_ecg)
```

Advanced settings live in `pyhearts.config.ProcessCycleConfig`. Typical runs
should not need them.

## What happens in `analyze_ecg`

1. R peaks are detected and the signal is split into cardiac cycles.
2. One Gaussian is fitted to each of the P, Q, R, S and T waves of every cycle.
3. The record-level T stage sets the global T-wave fiducial
   (`T_global_center_idx`).
4. Except for the mouse preset, a wavelet boundary layer anchored on the fitted peaks
   places wave onsets and offsets, and the clinical intervals (PR, QRS, QT and
   others) are computed from these boundaries.
5. `save_output` records the package version, pipeline tag and resolved
   settings.

R-peak detection corrects inverted QRS polarity by default
(`rpeak_auto_polarity=True`); `analyzer.signal_inverted` records whether the
signal was flipped.

## Heart rate variability

`analyzer.compute_hrv_metrics()` uses the detected R-peak series, gated by
`rr_bounds_ms`. It needs at least 60 valid RR intervals and returns mean heart
rate, SDNN, RMSSD and NN50.
