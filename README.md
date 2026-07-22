# PyHEARTS

**Python Heart Evaluation and Analysis for Rhythm and Temporal Shape**

PyHEARTS is a Python toolbox for beat-by-beat ECG morphology analysis. It takes
a single-lead ECG, detects cardiac cycles, fits physiologically constrained
Gaussian waveforms, and returns a structured feature table for each beat.

The analyzer:

1. Detects R/P peaks, segments cycles, and fits symmetric Gaussians for P, Q,
   R, S, and T waves.
2. Applies a record-level T stage at the end of `analyze_ecg` to set the
   global T-wave fiducial (`T_global_center_idx`).

Use it for interpretable ECG phenotyping, interval timing, and morphology
features in research settings.

## Installation

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Optional extras:

```bash
python -m pip install ".[sim]"   # NeuroKit2 ECG simulation
python -m pip install ".[wfdb]"  # PhysioNet / WFDB loaders
python -m pip install ".[all]"   # both
```

For development:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Quick start

Minimal path: construct `PyHEARTS`, preprocess (recommended), then analyze.

```python
import numpy as np
from pyhearts import PyHEARTS

sampling_rate = 500.0
ecg = np.asarray(..., dtype=float)  # 1-D single-lead ECG

analyzer = PyHEARTS(sampling_rate=sampling_rate, species="human")

filtered = analyzer.preprocess_signal(
    ecg,
    highpass_cutoff=0.5,
    lowpass_cutoff=50.0,
    filter_order=4,
    notch_frequency=50.0,  # use 60.0 where appropriate
    quality_factor=30.0,
)
features, cycles = analyzer.analyze_ecg(filtered)

print(features[[
    "P_global_center_idx",
    "R_global_center_idx",
    "T_global_center_idx",
    "r_squared",
]].head())
```

Simulate a short demo signal (requires `pip install "pyhearts[sim]"`):

```python
import neurokit2 as nk
from pyhearts import PyHEARTS

sampling_rate = 500.0
ecg = nk.ecg_simulate(duration=12, sampling_rate=int(sampling_rate), heart_rate=70)
analyzer = PyHEARTS(sampling_rate=sampling_rate, species="human")
features, cycles = analyzer.analyze_ecg(ecg)
```

Two-column monitor CSV:

```python
from pyhearts import PyHEARTS, load_monitor_csv

loaded = load_monitor_csv(
    "recording.csv",
    sampling_rate_hz=500.0,
    adc_midpoint=8192.0,
)
analyzer = PyHEARTS(loaded.sampling_rate_hz, species="human")
features, cycles = analyzer.analyze_ecg(loaded.ecg)
```

Runnable notebooks:

- `examples/demo.ipynb` — shortest end-to-end demo
- `examples/intro_overview.ipynb` — feature/plot walkthrough
- `examples/profile_pipeline_speed.ipynb` — runtime profiling

## What you get

Each call to `analyze_ecg` returns:

- **features** — one row per cardiac cycle (fiducials, Gaussian morphology,
  intervals, fit quality; typically ~136 columns)
- **cycles** — segmented per-beat waveform samples

Important T-wave columns:

- `T_global_center_idx` — record-level T fiducial
- `T_gaussian_global_center_idx` — Gaussian-fit T center used for morphology
- `t_source` — provenance (`record_t`, `record_t_miss`, or `missing`)

Gaussian reconstruction metrics (`r_squared`, `rmse`, `T_gauss_*`) remain tied
to the morphology fit.

## Configuration

Normal use is through `species=`:

| Setting | Behavior |
|---------|----------|
| `species="human"` | Human morphology preset + record-level T |
| `species=None` | Species-agnostic morphology defaults + record-level T |
| `species="mouse"` | Mouse morphology preset; record-T off by default |

See `docs/PRESETS.md` for details. Advanced preset knobs live in
`pyhearts.config.ProcessCycleConfig`; typical runs should not need them.

## Saving output

```python
analyzer.save_output("subject_001", "results")
```

Writes:

- `subject_001_pyhearts.csv`
- `subject_001_meta.json`

The metadata JSON records package version, pipeline tag, and the resolved
analysis settings for reproducibility.

## Validation snapshots

Research checks on public datasets (not clinical performance claims):

**SPH normal-repeat subset** (802 recordings):

- Median R²: 0.9688
- Cycles with R² > 0.9: 83.83%
- R / P / T availability: 99.59% / 93.14% / 98.06%

**QTDB manual annotations** (10 records, 300 beats, ECG1, ±40 ms):

- R / P / T sensitivity: 99.67% / 80.94% / 69.00%
- T coverage: 99.67%

## Development

```bash
python -m pytest
python -m build
python -m twine check dist/*
```

CI runs tests on Python 3.10, 3.11, and 3.12.

## Scope and license

PyHEARTS is research software and is not a medical device. It must not be used
for diagnosis or treatment decisions.

This release is proprietary software of The Regents of the University of
California. See `LICENSE.md` for terms; use requires prior written permission.
