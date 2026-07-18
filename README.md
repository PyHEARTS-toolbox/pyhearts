# PyHEARTS

Beat-by-beat ECG morphology analysis with a validated 2025 Gaussian core and
record-level STPQ T-wave detection.

## Pipeline in version 2.0

PyHEARTS 2.0 is the frozen **T-only hybrid**:

1. The validated 2025 pipeline preprocesses the signal, detects R/P peaks,
   segments cycles, and performs symmetric Gaussian morphology fitting.
2. The production human-unified record-level STPQ detector refines the global
   T-wave fiducial across the complete recording.

The public `PyHEARTS` class runs both stages in-process. It has no dependency on
an older checkout, subprocess workers, or machine-specific paths.

## Installation

PyHEARTS requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Optional signal simulation and WFDB support:

```bash
python -m pip install ".[all]"
```

For development:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Quick start

```python
import numpy as np
from pyhearts import PyHEARTS

sampling_rate = 500.0
ecg = np.load("lead_ii_ecg.npy")

analyzer = PyHEARTS(
    sampling_rate=sampling_rate,
    species="human",
)

filtered = analyzer.preprocess_signal(
    ecg,
    highpass_cutoff=0.5,
    lowpass_cutoff=50.0,
    filter_order=4,
    notch_frequency=50.0,
    quality_factor=30.0,
)
features, cycles = analyzer.analyze_ecg(filtered)

print(features[[
    "P_global_center_idx",
    "R_global_center_idx",
    "T_global_center_idx",
    "r_squared",
]])
```

For a two-column monitor export:

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

## Configuration

- `species="human"` selects the 2025 human core preset and enables STPQ T.
- `species=None` uses the species-agnostic 2025 defaults used in the full SPH
  validation and enables STPQ T.
- `species="mouse"` selects the 2025 mouse core and disables the human STPQ
  pass by default.
- `core_cfg=` accepts `pyhearts.CoreProcessCycleConfig` for advanced core
  configuration.
- `cfg=` accepts the current `ProcessCycleConfig` for advanced STPQ
  configuration.

## T-wave columns

The hybrid intentionally separates two T definitions:

- `T_global_center_idx`: record-level STPQ T fiducial.
- `T_gaussian_global_center_idx`: original 2025 Gaussian T center.
- `T_gauss_*`, T onset/offset, T morphology, and fit metrics: values from the
  2025 Gaussian morphology model.
- `t_source`: provenance of the global T fiducial.

This separation keeps the validated Gaussian reconstruction unchanged while
making the more complete STPQ T timing available explicitly.

## Validation

### SPH normal-repeat subset

Full run over 802 recordings:

- Median R²: 0.9688
- Cycles with R² > 0.9: 83.83%
- R availability: 99.59%
- P availability: 93.14%
- T availability: 98.06%
- Median within-record RT MAD: 6.0 ms

The 2025 baseline had the same median R² and T availability of 81.65%;
therefore the packaged hybrid retained reconstruction fidelity while increasing
T coverage by about 16 percentage points.

### QTDB manual annotations

Ten records, 300 expert-annotated beats, ECG1:

- R sensitivity within ±40 ms: 99.67%
- P sensitivity within ±40 ms: 80.94%
- T sensitivity within ±40 ms: 69.00% (2025 baseline: 50.00%)
- T coverage: 99.67% (2025 baseline: 50.00%)

These are research validation results, not a claim of clinical performance.

## Saving reproducible output

```python
analyzer.save_output("subject_001", "results")
```

This writes:

- `subject_001_pyhearts.csv`
- `subject_001_meta.json`

The metadata records both the 2025 core configuration and STPQ configuration.

## Development checks

```bash
python -m pytest
python -m ruff check pyhearts/core/hybrid.py tests/test_hybrid_pipeline.py
python -m build
python -m twine check dist/*
```

CI runs the test suite on Python 3.10, 3.11, and 3.12.

## Scope and license

PyHEARTS is research software and is not a medical device. It must not be used
for diagnosis or treatment decisions.

Copyright © 2025 The Regents of the University of California. The repository is
proprietary; see `LICENSE.md` for the applicable restrictions.
