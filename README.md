# PyHEARTS

**Python Heart Evaluation and Analysis for Rhythm and Temporal Shape**

PyHEARTS measures every wave in every beat of a single-lead ECG. It fits one
Gaussian to each of the P, Q, R, S and T waves of every cardiac cycle and
returns one row per beat: wave amplitude, timing and shape, clinical intervals
(PR, QRS, QT and others) and a fit-quality score.

## Install

Requires Python 3.10 or newer.

```bash
python -m pip install .
python -m pip install ".[wfdb]"   # optional: read PhysioNet/WFDB records
```

## Quick start

`run_standard` applies the same processing to every recording: a 0.5 to 50 Hz
zero-phase band-pass filter, a mains notch filter and handling of missing
samples.

```python
from pyhearts.standard import run_standard

ph, summary = run_standard(
    ecg,                       # 1-D lead II signal (mV)
    fs=500,                    # sampling rate of this recording (Hz)
    expected_fs=500,           # rate documented for the dataset
    mains_hz=60,               # 50 or 60
    record_id="subject_001",
    results_dir="results",     # optional: saves features and metadata
)

features = ph.output_df        # one row per beat
```

For step-by-step control, call the analyzer directly:

```python
from pyhearts import PyHEARTS

ph = PyHEARTS(sampling_rate=500, species="human")
filtered = ph.preprocess_signal(ecg, highpass_cutoff=0.5, lowpass_cutoff=50,
                                filter_order=4, notch_frequency=60,
                                quality_factor=30)
features, cycles = ph.analyze_ecg(filtered)
```

## Output

- **Per-beat features:** 144 columns covering fiducial points, Gaussian
  parameters, shape features (width, rise and decay time, symmetry,
  sharpness) and intervals. All columns are described in
  `docs/OUTPUT_VARIABLES.md`.
- **Fit quality:** `r_squared` for each beat. Use it to exclude poorly
  fitted beats.
- **Reconstruction:** `ph.reconstruct_ecg()` rebuilds the signal from the
  fitted waves.
- **Saved files:** `ph.save_output("subject_001", "results")` writes a CSV of
  features and a JSON file with the package version and settings.

Missing waves are returned as missing values rather than forced fits.

## Examples

- `examples/demo.ipynb`: shortest end-to-end run
- `examples/intro_overview.ipynb`: features and plots
- `examples/reconstruct_ecg.ipynb`: reconstruction and fit quality

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## License

PyHEARTS is research software and is not a medical device. It is copyright The
Regents of the University of California, patent pending (SD2024-342-2), and
licensed under the [PolyForm Strict License 1.0.0](LICENSE.md). You may use it
for noncommercial purposes such as academic research. You may not modify or
redistribute it. For commercial licensing, contact the UC San Diego Office of
Innovation and Commercialization at ucsdpatent@ucsd.edu.
