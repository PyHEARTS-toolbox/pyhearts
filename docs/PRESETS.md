# PyHEARTS pipeline presets

PyHEARTS exposes one public analyzer. Internally it uses a morphology-core
config and a record-level T config; those dual configs are private.

## Recommended human run

```python
from pyhearts import PyHEARTS

analyzer = PyHEARTS(
    sampling_rate=500.0,
    species="human",
)
features, cycles = analyzer.analyze_ecg(filtered_ecg)
```

This selects the human morphology preset and enables the record-level T stage at the
end of `analyze_ecg`.

## Species-agnostic defaults

Useful when reproducing the SPH normal-repeat validation setting:

```python
analyzer = PyHEARTS(sampling_rate=500.0)
```

record-level T remains enabled.

## Mouse configuration

```python
analyzer = PyHEARTS(sampling_rate=2000.0, species="mouse")
```

Uses the mouse morphology preset and disables the human record-level T stage by default.

## Heart rate variability

Optional HRV (`analyzer.compute_hrv_metrics()`) uses successive intervals from
the **detected R-peak series** (before epoch quality filtering), gated by
`rr_bounds_ms`. It does not use morphology-retained `RR_interval_ms` rows.
Requires ≥60 valid RR intervals; returns mean heart rate, SDNN, RMSSD, and NN50.

Morphology R detection enables **auto-polarity** by default
(`rpeak_auto_polarity=True`): inverted QRS / lead polarity is detected, the
working trace is negated for analysis, and `analyzer.signal_inverted` records
whether a flip occurred.

## What happens under the hood

1. Morphology detection finds R/P, segments cycles, and fits symmetric Gaussians.
2. When record-T is enabled, `analyze_ecg` finishes by writing `T_global_center_idx`.
3. `save_output` records package version, pipeline tag, and both private configs.

There is no skewed-Gaussian runtime option.

## QTDB annotation policy (development / benchmark)

QTDB is a **development and benchmark** corpus (including the Dec 2024 default
re-tune), not a held-out validation set for current public defaults. Held-out
morphology evaluation uses LUDB with a frozen config — see
``validation/README.md``.

The manual QTDB morphology reference uses ECG1 (channel 0) with the ``q1c``
cardiologist annotations. Treat these marks as a **reference** (report against
published inter-annotator variability), not absolute ground truth. Synthetic
R-peak stress tests in ``examples/sim_rpeak_*.py`` are a separate prerequisite
and do not validate morphology.

```python
from pyhearts import load_wfdb_signal

ecg, fs, lead_idx, lead_name = load_wfdb_signal(
    "data/qtdb/sel46",
    policy="first",
)
```

Runnable reference script: ``examples/qtdb_morphology_reference.py``.
