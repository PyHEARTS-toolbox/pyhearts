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

## What happens under the hood

1. Morphology detection finds R/P, segments cycles, and fits symmetric Gaussians.
2. When record-T is enabled, `analyze_ecg` finishes by writing `T_global_center_idx`.
3. `save_output` records package version, pipeline tag, and both private configs.

There is no skewed-Gaussian runtime option.

## QTDB annotation policy

The manual QTDB benchmark used ECG1 (channel 0) with the `q1c`
cardiologist annotations.

```python
from pyhearts import load_wfdb_signal

ecg, fs, lead_idx, lead_name = load_wfdb_signal(
    "data/qtdb/sel46",
    policy="first",
)
```
