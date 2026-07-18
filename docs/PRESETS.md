# PyHEARTS 2.0 pipeline presets

PyHEARTS 2.0 separates the 2025 morphology-core configuration from the STPQ
T-detector configuration.

## Recommended human run

```python
from pyhearts import PyHEARTS

analyzer = PyHEARTS(
    sampling_rate=500.0,
    species="human",
)
features, cycles = analyzer.analyze_ecg(filtered_ecg)
```

This selects:

- `CoreProcessCycleConfig.for_human()` for the 2025 R/P/Gaussian core.
- `ProcessCycleConfig.for_human_unified()` for record-level STPQ T.

## Validated SPH configuration

The full 802-record SPH benchmark used the species-agnostic 2025 core defaults:

```python
analyzer = PyHEARTS(sampling_rate=500.0)
```

STPQ T remains enabled. This is intentionally distinct from
`species="human"` so the published SPH result can be reproduced.

## Mouse configuration

```python
analyzer = PyHEARTS(sampling_rate=2000.0, species="mouse")
```

The mouse preset uses the 2025 mouse core and disables the human STPQ T pass by
default. It can be enabled explicitly with `apply_stpq_t=True`, but that
combination has not been validated.

## Advanced configuration

```python
from pyhearts import (
    CoreProcessCycleConfig,
    ProcessCycleConfig,
    PyHEARTS,
)

core_cfg = CoreProcessCycleConfig.for_human()
stpq_cfg = ProcessCycleConfig.for_human_unified()

analyzer = PyHEARTS(
    sampling_rate=500.0,
    core_cfg=core_cfg,
    cfg=stpq_cfg,
)
```

The 2025 core always uses symmetric three-parameter Gaussians. There is no
skewed-Gaussian runtime option.

## QTDB annotation policy

The frozen manual benchmark used ECG1 (channel 0) with the `q1c`
cardiologist annotations. Lead choice must remain fixed when comparing
pipelines because ECG2 performed poorly against this local reference subset.

```python
from pyhearts import load_wfdb_signal

ecg, fs, lead_idx, lead_name = load_wfdb_signal(
    "data/qtdb/sel46",
    lead_policy="first",
)
```
