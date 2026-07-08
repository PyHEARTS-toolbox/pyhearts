# PyHEARTS configuration presets

## Production (use this)

| Species | How to run | Config |
|---------|------------|--------|
| Human | `PyHEARTS(sampling_rate=fs, species="human")` | `ProcessCycleConfig.for_human_unified()` (v3.2.1) |
| Mouse | `PyHEARTS(sampling_rate=fs, species="mouse")` | `ProcessCycleConfig.for_mouse()` |

`for_human_unified()` (v3.2.1): derivative R + per-cycle P + record STPQ T on median-baseline
trace (no Savitzky–Golay). Threshold T apex, w1 floor (40% S→Q), morphology window,
signed-polarity apex filter, template-guided reconcile. Record T overwrites finite
per-cycle T; P is retained from per-cycle detection.

```python
from pyhearts import PyHEARTS
from pyhearts.config import ProcessCycleConfig

analyzer = PyHEARTS(
    sampling_rate=250.0,
    cfg=ProcessCycleConfig.for_human_unified(),
)
features_df, cycles_df = analyzer.analyze_ecg(ecg_mv)
```

## Other human presets on `ProcessCycleConfig`

| Method | Role |
|--------|------|
| `for_human()` | v2.3 baseline (per-cycle P/T only; not the `species="human"` default) |
| `for_human_unified_v321()` | Explicit alias for production v3.2.1 |
| `for_human_unified_template_prior_phase1()` | Template-prior windows experiment (skips record STPQ overwrite) |
| `for_human_unified_v33a()` | Archived fill-missing-T sprint variant |

R detection is **derivative + Phase A** only (`r_detection_method="derivative"`).

## QTDB lead policy

PyHEARTS, benchmarks, and Bland–Altman comparisons should use the **same WFDB channel**
as the expert annotation file (`q1c` vs `q2c`).

| Policy | Channel | Manual ann |
|--------|---------|------------|
| `ecg2_else_ecg1` (default) | ECG2 if named, else ECG1, else limb (MLII/II), else 0 | `q2c` if index 1, else `q1c` |
| `first` | Always channel 0 | `q1c` |
| `second` | Channel 1 if present | `q2c` |
| `limb_preferred` | MLII / II / ECG1-style names | per index |

```python
from pyhearts import load_wfdb_signal, pick_manual_annotation_ext

ecg, fs, lead_idx, lead_name = load_wfdb_signal("data/qtdb/1.0.0/sel100", "ecg2_else_ecg1")
ann_ext = pick_manual_annotation_ext(["MLII", "V5"])  # -> "q1c"
```

Output CSV columns include `p_source`, `t_source`, `p_confidence`, `t_confidence`,
`wfdb_lead_index`, `wfdb_lead_name`, `lead_policy`, `manual_ann_ext`.
