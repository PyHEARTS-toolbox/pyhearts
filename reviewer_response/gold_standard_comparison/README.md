# Gold Standard Comparison Analysis

This folder contains scripts and results for comparing PyHEARTS against gold-standard annotations from the QT Database.

## Contents

### Scripts

1. **`gold_standard_comparison_analysis.py`**
   - Compares PyHEARTS vs ECGPUWave annotations
   - Generates:
     - Mean absolute deviation (MAD) of fiducial points
     - Bland-Altman plots for key intervals (PR, QRS, QT, RT, TT)
     - Clinical error bounds assessment

2. **`gold_standard_manual_comparison.py`**
   - Compares PyHEARTS vs manual annotations from QT Database
   - Generates:
     - Mean absolute deviation (MAD) of fiducial points
     - Bland-Altman plots for key intervals (PR, QT)
     - Clinical error bounds assessment

### Output Directories

1. **`pyhearts_vs_ecgpuwave/`**
   - Results from PyHEARTS vs ECGPUWave comparison
   - Contains:
     - `fiducial_mad_summary.csv` - Mean absolute deviation statistics
     - `bland_altman_*.png` - Bland-Altman plots for intervals
     - `bland_altman_statistics.csv` - Summary statistics
     - `clinical_error_bounds_summary.csv` - Clinical bounds assessment

2. **`pyhearts_vs_manual/`**
   - Results from PyHEARTS vs QT Database manual annotations comparison
   - Contains:
     - `fiducial_mad_summary.csv` - Mean absolute deviation statistics
     - `bland_altman_*.png` - Bland-Altman plots for intervals
     - `bland_altman_statistics.csv` - Summary statistics
     - `clinical_error_bounds_summary.csv` - Clinical bounds assessment

### Documentation

- **`GOLD_STANDARD_COMPARISON_REPORT.md`** - Detailed report for ECGPUWave comparison
- **`GOLD_STANDARD_MANUAL_COMPARISON_REPORT.md`** - Detailed report for manual comparison

## Usage

### Running ECGPUWave Comparison

```bash
python3 gold_standard_comparison_analysis.py
```

### Running Manual Annotation Comparison

```bash
python3 gold_standard_manual_comparison.py
```

## Paths

The scripts use the following paths (update as needed):

- **PyHEARTS Results**: `/Users/morganfitzgerald/Documents/qtdb/qtdb_20251231_140913`
- **ECGPUWave Results**: `/Users/morganfitzgerald/Documents/qtdb/qtdb_full_ecgpuwave_results`
- **Manual Annotations**: `/Users/morganfitzgerald/Documents/pyhearts/data/qtdb/1.0.0`

## Output

All results are saved in the respective output directories (`pyhearts_vs_ecgpuwave/` and `pyhearts_vs_manual/`).
