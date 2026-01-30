# PyHEARTS  
Beat-by-beat ECG waveform morphology mapping for interpretable machine learning and AI  

---

## Overview  

PyHEARTS (Python Heart Evaluation and Analysis for Rhythm and Temporal Shape) is an open-source Python toolbox for high-resolution, physiologically grounded ECG analysis.  
It extracts 139 morphological, temporal, and interval features from each detected cardiac cycle, enabling interpretable, reproducible, and scalable modeling of cardiac electrophysiology in humans and animals.

Traditional ECG analysis pipelines summarize signals into a few metrics such as heart rate and heart rate variability, discarding valuable beat-to-beat variability.  
Deep learning models can recover this information but are often opaque and lack interpretability and cross-dataset generalizability.  

PyHEARTS bridges this gap by providing a transparent and physiologically interpretable framework for beat-level ECG feature extraction, supporting reproducible research across datasets, sessions, and species.

---

## Quickstart (run on a 2-column monitor CSV)

If your device export is two columns `(time, value)` (often `MM:SS.s` plus ADC counts), you can load it and run PyHEARTS like this:

```python
import pyhearts

# Load the monitor export. If values look like unsigned ADC with midpoint ~8192,
# keep adc_midpoint=8192.0. If you know the scaling, set mv_per_count to convert to mV.
loaded = pyhearts.load_monitor_csv(
    "/path/to/ecg.csv",
    sampling_rate_hz=500.0,
    adc_midpoint=8192.0,
    mv_per_count=None,
)

analyzer = pyhearts.PyHEARTS(sampling_rate=loaded.sampling_rate_hz, species="human", sensitivity="high")
features_df, cycles_df = analyzer.analyze_ecg(loaded.ecg)
```

## Flow chart (what PyHEARTS does)

Beginner-friendly visual summary:

- `docs/PYHEARTS_FLOWCHART.md` (Mermaid in Markdown)
- `docs/pyhearts_flowchart.html` (standalone HTML, colored + clickable step details)

---

## Tutorials

Tutorials live in a separate repository: [PyHEARTS-toolbox/tutorials](https://github.com/PyHEARTS-toolbox/tutorials).

---

## Key Features  

- **Beat-level phenotyping**: Extracts over 130 features per cardiac cycle.  
- **Physiologically constrained Gaussian modeling**: Fits P, Q, R, S, and T waves using reproducible, bounded optimization.  
- **Feature classes**:
  - Morphological (height, width, sharpness, voltage integral)
  - Temporal (rise/decay durations, symmetry)
  - Interval (PR, QRS, QT, ST, RR, PP)
  - Variability (standard deviation, coefficient of variation, interquartile range)
  - Heart rate variability (SDNN, RMSSD, NN50)
- **Cross-species compatibility**: Presets for human and mouse ECGs (`for_human()` and `for_mouse()`).
- **Reconstruction fidelity**: Over 75% of cycles exceed R² > 0.9.
- **High reproducibility**: Median feature ICC > 0.95 across sessions and datasets.
- **Transparent and configurable**: Every run saves full parameter configuration and analysis metadata.
- **Signal quality assessment**: Automatic quality checks (SNR, amplitude, baseline wander) with configurable thresholds.
- **Robust peak detection**: Improved P, R, and T wave detection with reduced false positives and enhanced timing accuracy.

---
