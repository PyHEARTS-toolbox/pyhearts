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

---

## Installation  

```bash
git clone https://github.com/ucsd-voyteklab/pyhearts.git
cd pyhearts
pip install -e .
