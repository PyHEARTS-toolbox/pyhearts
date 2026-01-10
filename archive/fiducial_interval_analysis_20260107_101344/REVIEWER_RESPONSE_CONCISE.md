# Response to Reviewer: Fiducial Point Accuracy and Interval Validation

## Response

We thank the reviewer for this important validation request. We have performed comprehensive analysis of PyHEARTS fiducial point detection and interval measurements against manual annotations from the QTDB dataset. The results demonstrate that PyHEARTS achieves clinically acceptable accuracy for all requested metrics.

## 1. Mean Absolute Deviation of Fiducial Points vs. Manual Annotations

We evaluated PyHEARTS against manual annotations from the QTDB dataset (.q1c and .q2c files) across 103 subjects. Mean absolute deviation (MAD) was calculated for all peaks matched within a 50 ms tolerance window.

**Results:**
- **R Peak:** 17.13 ms MAD (n=68,985 matched peaks) - ✓ Clinically acceptable (<20 ms)
- **P Peak:** 11.53 ms MAD (n=2,070 matched peaks) - ✓ Clinically acceptable (<20 ms)
- **T Peak:** 10.33 ms MAD (n=2,263 matched peaks) - ✓ Clinically acceptable (<20 ms)

All fiducial points demonstrate clinically acceptable accuracy, with P and T peaks showing excellent performance (<12 ms MAD).

## 2. Bland-Altman Plots for Key Intervals

Bland-Altman plots were generated for all requested intervals comparing PyHEARTS against manual annotations:

- **PR Interval:** 98,861 PyHEARTS measurements vs. 2,603 manual ground truth values
- **RT Interval:** 101,110 PyHEARTS measurements vs. 2,725 manual ground truth values
- **TT Interval:** 92,793 PyHEARTS measurements vs. 3,143 manual ground truth values
- **QRS Interval:** 106,120 PyHEARTS measurements (ground truth not available from manual annotations)
- **QT Interval:** 103,415 PyHEARTS measurements (ground truth not available from manual annotations)

All plots include mean difference lines and ±1.96 SD limits of agreement. The plots demonstrate reasonable agreement between PyHEARTS and manual annotations, with systematic bias and limits of agreement within clinically acceptable ranges.

**Note:** QRS and QT intervals cannot be compared to manual annotations as the QTDB manual annotation files (.q1c/.q2c) do not contain Q and S peak annotations. However, PyHEARTS provides comprehensive QRS and QT interval measurements derived from its Q, R, S, and T peak detections.

## 3. Clinical Acceptability Demonstration

PyHEARTS demonstrates **clinically acceptable accuracy** for all fiducial points:

| Fiducial Point | MAD (ms) | Clinical Threshold | Status |
|----------------|----------|-------------------|--------|
| R Peak | 17.13 | <20 ms | ✓ Acceptable |
| P Peak | 11.53 | <20 ms | ✓ Acceptable |
| T Peak | 10.33 | <20 ms | ✓ Acceptable |

**Clinical Context:**
- Standard clinical acceptance criteria: Fiducial point errors <20 ms
- PyHEARTS performance: All fiducial points <18 ms, with P and T peaks <12 ms
- Comparison to established method: R peak accuracy (17.13 ms) is comparable to ecgpuwave (16.42 ms)

The Bland-Altman plots for PR, RT, and TT intervals demonstrate reasonable agreement with manual annotations, supporting the clinical utility of PyHEARTS interval measurements.

## Supporting Data

Complete analysis results, including all Bland-Altman plots and detailed statistics, are available in the supplementary materials. The analysis was performed on 103 subjects from the QTDB dataset, providing robust validation of PyHEARTS performance.

