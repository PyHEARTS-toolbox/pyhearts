# Selected Subjects Analysis

This folder contains QTDB raw data files and PyHEARTS results for 4 subjects that exhibit systematic timing offsets.

## Subjects

- **sele0409**: Mean offset ~47ms (99.9% matched)
- **sele0211**: Mean offset ~60ms (99.9% matched)
- **sele0121**: Mean offset ~44ms (99.2% matched)
- **sele0122**: Mean offset ~42ms (43.9% matched, many missing detections)

## Files Structure

```
selected_subjects_analysis/
├── qtdb_raw_files/          # QTDB annotation and signal files
│   ├── sele0409.*
│   ├── sele0211.*
│   ├── sele0121.*
│   └── sele0122.*
├── pyhearts_results/        # PyHEARTS detection results
│   ├── sele0409_pyhearts.csv
│   ├── sele0211_pyhearts.csv
│   ├── sele0121_pyhearts.csv
│   └── sele0122_pyhearts.csv
└── README.md
```

## QTDB File Types

For each subject, the QTDB raw files include:
- `.hea` - Header file
- `.dat` - Signal data
- `.atr` - QRS annotations (R-peaks)
- `.qt1` - QT interval annotations
- `.man` - Manual annotations
- `.pu`, `.pu0`, `.pu1` - P-wave annotations
- `.q1c` - QRS complex annotations
- `.xws` - Wave annotations

## Analysis Summary

These subjects show **systematic timing offsets** between PyHEARTS detections and QTDB annotations:

- Most subjects have offsets of 40-60ms, just outside the 20ms tolerance window
- The offsets are very consistent (low standard deviation ~2-3ms)
- This suggests systematic timing differences in annotation schemes rather than detection errors

## Generated

Created: December 19, 2024
Source: qtdb_20251219_102107 PyHEARTS run

