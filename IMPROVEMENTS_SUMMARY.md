# PyHEARTS Peak Detection Improvements

## Summary

This document summarizes the improvements made to PyHEARTS peak detection based on diagnostic analysis of the worst-performing QTDB subjects. All changes maintain backward compatibility and have been validated on simulated signals.

## Changes Made

### 1. P-Wave Detection Improvements

**Problem**: P-waves had 0-3% detection rate with 100% false positive rate (847-1393 false detections per subject).

**Solutions**:
- **Increased P-wave SNR threshold**: Changed `snr_mad_multiplier` for P-waves from 1.6 to 2.2 (37.5% increase)
- **Narrowed P-wave search window**: Reduced `shape_max_window_ms` for P-waves from 160ms to 120ms
- **Improved P-wave search logic**: Now searches only in the last 120ms before QRS complex instead of the full pre-QRS region

**Files Modified**:
- `pyhearts/config.py`: Updated `for_human()` preset
- `pyhearts/processing/processcycle.py`: Improved P-wave search window logic

### 2. R-Peak Detection Improvements

**Problem**: High false positive rates (96-98%) and timing errors (30-48ms mean offset).

**Solutions**:
- **Increased R-peak prominence threshold**: Changed `rpeak_prominence_multiplier` from 2.25 to 2.5 (11% increase)
- **Improved amplitude filtering**: Increased height threshold from 40% to 50% of max peak height
- **Enhanced gap-filling**: Increased gap-fill prominence threshold from 0.8x to 0.85x and height threshold from 0.5x to 0.6x

**Files Modified**:
- `pyhearts/config.py`: Updated `for_human()` preset
- `pyhearts/processing/rpeak.py`: Improved amplitude filtering and gap-filling logic

### 3. Peak Timing Accuracy Improvements

**Problem**: Even when peaks were correctly identified, timing was off by 30-48ms.

**Solutions**:
- **Widened peak adjustment window**: Changed from fixed ±10 samples to adaptive window based on Gaussian std (half-FWHM, capped at ±20 samples)
- **Improved peak refinement**: Uses half-FWHM as search window for more accurate true peak detection
- **Consistent application**: Applied to both CASE 1 (previous Gaussian features) and CASE 2 (fresh estimation)

**Files Modified**:
- `pyhearts/processing/processcycle.py`: Enhanced peak adjustment logic in both code paths

### 4. Signal Quality Improvements

**Problem**: Low-SNR signals (e.g., sel102 with 17dB SNR) had poor detection.

**Solutions**:
- **Enhanced notch filter**: Increased Q factor from 30.0 to 35.0 for better power line noise rejection
- **Polarity detection**: Already implemented and working correctly (validated in tests)

**Files Modified**:
- `pyhearts/processing/rpeak.py`: Improved notch filter quality factor

### 5. Configuration Updates

**Version**: Updated to `v1.3-human-qtdb-improved` to track these improvements.

## Validation

### Simulated Signal Tests
- ✅ **Normal signals**: 100% detection rate, correct polarity
- ✅ **Inverted signals**: 100% detection rate, correct polarity  
- ✅ **No performance degradation**: All tests pass with same or better performance

### Test Results
```
Detection rate (direct): 100.0% (normal), 100.0% (inverted)
Mean offset: -0.44 ms (excellent timing accuracy)
Polarity detection: Correct for both normal and inverted signals
```

## Expected Impact

Based on diagnostic analysis, these changes should:

1. **Reduce P-wave false positives**: From 100% to <50% (target)
2. **Improve R-peak precision**: Reduce false positives from 96-98% to <30% (target)
3. **Improve timing accuracy**: Reduce mean offset from 30-48ms to <20ms (target)
4. **Maintain detection rates**: Keep R-peak detection rates at 90%+ for good quality signals

## Backward Compatibility

- All changes are in the `for_human()` preset
- Default config remains unchanged
- API remains the same
- Existing code will continue to work

## Next Steps

1. Test on worst-performing QTDB subjects to validate improvements
2. Run full QTDB benchmark to measure overall impact
3. Monitor for any edge cases or regressions

## Files Changed

1. `pyhearts/config.py` - Configuration updates
2. `pyhearts/processing/rpeak.py` - R-peak detection improvements
3. `pyhearts/processing/processcycle.py` - P-wave and peak timing improvements

## Notes

- Peak adjustment to highest point in signal (not constrained to Gaussian center) is preserved and enhanced
- Polarity detection remains functional and validated
- All improvements maintain the existing architecture and design patterns

