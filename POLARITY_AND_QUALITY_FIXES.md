# Polarity Detection and Signal Quality Filtering Fixes

## Issues Identified

### 1. Incorrect Polarity Detection for sel35
- **Problem**: sel35 was incorrectly detected as inverted
- **Evidence**: 
  - QTDB annotations show all 31 R-peaks are positive (mean: 1.198 mV)
  - But PyHEARTS detected signal as inverted and found negative peaks
  - This caused 0 R-peaks to be detected in full analysis

### 2. No Signal Quality Filtering
- **Problem**: No mechanism to filter out low-quality signals
- **Impact**: Poor results on signals that shouldn't be analyzed

## Fixes Implemented

### 1. Improved Polarity Detection (`pyhearts/processing/rpeak.py`)

**Changes**:
- **Stricter thresholds**: Increased from 1.1x/1.15x to 1.2x/1.3x for inversion detection
- **Peak value checking**: Now checks actual peak amplitudes, not just prominence
- **Conservative fallback**: When uncertain, assumes normal (not inverted) polarity
- **Primary amplitude check**: If positive peaks are substantial (>0.5 mV) and comparable to negative peaks, assumes normal

**Key improvements**:
```python
# Primary check: compare actual peak amplitudes
if max_pos_val > 0.5 and max_pos_val > abs(min_neg_val) * 0.7:
    is_inverted = False  # Conservative: assume normal if positive peaks present
# Only invert if negative peaks are clearly and substantially dominant
elif abs(min_neg_val) > 0.5 and abs(min_neg_val) > max_pos_val * 1.3:
    is_inverted = True
```

**Results**:
- ✅ sel35 now correctly detected as NOT inverted
- ✅ Detects 843 R-peaks (all positive) instead of 0
- ✅ Maintains correct detection for truly inverted signals

### 2. Signal Quality Assessment (`pyhearts/processing/quality.py`)

**New module**: `assess_signal_quality()` function

**Metrics calculated**:
- **SNR (dB)**: Signal-to-noise ratio
- **Amplitude range (mV)**: Peak-to-peak amplitude
- **Baseline wander (mV)**: Standard deviation of baseline

**Quality thresholds** (configurable):
- Minimum SNR: 15 dB
- Minimum amplitude: 0.3 mV
- Maximum baseline wander: 0.3 mV

**Integration**:
- Added to `PyHEARTS.analyze_ecg()` method
- Logs warning if quality check fails but continues processing
- Allows user to decide whether to proceed with low-quality signals

### 3. Updated Processing Pipeline (`pyhearts/core/fit.py`)

**Changes**:
- Signal quality check runs before R-peak detection
- Quality metrics logged if verbose mode enabled
- Processing continues even if quality check fails (user decision)

## Testing Results

### Simulated Signals
✅ **All tests pass**:
- Normal signal: 100% detection, correct polarity
- Inverted signal: 100% detection, correct polarity
- **No performance degradation**

### Real Subjects
- **sel35**: 
  - Before: 0 R-peaks (incorrectly detected as inverted)
  - After: 843 R-peaks (correctly detected as normal, positive peaks)
  - Polarity: ✅ Correct

### Signal Quality Assessment
All 5 poor-performing subjects pass quality checks:
- sel35: SNR 39.0 dB, Range 2.840 mV ✅
- sel102: SNR 17.0 dB, Range 2.055 mV ✅
- sel52: SNR 29.9 dB, Range 2.030 mV ✅
- sel16786: SNR 24.2 dB, Range 3.645 mV ✅
- sel820: SNR 21.0 dB, Range 2.070 mV ✅

## Files Modified

1. `pyhearts/processing/rpeak.py` - Fixed polarity detection logic
2. `pyhearts/processing/quality.py` - New signal quality assessment module
3. `pyhearts/core/fit.py` - Integrated quality checking
4. `pyhearts/processing/__init__.py` - Exported quality function

## Configuration

Signal quality thresholds can be adjusted in `PyHEARTS.analyze_ecg()`:
```python
assess_signal_quality(
    ecg_signal,
    sampling_rate,
    min_snr_db=15.0,           # Adjustable
    min_amplitude_range_mv=0.3, # Adjustable
    max_baseline_wander_mv=0.3,  # Adjustable
)
```

## Recommendations

1. **Monitor quality warnings**: Check logs for quality warnings in production
2. **Adjust thresholds**: Fine-tune thresholds based on your signal characteristics
3. **Consider rejection**: Optionally reject signals that fail quality checks (currently just warns)

## Conclusion

✅ **Polarity detection fixed**: sel35 now correctly detected as normal (not inverted)
✅ **Signal quality filtering added**: Warns on low-quality signals
✅ **Performance maintained**: All simulated signal tests pass
✅ **Backward compatible**: Changes don't break existing functionality

