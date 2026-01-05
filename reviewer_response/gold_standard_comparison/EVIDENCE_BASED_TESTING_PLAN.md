# Evidence-Based Testing Plan for T and P Peak Detection Improvements

## Overview

Before refactoring the entire PyHEARTS toolbox, we should test which improvements actually improve performance. This document outlines a systematic testing approach.

## Suggested Improvements from ALGORITHM_DIAGNOSIS.md

### T-Peak Detection Improvements

1. **Dynamic Search Window** (High Priority)
   - Current: Fixed 450ms window starting 100ms after R peak
   - Proposed: RR-interval adaptive window (extend to next R peak or use estimated QT interval)
   - Test: Compare fixed vs dynamic window performance

2. **Full-Signal Processing** (Medium Priority)
   - Current: Cycle-by-cycle processing with detrending
   - Proposed: Process T-waves on full continuous signal, then map to cycles
   - Test: Compare cycle-based vs full-signal processing

3. **Minimize Detrending Effects** (Medium Priority)
   - Current: Uses detrended signal for T-detection
   - Proposed: Use original signal or less aggressive detrending
   - Test: Compare detrended vs original signal

### P-Peak Detection Improvements

1. **Training-Phase Adaptive Thresholds** (High Priority)
   - Current: Fixed thresholds
   - Proposed: Analyze first 1-3 seconds to learn signal characteristics
   - Test: Compare fixed vs adaptive thresholds

2. **Optimized Search Window Boundaries** (Medium Priority)
   - Current: 60ms safety margin before Q/R
   - Proposed: Reduce to 40ms or adaptive based on Q detection quality
   - Test: Compare different safety margins

## Testing Strategy

### Phase 1: Implement Improvements as Experimental Options

Instead of refactoring immediately, add experimental flags/options to test improvements:

```python
# In ProcessCycleConfig
class ProcessCycleConfig:
    # Experimental T-peak options
    t_use_dynamic_window: bool = False  # Use RR-adaptive window
    t_use_full_signal: bool = False     # Process on full signal
    t_use_original_signal: bool = False # Use original signal instead of detrended
    
    # Experimental P-peak options
    p_use_training_phase: bool = False  # Enable training-phase adaptive thresholds
    p_safety_margin_ms: float = 60.0    # Safety margin before Q/R (adjustable)
```

### Phase 2: Create Test Framework

1. **Test Script**: `test_t_p_improvements.py` (created)
   - Runs PyHEARTS with different configurations
   - Compares results to manual annotations
   - Computes metrics (MAD, % within ±20ms)

2. **Test Dataset**: Use QT Database subjects with manual annotations
   - Same subjects used in performance comparison
   - Ensure sufficient coverage (variety of morphologies, heart rates)

3. **Metrics to Track**:
   - P-peak: Mean MAD, % within ±20ms
   - T-peak: Mean MAD, % within ±20ms
   - Compare to baseline (current PyHEARTS)
   - Statistical significance testing

### Phase 3: Systematic Testing

Test each improvement **individually** first, then test combinations:

1. **Baseline**: Current PyHEARTS (no changes)
2. **T-Dynamic-Window**: Only dynamic window improvement
3. **T-Full-Signal**: Only full-signal processing
4. **P-Training-Phase**: Only training-phase thresholds
5. **P-Optimized-Bounds**: Only reduced safety margin
6. **Combined-T**: Both T improvements together
7. **Combined-P**: Both P improvements together
8. **All-Combined**: All improvements together

### Phase 4: Analysis

1. **Compare metrics** for each test vs baseline
2. **Identify improvements** that provide significant gains
3. **Check for regressions** (does improvement help one peak but hurt another?)
4. **Statistical testing** (e.g., paired t-test, Wilcoxon signed-rank test)

### Phase 5: Decision

- Only implement improvements that show **significant improvement**
- Avoid changes that provide minimal/no improvement
- Consider trade-offs (complexity vs performance)

## Implementation Steps

### Step 1: Add Experimental Flags

Modify `pyhearts/config.py` to add experimental options (as shown above).

### Step 2: Implement Improvements (One at a Time)

Start with highest priority improvements:

1. **T-Dynamic Window** (easiest to implement)
   - Modify `processcycle.py` T-detection code
   - Use RR interval to compute adaptive window
   - Keep current code as fallback if flag is False

2. **P-Training Phase** (more complex)
   - Add training phase function (similar to R-peak training)
   - Compute adaptive thresholds from first 1-3 seconds
   - Use adaptive thresholds in P-detection

3. **T-Full-Signal Processing** (most complex)
   - Requires architectural changes
   - May need to delay until after cycle processing
   - More invasive change

### Step 3: Run Tests

Use `test_t_p_improvements.py` to systematically test each improvement.

### Step 4: Analyze Results

- Compute improvement metrics
- Statistical significance
- Identify which improvements are worth keeping

### Step 5: Refactor (Only If Successful)

Only after evidence shows improvements work:
- Make successful improvements default (or remove experimental flags)
- Remove unsuccessful experimental code
- Update documentation

## Expected Outcomes

Based on ALGORITHM_DIAGNOSIS.md findings:

- **T-Dynamic Window**: Likely to help (addresses fixed window issues)
- **P-Training Phase**: Likely to help significantly (addresses fixed threshold issue)
- **T-Full-Signal**: May help (reduces edge artifacts) but more complex
- **P-Optimized-Bounds**: May help slightly (marginal improvement expected)

## Recommended Testing Order

1. **Start with T-Dynamic Window** (easiest, high priority)
   - Quick to implement
   - Addresses clear issue (fixed window)
   - Low risk

2. **Then P-Training Phase** (high priority, medium complexity)
   - Addresses fundamental difference vs ECGpuwave
   - Expected significant improvement
   - More complex but manageable

3. **Then test combinations** of successful improvements

4. **Consider T-Full-Signal only if** previous improvements don't fully address issues

5. **Skip P-Optimized-Bounds** if training phase solves the problem

## Next Steps

1. ✅ Create testing framework plan
2. ⬜ Add experimental flags to config
3. ⬜ Implement T-dynamic-window improvement (experimental)
4. ⬜ Test T-dynamic-window improvement
5. ⬜ Implement P-training-phase improvement (experimental)
6. ⬜ Test P-training-phase improvement
7. ⬜ Analyze results and decide which to keep
8. ⬜ Refactor based on evidence (only successful improvements)

## Notes

- Keep experimental code separate/flagged to avoid breaking existing functionality
- Test on same dataset used for performance comparison (for consistency)
- Consider computational cost trade-offs (some improvements may be slower)
- Document which improvements help and which don't for future reference
- Use existing comparison infrastructure (`gold_standard_manual_comparison.py`) as reference
