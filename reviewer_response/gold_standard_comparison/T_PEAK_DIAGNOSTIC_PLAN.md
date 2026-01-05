# T-Peak Detection Error Diagnostic Plan

## Current Status

We have ruled out two major potential causes:
1. ✅ **Window cutoff**: 0% of T peaks are cut off by the fixed 450ms window
2. ✅ **Detrending**: Minimal effect (only ~2ms difference between original and detrended signals)

## Performance Gap

- **ECGpuwave**: 10.15 ms mean MAD, 93.1% within ±20ms
- **PyHEARTS**: 38.55 ms mean MAD, 61.1% within ±20ms
- **Gap**: ~28 ms mean error difference

## Remaining Hypotheses to Test

Based on ALGORITHM_DIAGNOSIS.md and code analysis, the following factors remain as potential causes:

**Note**: Since cycles are properly segmented (each cycle contains a full heartbeat with T-wave), cycle-by-cycle processing architecture is not inherently problematic. The issue is likely in HOW cycles are processed, not that they're processed cycle-by-cycle.

### 1. Double Detrending and Signal Processing Artifacts ⭐ **HIGH PRIORITY**

**Hypothesis**: PyHEARTS applies detrending twice - once during epoching (`epoch.py` line 142: `window_detrended = detrend(window, type="linear")`) and again in `processcycle.py` (line 122: `detrend_signal(...)`). Additionally, filtering operations (40Hz low-pass for derivative) are applied to individual cycle segments, which may create edge artifacts compared to filtering the full continuous signal.

**Test Strategy**:
- Compare T-detection with single vs double detrending
- Compare filtering on full signal vs cycle segments (edge artifacts)
- Test if derivative computation on cycle segments creates artifacts
- Analyze if edge effects from filtering short segments affect detection

**Implementation**:
- Add experimental flag to skip double detrending
- Compute derivative on full signal, then extract cycle segments
- Compare accuracy with different processing orders

**Expected Outcome**: If double detrending or cycle-segment filtering creates artifacts, using full-signal preprocessing should improve accuracy.

---

### 2. Algorithm Parameters and Thresholds ⭐ **HIGH PRIORITY**

**Hypothesis**: While PyHEARTS uses the same derivative-based T-detection algorithm as ECGpuwave, the parameters, thresholds, or validation criteria may differ, affecting accuracy.

**Test Strategy**:
- Compare all parameters used in T-detection (40Hz cutoff, thresholds, validation criteria)
- Test sensitivity to parameter variations
- Analyze if validation/gating is too strict or too lenient
- Check for implementation differences in the algorithm application

**Implementation**:
- Extract and compare all parameters from code
- Create parameter sensitivity analysis
- Test parameter variations systematically

**Expected Outcome**: Identify parameter differences or optimization opportunities that affect accuracy.

---

### 3. Signal Preprocessing Pipeline Differences ⭐ **MEDIUM PRIORITY**

**Hypothesis**: While detrending itself doesn't matter, other preprocessing steps (filtering, baseline correction, normalization) may differ between PyHEARTS and ECGpuwave.

**Test Strategy**:
- Document all preprocessing steps in PyHEARTS T-detection pipeline
- Compare to ECGpuwave preprocessing (from documentation/source)
- Test T-detection with minimal preprocessing
- Analyze preprocessing sequence and parameters

**Implementation**:
- Create preprocessing comparison table
- Test with minimal preprocessing (just derivative-based detection)
- Test with different preprocessing combinations

**Expected Outcome**: Identify specific preprocessing steps that affect accuracy.

---


---

### 4. Error Pattern Analysis ⭐ **MEDIUM PRIORITY**

**Hypothesis**: Errors may be systematic (bias) rather than random, indicating a specific algorithmic issue.

**Test Strategy**:
- Analyze error patterns: are errors biased (early vs late)?
- Check if errors correlate with signal characteristics (amplitude, morphology, RR interval)
- Analyze error distribution: uniform or clustered?
- Check if errors are in specific signal segments

**Implementation**:
- Create error analysis script that categorizes errors:
  - Early vs late (systematic bias)
  - Correlated with signal amplitude, RR interval, cycle position
  - Error distribution patterns
- Visualize error patterns

**Expected Outcome**: Identify systematic biases or error patterns that point to specific issues.

---

### 5. Integration Effects ⭐ **LOW PRIORITY**

**Hypothesis**: How T-detection integrates with R-peak detection, cycle segmentation, and other pipeline components may affect accuracy.

**Test Strategy**:
- Compare T-detection using manual R-peaks vs detected R-peaks
- Analyze if cycle segmentation affects T-detection
- Check if template filtering or other cycle-level processing affects T-detection

**Implementation**:
- Test T-detection with manual annotations as input
- Compare cycle-by-cycle vs using exact manual cycles

**Expected Outcome**: Determine if integration effects contribute to errors.

---

## Recommended Testing Order

1. **Error Pattern Analysis** (quick, informative)
   - Fast to implement
   - Provides insights into error characteristics
   - Helps prioritize other tests

2. **Double Detrending and Signal Processing Artifacts** (high impact potential)
   - Test if double detrending or cycle-segment filtering creates artifacts
   - Compare preprocessing on full signal vs cycle segments
   - Could reveal edge artifacts from filtering short segments

3. **Algorithm Parameters and Thresholds** (high impact potential)
   - Compare parameters between implementations
   - Test sensitivity to parameter variations
   - Could reveal optimization opportunities

4. **Signal Preprocessing Pipeline Differences** (systematic comparison)
   - Requires detailed documentation comparison
   - Could reveal subtle but important differences

5. **Integration Effects** (if other tests don't reveal issues)
   - Lower priority, test if other factors don't explain the gap

## Implementation Plan

### Phase 1: Quick Diagnostics (1-2 days)
1. Error pattern analysis script
2. Basic QRS removal test (add flag, compare)

### Phase 2: Core Testing (3-5 days)
3. Full-signal processing test
4. Preprocessing pipeline comparison
5. Parameter comparison

### Phase 3: Analysis & Refinement (2-3 days)
6. Analyze all results
7. Identify root causes
8. Develop targeted improvements

## Success Criteria

A diagnostic is successful if it:
- Identifies factors that explain at least 50% of the performance gap (~14 ms reduction)
- Provides actionable insights for improvement
- Shows statistically significant differences (paired tests, p < 0.05)
- Is reproducible across multiple subjects

## Notes

- Focus on **actionable diagnostics**: tests that can lead to concrete improvements
- Prioritize **high-impact, low-effort** tests first
- Use **evidence-based approach**: test hypotheses, don't assume
- Document findings systematically for each test

