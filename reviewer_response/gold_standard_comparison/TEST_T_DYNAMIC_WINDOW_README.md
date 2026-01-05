# T-Dynamic-Window Improvement Test

This test compares the T-dynamic-window improvement against the baseline (fixed 450ms window) to determine if it actually improves T-peak detection accuracy.

## What It Tests

1. **Baseline Configuration**: PyHEARTS with `t_use_dynamic_window=False` (fixed 450ms window)
2. **Experimental Configuration**: PyHEARTS with `t_use_dynamic_window=True` (RR-adaptive window)

Both configurations are tested on the same QT Database subjects with manual annotations to compare:
- Mean Absolute Deviation (MAD) of T-peak detection
- Percentage of T-peaks within ±20ms of manual annotations

## Running the Test

```bash
cd reviewer_response/gold_standard_comparison
python3 test_t_dynamic_window.py
```

## Output

The script generates:
1. **`t_dynamic_window_test_results.csv`**: Detailed results for each subject and configuration
2. **`t_dynamic_window_comparison.csv`**: Per-subject comparison showing changes
3. **Console output**: Summary statistics and comparison

## Test Subjects

By default, the script tests a subset of subjects for faster testing:
- sel853, sel233, sel45, sel46, sel811, sel883, sele0210, sele0606, sele0612

You can modify the `TEST_SUBJECTS` list in the script to test more subjects.

## Expected Results

If the T-dynamic-window improvement helps:
- **Mean MAD should decrease** (lower is better)
- **% within ±20ms should increase** (higher is better)

The test will show whether the improvement provides a consistent benefit across subjects or if it only helps in specific cases (e.g., subjects with variable RR intervals).

