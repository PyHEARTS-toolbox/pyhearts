# Analysis Scripts

This directory contains analysis, comparison, debugging, and validation scripts that are not part of the core PyHEARTS toolbox.

## Script Categories

### Analysis Scripts
- `analyze_fiducials_and_intervals.py` - Analyze fiducial points and intervals
- `analyze_missing_data.py` - Analyze missing data patterns

### Comparison Scripts
- `compare_all_existing_results.py` - Compare all existing results
- `compare_existing_results_sel100.py` - Compare results for sel100 subject
- `compare_pyhearts_qtdb_ground_truth.py` - Compare PyHEARTS vs QTDB ground truth
- `compare_pyhearts_vs_ecgpuwave.py` - Compare PyHEARTS vs ECGPUWAVE
- `compare_rpeak_methods.py` - Compare different R peak detection methods
- `compare_sel117_annotations.py` - Compare annotations for sel117 subject

### Debug Scripts
- `debug_r_gaussian_fit.py` - Debug R wave Gaussian fitting
- `debug_r_std_availability.py` - Debug R wave standard deviation availability
- `debug_r_wave_features.py` - Debug R wave features
- `debug_std_dict_r.py` - Debug standard deviation dictionary for R waves
- `debug_why_fix_not_working.py` - Debug why a fix is not working
- `explain_std_dict_none.py` - Explain standard deviation dictionary None values
- `investigate_std_dict_none_real.py` - Investigate standard deviation dictionary None values

### Processing Scripts
- `process_qtdb_full.py` - Process full QTDB dataset

### Test/Validation Scripts
- `test_ecgpuwave_style_detection.py` - Test ECGPUWAVE-style detection
- `test_half_height_limitations.py` - Test half-height limitations
- `test_hrv_metrics.py` - Test HRV metrics
- `test_p_validation_impact.py` - Test P wave validation impact
- `test_p_validation_quick.py` - Quick P wave validation test
- `test_p_validation_with_gt.py` - Test P wave validation with ground truth
- `test_performance_verification.py` - Performance verification test
- `test_qtc_features.py` - Test QTc features
- `test_qtdb_detection_rates.py` - Test QTDB detection rates
- `test_sel117_detection.py` - Test detection on sel117 subject
- `test_single_r_fit.py` - Test single R wave fitting
- `test_slope_features.py` - Test slope features
- `test_st_segment_features.py` - Test ST segment features
- `test_validation_removal_impact.py` - Test validation removal impact
- `test_variability_metrics.py` - Test variability metrics

## Note

These scripts are provided for analysis, debugging, and validation purposes. They are not part of the core PyHEARTS package and may require specific data files or configurations to run.

