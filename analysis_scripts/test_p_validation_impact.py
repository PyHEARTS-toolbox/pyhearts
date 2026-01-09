#!/usr/bin/env python3
"""
Test the impact of distance and morphology validation on P-wave detection.
Compares four configurations by analyzing detection rates and characteristics:
1. Both validations enabled (default)
2. Distance validation disabled
3. Morphology validation disabled
4. Both validations disabled

Note: QTDB does not have P-wave ground truth annotations, so we compare
detection statistics and characteristics instead of accuracy metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from dataclasses import replace
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.absolute()))
from pyhearts import PyHEARTS
from pyhearts.config import ProcessCycleConfig

SCRIPT_DIR = Path(__file__).parent.absolute()
QTDB_DATA_DIR = SCRIPT_DIR / "data" / "qtdb" / "1.0.0"

# QTDB subjects to test (using a small subset for speed)
TEST_SUBJECTS = [
    "sel100", "sel102", "sel103"
]


def analyze_p_detections(output_df, sampling_rate):
    """Analyze P-wave detection characteristics."""
    if output_df is None or len(output_df) == 0:
        return {
            'detection_rate': 0.0,
            'mean_amplitude': np.nan,
            'mean_pr_interval_ms': np.nan,
            'mean_duration_ms': np.nan,
            'valid_pr_ratio': 0.0
        }
    
    # Detection rate (P waves detected / total cycles)
    total_cycles = len(output_df)
    # Try different possible column names (capital P is correct)
    p_col = None
    for col_name in ['P_global_center_idx', 'p_global_center_idx', 'p_center_idx']:
        if col_name in output_df.columns:
            p_col = col_name
            break
    
    if p_col is None:
        # No P column found
        return {
            'detection_rate': 0.0,
            'mean_amplitude': np.nan,
            'mean_pr_interval_ms': np.nan,
            'mean_duration_ms': np.nan,
            'valid_pr_ratio': 0.0,
            'p_detected': 0,
            'total_cycles': total_cycles
        }
    
    p_detected = output_df[p_col].notna().sum()
    detection_rate = (p_detected / total_cycles) * 100.0 if total_cycles > 0 else 0.0
    
    # Mean amplitude (relative to R) - try different column names
    p_height_col = None
    for col_name in ['P_height', 'p_height']:
        if col_name in output_df.columns:
            p_height_col = col_name
            break
    
    if p_height_col:
        p_amplitudes = output_df[p_height_col].dropna()
        mean_amplitude = p_amplitudes.mean() if len(p_amplitudes) > 0 else np.nan
    else:
        mean_amplitude = np.nan
    
    # Mean PR interval - try different column names
    pr_col = None
    for col_name in ['PR_interval_ms', 'pr_interval_ms']:
        if col_name in output_df.columns:
            pr_col = col_name
            break
    
    if pr_col:
        pr_intervals = output_df[pr_col].dropna()
        # Filter to physiological range (80-300ms)
        pr_valid = pr_intervals[(pr_intervals >= 80) & (pr_intervals <= 300)]
        mean_pr_interval_ms = pr_valid.mean() if len(pr_valid) > 0 else np.nan
        valid_pr_ratio = (len(pr_valid) / len(pr_intervals)) * 100.0 if len(pr_intervals) > 0 else 0.0
    else:
        mean_pr_interval_ms = np.nan
        valid_pr_ratio = 0.0
    
    # Mean duration (if available) - try different column names
    p_dur_col = None
    for col_name in ['P_duration_ms', 'p_duration_ms']:
        if col_name in output_df.columns:
            p_dur_col = col_name
            break
    
    if p_dur_col:
        durations = output_df[p_dur_col].dropna()
        mean_duration_ms = durations.mean() if len(durations) > 0 else np.nan
    else:
        mean_duration_ms = np.nan
    
    return {
        'detection_rate': detection_rate,
        'mean_amplitude': mean_amplitude,
        'mean_pr_interval_ms': mean_pr_interval_ms,
        'mean_duration_ms': mean_duration_ms,
        'valid_pr_ratio': valid_pr_ratio,
        'p_detected': p_detected,
        'total_cycles': total_cycles
    }




def test_configuration(subject, cfg, config_name):
    """Test P-wave detection with a specific configuration."""
    print(f"    {config_name}...", end=" ", flush=True)
    
    try:
        # Load ECG signal
        old_dir = os.getcwd()
        os.chdir(QTDB_DATA_DIR)
        record = wfdb.rdrecord(subject)
        os.chdir(old_dir)
        
        ecg_signal = record.p_signal[:, 0]  # Lead II
        sampling_rate = record.fs
        
        # Run PyHEARTS
        analyzer = PyHEARTS(sampling_rate=sampling_rate, cfg=cfg)
        output_df, epochs_df = analyzer.analyze_ecg(ecg_signal)
        
        # Analyze P-wave detections
        stats = analyze_p_detections(output_df, sampling_rate)
        
        print(f"✓ ({stats['p_detected']}/{stats['total_cycles']} detected, {stats['detection_rate']:.1f}%)")
        return stats
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return None


def main():
    print("=" * 80)
    print("P-Wave Validation Impact Test")
    print("=" * 80)
    print(f"Testing {len(TEST_SUBJECTS)} QTDB subjects")
    print("Note: QTDB does not have P-wave ground truth, comparing detection statistics")
    print()
    
    all_results = []
    
    for subject in TEST_SUBJECTS:
        print(f"\n{subject}:")
        
        # Test four configurations
        base_cfg = ProcessCycleConfig.for_human()
        configs = [
            ("Both enabled", replace(
                base_cfg,
                p_use_derivative_validated_method=True,
                p_enable_distance_validation=True,
                p_enable_morphology_validation=True
            )),
            ("Distance disabled", replace(
                base_cfg,
                p_use_derivative_validated_method=True,
                p_enable_distance_validation=False,
                p_enable_morphology_validation=True
            )),
            ("Morphology disabled", replace(
                base_cfg,
                p_use_derivative_validated_method=True,
                p_enable_distance_validation=True,
                p_enable_morphology_validation=False
            )),
            ("Both disabled", replace(
                base_cfg,
                p_use_derivative_validated_method=True,
                p_enable_distance_validation=False,
                p_enable_morphology_validation=False
            ))
        ]
        
        for config_name, cfg in configs:
            stats = test_configuration(subject, cfg, config_name)
            if stats is not None:
                all_results.append({
                    'subject': subject,
                    'config': config_name,
                    **stats
                })
    
    if len(all_results) == 0:
        print("\nNo results collected. Check if subjects are available.")
        return
    
    # Aggregate results
    print(f"\n{'='*80}")
    print("AGGREGATE RESULTS")
    print(f"{'='*80}")
    
    df = pd.DataFrame(all_results)
    
    for config_name in ["Both enabled", "Distance disabled", "Morphology disabled", "Both disabled"]:
        config_df = df[df['config'] == config_name]
        if len(config_df) == 0:
            continue
        
        print(f"\n{config_name}:")
        print(f"  Detection rate:     {config_df['detection_rate'].mean():.2f}% ± {config_df['detection_rate'].std():.2f}%")
        print(f"  Mean amplitude:      {config_df['mean_amplitude'].mean():.4f} ± {config_df['mean_amplitude'].std():.4f} mV")
        print(f"  Mean PR interval:    {config_df['mean_pr_interval_ms'].mean():.1f} ± {config_df['mean_pr_interval_ms'].std():.1f} ms")
        print(f"  Valid PR ratio:      {config_df['valid_pr_ratio'].mean():.1f}% ± {config_df['valid_pr_ratio'].std():.1f}%")
        print(f"  Total detected:     {config_df['p_detected'].sum()}")
        print(f"  Total cycles:       {config_df['total_cycles'].sum()}")
    
    # Determine winner
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    
    aggregate_metrics = df.groupby('config').agg({
        'detection_rate': 'mean',
        'mean_amplitude': 'mean',
        'mean_pr_interval_ms': 'mean',
        'valid_pr_ratio': 'mean',
        'p_detected': 'sum',
        'total_cycles': 'sum'
    }).round(2)
    
    print("\nHighest Detection Rate:")
    best_detection = aggregate_metrics['detection_rate'].idxmax()
    print(f"  {best_detection}: {aggregate_metrics.loc[best_detection, 'detection_rate']:.2f}%")
    
    print("\nBest Valid PR Ratio:")
    best_pr = aggregate_metrics['valid_pr_ratio'].idxmax()
    print(f"  {best_pr}: {aggregate_metrics.loc[best_pr, 'valid_pr_ratio']:.2f}%")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    # Compare both enabled vs both disabled
    both_enabled = aggregate_metrics.loc['Both enabled']
    both_disabled = aggregate_metrics.loc['Both disabled']
    
    detection_diff = both_enabled['detection_rate'] - both_disabled['detection_rate']
    pr_ratio_diff = both_enabled['valid_pr_ratio'] - both_disabled['valid_pr_ratio']
    
    print(f"\nBoth enabled vs Both disabled:")
    print(f"  Detection rate difference: {detection_diff:+.2f}%")
    print(f"  Valid PR ratio difference: {pr_ratio_diff:+.2f}%")
    
    if detection_diff < -5.0:
        print("\n✓ RECOMMENDATION: Disable validations (significantly increases detection rate)")
        print("  Validations may be too restrictive, rejecting valid P-waves")
    elif pr_ratio_diff > 10.0:
        print("\n✓ RECOMMENDATION: Keep validations enabled (significantly improves PR interval validity)")
        print("  Validations help filter out false positives with invalid timing")
    elif abs(detection_diff) < 2.0 and abs(pr_ratio_diff) < 5.0:
        print("\n✓ RECOMMENDATION: Validations have minimal impact - can be kept optional")
        print("  Both configurations perform similarly")
    else:
        print("\n✓ RECOMMENDATION: Keep validations optional based on use case")
        print("  Enable for higher precision, disable for higher recall")
    
    # Save detailed results
    output_file = SCRIPT_DIR / "results" / "p_validation_impact_results.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()

