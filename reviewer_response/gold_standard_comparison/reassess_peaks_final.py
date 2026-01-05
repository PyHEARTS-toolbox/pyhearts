#!/usr/bin/env python3
"""
Re-assess T, P, S, and Q Peak Detection: PyHEARTS vs ECGpuwave
With Major Outlier Removal

This script:
1. Uses the comparison data from PyHEARTS vs ECGpuwave direct comparison
2. Removes major outliers using multiple thresholds
3. Recalculates statistics for T, P, S, and Q peak detection
4. Creates comprehensive comparison report
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Paths
COMPARISON_FILE = "/Users/morganfitzgerald/Documents/pyhearts/reviewer_response/gold_standard_comparison/peak_detection_reassessment_v3/full_comparison_data.csv"
OUTPUT_DIR = "/Users/morganfitzgerald/Documents/pyhearts/reviewer_response/gold_standard_comparison/peak_detection_final_reassessment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_statistics_with_outlier_removal(
    errors: np.ndarray,
    detected: np.ndarray,
    outlier_thresholds: List[float] = [200.0, 150.0, 100.0, 80.0]
) -> Dict[str, Dict]:
    """Calculate statistics with different outlier removal thresholds."""
    stats_by_threshold = {}
    
    # All data (no outlier removal)
    detected_errors_all = errors[detected]
    if len(detected_errors_all) > 0:
        abs_errors_all = np.abs(detected_errors_all)
        stats_by_threshold['all'] = {
            'n': len(detected_errors_all),
            'mean_error': np.mean(abs_errors_all),
            'median_error': np.median(abs_errors_all),
            'std_error': np.std(abs_errors_all),
            'min_error': np.min(abs_errors_all),
            'max_error': np.max(abs_errors_all),
            'p95_error': np.percentile(abs_errors_all, 95),
            'p99_error': np.percentile(abs_errors_all, 99),
            'within_20ms': (abs_errors_all <= 20).sum() / len(detected_errors_all) * 100,
            'within_50ms': (abs_errors_all <= 50).sum() / len(detected_errors_all) * 100,
            'outliers_removed': 0,
            'outliers_removed_pct': 0.0,
        }
    
    # With outlier removal
    for threshold in outlier_thresholds:
        mask = np.abs(errors) <= threshold
        filtered_detected = detected & mask
        filtered_errors = errors[filtered_detected]
        
        if len(filtered_errors) > 0:
            outliers_removed = detected.sum() - filtered_detected.sum()
            abs_filtered_errors = np.abs(filtered_errors)
            stats_by_threshold[f'threshold_{int(threshold)}ms'] = {
                'n': len(filtered_errors),
                'outliers_removed': outliers_removed,
                'outliers_removed_pct': outliers_removed / detected.sum() * 100 if detected.sum() > 0 else 0.0,
                'mean_error': np.mean(abs_filtered_errors),
                'median_error': np.median(abs_filtered_errors),
                'std_error': np.std(abs_filtered_errors),
                'min_error': np.min(abs_filtered_errors),
                'max_error': np.max(abs_filtered_errors),
                'p95_error': np.percentile(abs_filtered_errors, 95),
                'p99_error': np.percentile(abs_filtered_errors, 99),
                'within_20ms': (abs_filtered_errors <= 20).sum() / len(filtered_errors) * 100,
                'within_50ms': (abs_filtered_errors <= 50).sum() / len(filtered_errors) * 100,
            }
    
    return stats_by_threshold

def main():
    print("="*80)
    print("Re-assess Peak Detection: PyHEARTS vs ECGpuwave")
    print("With Major Outlier Removal")
    print("="*80)
    
    # Load comparison data
    if not os.path.exists(COMPARISON_FILE):
        print(f"\nError: Comparison file not found: {COMPARISON_FILE}")
        return 1
    
    print(f"\nLoading comparison data from: {COMPARISON_FILE}")
    df = pd.read_csv(COMPARISON_FILE)
    print(f"Loaded {len(df)} beats")
    
    # Analyze each peak type with outlier removal
    peak_types = ['P', 'T', 'Q', 'S']
    outlier_thresholds = [200.0, 150.0, 100.0, 80.0]
    
    print("\n" + "="*80)
    print("ANALYSIS WITH OUTLIER REMOVAL")
    print("="*80)
    
    all_summary_rows = []
    
    for peak_type in peak_types:
        error_col = f'{peak_type.lower()}_error_ms'
        detected_col = f'{peak_type.lower()}_detected'
        
        if error_col not in df.columns or detected_col not in df.columns:
            print(f"\n{peak_type} Peak: Columns not found, skipping")
            continue
        
        print(f"\n{peak_type} Peak Detection:")
        print("-" * 70)
        
        errors = df[error_col].fillna(0).values
        detected = df[detected_col].fillna(False).values
        
        # Calculate statistics
        stats_dict = calculate_statistics_with_outlier_removal(
            errors, detected, outlier_thresholds
        )
        
        for threshold_name, stats in stats_dict.items():
            if threshold_name == 'all':
                print(f"\n  All data (no outlier removal):")
            else:
                threshold_ms = threshold_name.replace('threshold_', '').replace('ms', '')
                print(f"\n  Outlier removal: > {threshold_ms} ms")
                print(f"    Outliers removed: {stats['outliers_removed']} ({stats['outliers_removed_pct']:.1f}%)")
            
            print(f"    N: {stats['n']}")
            print(f"    Mean error: {stats['mean_error']:.2f} ms")
            print(f"    Median error: {stats['median_error']:.2f} ms")
            print(f"    Std error: {stats['std_error']:.2f} ms")
            print(f"    P95 error: {stats['p95_error']:.2f} ms")
            print(f"    % within ±20ms: {stats['within_20ms']:.1f}%")
            print(f"    % within ±50ms: {stats['within_50ms']:.1f}%")
            
            all_summary_rows.append({
                'peak': peak_type,
                'outlier_method': threshold_name,
                'n': stats['n'],
                'mean_error_ms': stats['mean_error'],
                'median_error_ms': stats['median_error'],
                'std_error_ms': stats['std_error'],
                'p95_error_ms': stats['p95_error'],
                'within_20ms_pct': stats['within_20ms'],
                'within_50ms_pct': stats['within_50ms'],
                'outliers_removed': stats['outliers_removed'],
                'outliers_removed_pct': stats['outliers_removed_pct'],
            })
    
    # Create summary dataframe
    summary_df = pd.DataFrame(all_summary_rows)
    summary_file = os.path.join(OUTPUT_DIR, 'summary_with_outlier_removal.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\n\nSummary saved to: {summary_file}")
    
    # Create comparison table (recommended threshold: 100ms)
    print("\n" + "="*80)
    print("RECOMMENDED COMPARISON (Outlier Threshold: 100ms)")
    print("="*80)
    
    comparison_rows = []
    for peak_type in peak_types:
        peak_data = summary_df[
            (summary_df['peak'] == peak_type) & 
            (summary_df['outlier_method'] == 'threshold_100ms')
        ]
        if len(peak_data) > 0:
            row = peak_data.iloc[0]
            comparison_rows.append({
                'peak': peak_type,
                'n': row['n'],
                'mean_error_ms': row['mean_error_ms'],
                'median_error_ms': row['median_error_ms'],
                'within_20ms_pct': row['within_20ms_pct'],
                'within_50ms_pct': row['within_50ms_pct'],
                'outliers_removed': row['outliers_removed'],
                'outliers_removed_pct': row['outliers_removed_pct'],
            })
    
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_file = os.path.join(OUTPUT_DIR, 'comparison_100ms_threshold.csv')
    comparison_df.to_csv(comparison_file, index=False)
    print("\nPeak Detection Performance (outliers > 100ms removed):")
    print(comparison_df.to_string(index=False))
    print(f"\nComparison saved to: {comparison_file}")
    
    # Create comprehensive report
    report_file = os.path.join(OUTPUT_DIR, 'comprehensive_report.txt')
    with open(report_file, 'w') as f:
        f.write("Peak Detection Re-assessment: PyHEARTS vs ECGpuwave\n")
        f.write("="*80 + "\n\n")
        f.write("This analysis compares PyHEARTS and ECGpuwave peak detection\n")
        f.write("with different outlier removal thresholds.\n\n")
        f.write("="*80 + "\n")
        f.write("RECOMMENDED COMPARISON (Outlier Threshold: 100ms)\n")
        f.write("="*80 + "\n\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("DETAILED RESULTS BY THRESHOLD\n")
        f.write("="*80 + "\n\n")
        f.write(summary_df.to_string(index=False))
    
    print(f"\nComprehensive report saved to: {report_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

