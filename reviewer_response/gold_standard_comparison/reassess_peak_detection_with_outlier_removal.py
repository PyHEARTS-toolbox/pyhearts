#!/usr/bin/env python3
"""
Re-assess T, P, S, and Q Peak Detection: PyHEARTS vs ECGpuwave
With Major Outlier Removal

This script:
1. Loads PyHEARTS and ECGpuwave comparison data
2. Removes major outliers (e.g., errors > 200ms or using IQR method)
3. Recalculates statistics for T, P, S, and Q peak detection
4. Compares PyHEARTS vs ECGpuwave performance
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add pyhearts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Paths
COMPARISON_FILE = "/Users/morganfitzgerald/Documents/pyhearts/reviewer_response/gold_standard_comparison/performance_comparison/ecgpuwave_vs_pyhearts_comparison.csv"
OUTPUT_DIR = "/Users/morganfitzgerald/Documents/pyhearts/reviewer_response/gold_standard_comparison/peak_detection_reassessment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def remove_outliers_iqr(values: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
    """
    Remove outliers using IQR (Interquartile Range) method.
    
    Parameters
    ----------
    values : np.ndarray
        Array of values.
    multiplier : float, default 1.5
        IQR multiplier (1.5 = standard, 3.0 = more conservative).
    
    Returns
    -------
    np.ndarray
        Boolean mask (True = keep, False = outlier).
    """
    if len(values) == 0:
        return np.array([])
    
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    
    if iqr == 0:
        # No variability, keep all
        return np.ones(len(values), dtype=bool)
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return (values >= lower_bound) & (values <= upper_bound)

def remove_outliers_threshold(values: np.ndarray, threshold_ms: float = 200.0) -> np.ndarray:
    """
    Remove outliers using absolute threshold.
    
    Parameters
    ----------
    values : np.ndarray
        Array of error values (in ms).
    threshold_ms : float, default 200.0
        Maximum allowed error (ms).
    
    Returns
    -------
    np.ndarray
        Boolean mask (True = keep, False = outlier).
    """
    return np.abs(values) <= threshold_ms

def calculate_peak_stats(
    errors: np.ndarray,
    detected: np.ndarray,
    label: str
) -> Dict:
    """
    Calculate statistics for peak detection errors.
    
    Parameters
    ----------
    errors : np.ndarray
        Array of errors (ms).
    detected : np.ndarray
        Boolean array indicating if peak was detected.
    label : str
        Label for the peak type (e.g., "P", "T").
    
    Returns
    -------
    dict
        Dictionary with statistics.
    """
    stats = {}
    
    # Detection rate
    total = len(detected)
    detected_count = detected.sum()
    stats['detection_rate'] = detected_count / total * 100 if total > 0 else 0.0
    stats['detected_count'] = detected_count
    stats['total_count'] = total
    
    # Error statistics (only for detected peaks)
    detected_errors = errors[detected]
    if len(detected_errors) > 0:
        stats['mean_error'] = np.mean(np.abs(detected_errors))
        stats['median_error'] = np.median(np.abs(detected_errors))
        stats['std_error'] = np.std(detected_errors)
        stats['min_error'] = np.min(np.abs(detected_errors))
        stats['max_error'] = np.max(np.abs(detected_errors))
        stats['within_20ms'] = (np.abs(detected_errors) <= 20).sum() / len(detected_errors) * 100
        stats['within_50ms'] = (np.abs(detected_errors) <= 50).sum() / len(detected_errors) * 100
        stats['within_100ms'] = (np.abs(detected_errors) <= 100).sum() / len(detected_errors) * 100
    else:
        stats['mean_error'] = np.nan
        stats['median_error'] = np.nan
        stats['std_error'] = np.nan
        stats['min_error'] = np.nan
        stats['max_error'] = np.nan
        stats['within_20ms'] = 0.0
        stats['within_50ms'] = 0.0
        stats['within_100ms'] = 0.0
    
    return stats

def analyze_peak_detection(
    df: pd.DataFrame,
    peak_type: str,
    outlier_method: str = "threshold",
    outlier_param: float = 200.0
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Analyze peak detection with outlier removal.
    
    Parameters
    ----------
    df : pd.DataFrame
        Comparison dataframe.
    peak_type : str
        Peak type ("P", "T", "Q", "S").
    outlier_method : str
        Outlier removal method ("threshold" or "iqr").
    outlier_param : float
        Parameter for outlier removal (threshold in ms or IQR multiplier).
    
    Returns
    -------
    tuple
        (filtered_df, pyhearts_stats, ecgpuwave_stats)
    """
    # Extract error columns
    pyhearts_error_col = f'pyhearts_{peak_type.lower()}_error_ms'
    ecgpuwave_error_col = f'ecgpuwave_{peak_type.lower()}_error_ms'
    
    # Extract detection columns
    pyhearts_detected_col = f'pyhearts_{peak_type.lower()}_detected'
    ecgpuwave_detected_col = f'ecgpuwave_{peak_type.lower()}_detected'
    
    if pyhearts_error_col not in df.columns or ecgpuwave_error_col not in df.columns:
        print(f"  Warning: Missing columns for {peak_type} peak")
        return df, {}, {}
    
    # Get errors (only for detected peaks)
    pyhearts_errors = df[df[pyhearts_detected_col] == True][pyhearts_error_col].dropna().values
    ecgpuwave_errors = df[df[ecgpuwave_detected_col] == True][ecgpuwave_error_col].dropna().values
    
    # Remove outliers
    if outlier_method == "threshold":
        pyhearts_mask = remove_outliers_threshold(pyhearts_errors, outlier_param)
        ecgpuwave_mask = remove_outliers_threshold(ecgpuwave_errors, outlier_param)
    elif outlier_method == "iqr":
        pyhearts_mask = remove_outliers_iqr(pyhearts_errors, outlier_param)
        ecgpuwave_mask = remove_outliers_iqr(ecgpuwave_errors, outlier_param)
    else:
        raise ValueError(f"Unknown outlier method: {outlier_method}")
    
    # Count outliers removed
    pyhearts_outliers = (~pyhearts_mask).sum()
    ecgpuwave_outliers = (~ecgpuwave_mask).sum()
    
    # Calculate statistics (all detected peaks)
    pyhearts_all_detected = df[pyhearts_detected_col] == True
    ecgpuwave_all_detected = df[ecgpuwave_detected_col] == True
    
    # For filtered statistics, we need to mark outliers in the full dataframe
    # Create a filtered dataframe by removing rows with outlier errors
    filtered_df = df.copy()
    
    # Mark outliers in dataframe
    pyhearts_error_values = df[pyhearts_error_col].fillna(0).values
    ecgpuwave_error_values = df[ecgpuwave_error_col].fillna(0).values
    
    if outlier_method == "threshold":
        pyhearts_outlier_mask = np.abs(pyhearts_error_values) > outlier_param
        ecgpuwave_outlier_mask = np.abs(ecgpuwave_error_values) > outlier_param
    else:  # iqr
        # Apply IQR to detected errors
        pyhearts_detected_errors = df[pyhearts_all_detected][pyhearts_error_col].dropna().values
        ecgpuwave_detected_errors = df[ecgpuwave_all_detected][ecgpuwave_error_col].dropna().values
        
        if len(pyhearts_detected_errors) > 0:
            pyhearts_q1 = np.percentile(pyhearts_detected_errors, 25)
            pyhearts_q3 = np.percentile(pyhearts_detected_errors, 75)
            pyhearts_iqr = pyhearts_q3 - pyhearts_q1
            pyhearts_lower = pyhearts_q1 - outlier_param * pyhearts_iqr
            pyhearts_upper = pyhearts_q3 + outlier_param * pyhearts_iqr
            pyhearts_outlier_mask = (pyhearts_error_values < pyhearts_lower) | (pyhearts_error_values > pyhearts_upper)
        else:
            pyhearts_outlier_mask = np.zeros(len(df), dtype=bool)
        
        if len(ecgpuwave_detected_errors) > 0:
            ecgpuwave_q1 = np.percentile(ecgpuwave_detected_errors, 25)
            ecgpuwave_q3 = np.percentile(ecgpuwave_detected_errors, 75)
            ecgpuwave_iqr = ecgpuwave_q3 - ecgpuwave_q1
            ecgpuwave_lower = ecgpuwave_q1 - outlier_param * ecgpuwave_iqr
            ecgpuwave_upper = ecgpuwave_q3 + outlier_param * ecgpuwave_iqr
            ecgpuwave_outlier_mask = (ecgpuwave_error_values < ecgpuwave_lower) | (ecgpuwave_error_values > ecgpuwave_upper)
        else:
            ecgpuwave_outlier_mask = np.zeros(len(df), dtype=bool)
    
    # Remove rows where either method has an outlier for this peak type
    combined_outlier_mask = pyhearts_outlier_mask | ecgpuwave_outlier_mask
    filtered_df = df[~combined_outlier_mask].copy()
    
    # Calculate statistics on filtered data
    pyhearts_filtered_detected = filtered_df[pyhearts_detected_col] == True
    ecgpuwave_filtered_detected = filtered_df[ecgpuwave_detected_col] == True
    
    pyhearts_filtered_errors = filtered_df[pyhearts_filtered_detected][pyhearts_error_col].dropna().values
    ecgpuwave_filtered_errors = filtered_df[ecgpuwave_filtered_detected][ecgpuwave_error_col].dropna().values
    
    pyhearts_stats = calculate_peak_stats(
        filtered_df[pyhearts_error_col].fillna(0).values,
        pyhearts_filtered_detected.values,
        f"PyHEARTS {peak_type}"
    )
    pyhearts_stats['outliers_removed'] = pyhearts_outliers
    pyhearts_stats['outliers_removed_pct'] = pyhearts_outliers / len(pyhearts_errors) * 100 if len(pyhearts_errors) > 0 else 0.0
    
    ecgpuwave_stats = calculate_peak_stats(
        filtered_df[ecgpuwave_error_col].fillna(0).values,
        ecgpuwave_filtered_detected.values,
        f"ECGpuwave {peak_type}"
    )
    ecgpuwave_stats['outliers_removed'] = ecgpuwave_outliers
    ecgpuwave_stats['outliers_removed_pct'] = ecgpuwave_outliers / len(ecgpuwave_errors) * 100 if len(ecgpuwave_errors) > 0 else 0.0
    
    return filtered_df, pyhearts_stats, ecgpuwave_stats

def main():
    print("="*80)
    print("Re-assess Peak Detection: PyHEARTS vs ECGpuwave (with Outlier Removal)")
    print("="*80)
    
    # Load comparison data
    if not os.path.exists(COMPARISON_FILE):
        print(f"\nError: Comparison file not found: {COMPARISON_FILE}")
        return 1
    
    print(f"\nLoading comparison data from: {COMPARISON_FILE}")
    df = pd.read_csv(COMPARISON_FILE)
    print(f"Loaded {len(df)} comparisons")
    
    # Analyze each peak type with different outlier removal methods
    peak_types = ['T', 'P', 'Q', 'S']
    outlier_methods = [
        ("threshold_200ms", "threshold", 200.0),
        ("threshold_150ms", "threshold", 150.0),
        ("iqr_1.5", "iqr", 1.5),
        ("iqr_3.0", "iqr", 3.0),
    ]
    
    all_results = {}
    
    for method_name, method_type, method_param in outlier_methods:
        print(f"\n{'='*80}")
        print(f"Outlier Removal Method: {method_name} (param={method_param})")
        print(f"{'='*80}")
        
        results = {}
        filtered_df = df.copy()
        
        for peak_type in peak_types:
            print(f"\nAnalyzing {peak_type} peak detection...")
            
            try:
                filtered_df, pyhearts_stats, ecgpuwave_stats = analyze_peak_detection(
                    filtered_df,
                    peak_type,
                    outlier_method=method_type,
                    outlier_param=method_param
                )
                
                if pyhearts_stats and ecgpuwave_stats:
                    results[peak_type] = {
                        'pyhearts': pyhearts_stats,
                        'ecgpuwave': ecgpuwave_stats,
                    }
                    
                    print(f"  PyHEARTS:")
                    print(f"    Detection rate: {pyhearts_stats['detection_rate']:.1f}% ({pyhearts_stats['detected_count']}/{pyhearts_stats['total_count']})")
                    if not np.isnan(pyhearts_stats['mean_error']):
                        print(f"    Mean error: {pyhearts_stats['mean_error']:.2f} ms")
                        print(f"    Median error: {pyhearts_stats['median_error']:.2f} ms")
                        print(f"    % within ±20ms: {pyhearts_stats['within_20ms']:.1f}%")
                        print(f"    Outliers removed: {pyhearts_stats['outliers_removed']} ({pyhearts_stats['outliers_removed_pct']:.1f}%)")
                    
                    print(f"  ECGpuwave:")
                    print(f"    Detection rate: {ecgpuwave_stats['detection_rate']:.1f}% ({ecgpuwave_stats['detected_count']}/{ecgpuwave_stats['total_count']})")
                    if not np.isnan(ecgpuwave_stats['mean_error']):
                        print(f"    Mean error: {ecgpuwave_stats['mean_error']:.2f} ms")
                        print(f"    Median error: {ecgpuwave_stats['median_error']:.2f} ms")
                        print(f"    % within ±20ms: {ecgpuwave_stats['within_20ms']:.1f}%")
                        print(f"    Outliers removed: {ecgpuwave_stats['outliers_removed']} ({ecgpuwave_stats['outliers_removed_pct']:.1f}%)")
                    
                    if not np.isnan(pyhearts_stats['mean_error']) and not np.isnan(ecgpuwave_stats['mean_error']):
                        diff = pyhearts_stats['mean_error'] - ecgpuwave_stats['mean_error']
                        print(f"  Difference: {diff:.2f} ms (PyHEARTS - ECGpuwave)")
            except Exception as e:
                import traceback
                print(f"  Error analyzing {peak_type}: {e}")
                traceback.print_exc()
        
        all_results[method_name] = results
        
        # Save filtered data for this method
        output_file = os.path.join(OUTPUT_DIR, f'filtered_comparison_{method_name}.csv')
        filtered_df.to_csv(output_file, index=False)
        print(f"\n  Filtered data saved to: {output_file}")
    
    # Create summary report
    print(f"\n{'='*80}")
    print("SUMMARY REPORT")
    print(f"{'='*80}")
    
    summary_file = os.path.join(OUTPUT_DIR, 'summary_report.txt')
    with open(summary_file, 'w') as f:
        f.write("Peak Detection Re-assessment: PyHEARTS vs ECGpuwave\n")
        f.write("="*80 + "\n\n")
        
        for method_name in outlier_methods:
            method_name_only = method_name[0]
            if method_name_only not in all_results:
                continue
            
            f.write(f"\n{method_name_only.upper()}\n")
            f.write("-" * 80 + "\n\n")
            
            results = all_results[method_name_only]
            for peak_type in peak_types:
                if peak_type not in results:
                    continue
                
                pyhearts = results[peak_type]['pyhearts']
                ecgpuwave = results[peak_type]['ecgpuwave']
                
                f.write(f"{peak_type} Peak Detection:\n")
                f.write(f"  PyHEARTS: {pyhearts['detection_rate']:.1f}% detected")
                if not np.isnan(pyhearts['mean_error']):
                    f.write(f", {pyhearts['mean_error']:.2f} ms mean error")
                    f.write(f", {pyhearts['within_20ms']:.1f}% within ±20ms")
                f.write("\n")
                
                f.write(f"  ECGpuwave: {ecgpuwave['detection_rate']:.1f}% detected")
                if not np.isnan(ecgpuwave['mean_error']):
                    f.write(f", {ecgpuwave['mean_error']:.2f} ms mean error")
                    f.write(f", {ecgpuwave['within_20ms']:.1f}% within ±20ms")
                f.write("\n")
                
                if not np.isnan(pyhearts['mean_error']) and not np.isnan(ecgpuwave['mean_error']):
                    diff = pyhearts['mean_error'] - ecgpuwave['mean_error']
                    f.write(f"  Difference: {diff:.2f} ms\n")
                f.write("\n")
    
    print(f"\nSummary report saved to: {summary_file}")
    
    # Create detailed CSV with all results
    detailed_rows = []
    for method_name in outlier_methods:
        method_name_only = method_name[0]
        if method_name_only not in all_results:
            continue
        
        results = all_results[method_name_only]
        for peak_type in peak_types:
            if peak_type not in results:
                continue
            
            pyhearts = results[peak_type]['pyhearts']
            ecgpuwave = results[peak_type]['ecgpuwave']
            
            detailed_rows.append({
                'outlier_method': method_name_only,
                'peak_type': peak_type,
                'method': 'PyHEARTS',
                'detection_rate': pyhearts['detection_rate'],
                'mean_error': pyhearts['mean_error'],
                'median_error': pyhearts['median_error'],
                'std_error': pyhearts['std_error'],
                'within_20ms': pyhearts['within_20ms'],
                'within_50ms': pyhearts['within_50ms'],
                'within_100ms': pyhearts['within_100ms'],
                'outliers_removed': pyhearts['outliers_removed'],
                'outliers_removed_pct': pyhearts['outliers_removed_pct'],
            })
            
            detailed_rows.append({
                'outlier_method': method_name_only,
                'peak_type': peak_type,
                'method': 'ECGpuwave',
                'detection_rate': ecgpuwave['detection_rate'],
                'mean_error': ecgpuwave['mean_error'],
                'median_error': ecgpuwave['median_error'],
                'std_error': ecgpuwave['std_error'],
                'within_20ms': ecgpuwave['within_20ms'],
                'within_50ms': ecgpuwave['within_50ms'],
                'within_100ms': ecgpuwave['within_100ms'],
                'outliers_removed': ecgpuwave['outliers_removed'],
                'outliers_removed_pct': ecgpuwave['outliers_removed_pct'],
            })
    
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_file = os.path.join(OUTPUT_DIR, 'detailed_results.csv')
    detailed_df.to_csv(detailed_file, index=False)
    print(f"Detailed results saved to: {detailed_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

