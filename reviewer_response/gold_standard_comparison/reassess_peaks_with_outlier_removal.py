#!/usr/bin/env python3
"""
Re-assess T, P, S, and Q Peak Detection: PyHEARTS vs ECGpuwave
With Major Outlier Removal

This script:
1. Loads per-beat comparison data from PyHEARTS vs Manual and ECGpuwave vs Manual
2. Combines them to create PyHEARTS vs ECGpuwave comparisons
3. Removes major outliers (errors > threshold)
4. Recalculates statistics for T, P, S, and Q peak detection
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Paths - need to check what directories exist
PYHEARTS_VS_MANUAL_DIR = "/Users/morganfitzgerald/Documents/pyhearts/reviewer_response/gold_standard_comparison/pyhearts_vs_manual"
ECGPUWAVE_VS_MANUAL_DIR = "/Users/morganfitzgerald/Documents/pyhearts/reviewer_response/gold_standard_comparison/ecgpuwave_vs_manual"
OUTPUT_DIR = "/Users/morganfitzgerald/Documents/pyhearts/reviewer_response/gold_standard_comparison/peak_detection_reassessment_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_summary_statistics():
    """Load summary statistics from comparison files."""
    pyhearts_stats = {}
    ecgpuwave_stats = {}
    
    # Load PyHEARTS vs Manual
    pyhearts_mad_file = os.path.join(PYHEARTS_VS_MANUAL_DIR, 'fiducial_mad_summary.csv')
    if os.path.exists(pyhearts_mad_file):
        df = pd.read_csv(pyhearts_mad_file)
        for _, row in df.iterrows():
            fiducial = row['Fiducial'].lower()
            pyhearts_stats[fiducial] = {
                'mean_mad': row['Mean MAD (ms)'],
                'median_mad': row['Median MAD (ms)'],
                'total_matched': row['Total Matched'],
            }
    
    # Load ECGpuwave vs Manual
    ecgpuwave_mad_file = os.path.join(ECGPUWAVE_VS_MANUAL_DIR, 'fiducial_mad_summary.csv')
    if os.path.exists(ecgpuwave_mad_file):
        df = pd.read_csv(ecgpuwave_mad_file)
        for _, row in df.iterrows():
            fiducial = row['Fiducial'].lower()
            ecgpuwave_stats[fiducial] = {
                'mean_mad': row['Mean MAD (ms)'],
                'median_mad': row['Median MAD (ms)'],
                'total_matched': row['Total Matched'],
            }
    
    return pyhearts_stats, ecgpuwave_stats

def create_reassessment_report(
    pyhearts_stats: Dict,
    ecgpuwave_stats: Dict,
    outlier_thresholds: List[float] = [200.0, 150.0, 100.0]
):
    """
    Create reassessment report with outlier removal simulation.
    
    Since we only have summary statistics, we'll simulate the effect of
    outlier removal using reasonable assumptions about error distributions.
    """
    
    peak_types = {
        'p_peak': 'P',
        'r_peak': 'R',
        't_peak': 'T',
        'q_peak': 'Q',
        's_peak': 'S',
    }
    
    results = {}
    
    print("="*80)
    print("Peak Detection Re-assessment: PyHEARTS vs ECGpuwave")
    print("="*80)
    print("\nNote: This analysis uses summary statistics. For detailed per-beat")
    print("outlier removal, per-beat comparison data would be needed.")
    print("\nCurrent Summary Statistics (all data, no outlier removal):")
    print("-"*80)
    
    for peak_key, peak_label in peak_types.items():
        print(f"\n{peak_label} Peak Detection:")
        
        ph = pyhearts_stats.get(peak_key, {})
        ecg = ecgpuwave_stats.get(peak_key, {})
        
        if ph and ecg:
            print(f"  PyHEARTS:")
            print(f"    Mean MAD: {ph.get('mean_mad', np.nan):.2f} ms")
            print(f"    Median MAD: {ph.get('median_mad', np.nan):.2f} ms")
            print(f"    Total matched: {ph.get('total_matched', 0)}")
            
            print(f"  ECGpuwave:")
            print(f"    Mean MAD: {ecg.get('mean_mad', np.nan):.2f} ms")
            print(f"    Median MAD: {ecg.get('median_mad', np.nan):.2f} ms")
            print(f"    Total matched: {ecg.get('total_matched', 0)}")
            
            if 'mean_mad' in ph and 'mean_mad' in ecg:
                diff = ph['mean_mad'] - ecg['mean_mad']
                print(f"  Difference: {diff:.2f} ms (PyHEARTS - ECGpuwave)")
            
            results[peak_label] = {
                'pyhearts': ph,
                'ecgpuwave': ecg,
            }
    
    # Create summary table
    summary_rows = []
    for peak_key, peak_label in peak_types.items():
        ph = pyhearts_stats.get(peak_key, {})
        ecg = ecgpuwave_stats.get(peak_key, {})
        
        if ph and ecg:
            summary_rows.append({
                'peak': peak_label,
                'pyhearts_mean_mad_ms': ph.get('mean_mad', np.nan),
                'pyhearts_median_mad_ms': ph.get('median_mad', np.nan),
                'pyhearts_total': ph.get('total_matched', 0),
                'ecgpuwave_mean_mad_ms': ecg.get('mean_mad', np.nan),
                'ecgpuwave_median_mad_ms': ecg.get('median_mad', np.nan),
                'ecgpuwave_total': ecg.get('total_matched', 0),
                'difference_ms': ph.get('mean_mad', np.nan) - ecg.get('mean_mad', np.nan),
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_file = os.path.join(OUTPUT_DIR, 'summary_comparison.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\n\nSummary comparison saved to: {summary_file}")
    
    return results, summary_df

def main():
    print("="*80)
    print("Re-assess Peak Detection: PyHEARTS vs ECGpuwave")
    print("(Using Summary Statistics)")
    print("="*80)
    
    # Load summary statistics
    print("\nLoading summary statistics...")
    pyhearts_stats, ecgpuwave_stats = load_summary_statistics()
    
    if not pyhearts_stats or not ecgpuwave_stats:
        print("\nError: Could not load summary statistics files.")
        print(f"  PyHEARTS stats: {PYHEARTS_VS_MANUAL_DIR}")
        print(f"  ECGpuwave stats: {ECGPUWAVE_VS_MANUAL_DIR}")
        return 1
    
    print(f"  Loaded PyHEARTS stats: {len(pyhearts_stats)} peak types")
    print(f"  Loaded ECGpuwave stats: {len(ecgpuwave_stats)} peak types")
    
    # Create reassessment report
    results, summary_df = create_reassessment_report(pyhearts_stats, ecgpuwave_stats)
    
    # Create detailed report
    report_file = os.path.join(OUTPUT_DIR, 'reassessment_report.txt')
    with open(report_file, 'w') as f:
        f.write("Peak Detection Re-assessment: PyHEARTS vs ECGpuwave\n")
        f.write("="*80 + "\n\n")
        f.write("NOTE: This analysis uses summary statistics from comparison files.\n")
        f.write("For detailed per-beat outlier removal, per-beat comparison data\n")
        f.write("would need to be generated from raw PyHEARTS and ECGpuwave results.\n\n")
        f.write("Current Summary (All Data, No Outlier Removal):\n")
        f.write("-"*80 + "\n\n")
        
        for peak_type in ['P', 'R', 'T', 'Q', 'S']:
            peak_key = f'{peak_type.lower()}_peak'
            if peak_key in results:
                ph = results[peak_type]['pyhearts']
                ecg = results[peak_type]['ecgpuwave']
                
                f.write(f"{peak_type} Peak:\n")
                f.write(f"  PyHEARTS: {ph.get('mean_mad', np.nan):.2f} ms mean MAD")
                f.write(f" ({ph.get('total_matched', 0)} matched)\n")
                f.write(f"  ECGpuwave: {ecg.get('mean_mad', np.nan):.2f} ms mean MAD")
                f.write(f" ({ecg.get('total_matched', 0)} matched)\n")
                
                if 'mean_mad' in ph and 'mean_mad' in ecg:
                    diff = ph['mean_mad'] - ecg['mean_mad']
                    f.write(f"  Difference: {diff:.2f} ms\n")
                f.write("\n")
    
    print(f"\nReport saved to: {report_file}")
    
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    print("\nFor detailed outlier removal analysis, per-beat comparison data")
    print("is needed. The current summary statistics don't allow per-beat")
    print("outlier filtering. To generate per-beat comparisons:")
    print("\n  1. Run PyHEARTS on test subjects")
    print("  2. Load ECGpuwave annotations")
    print("  3. Match peaks beat-by-beat")
    print("  4. Calculate per-beat errors")
    print("  5. Remove outliers and recalculate statistics")
    print("\nCurrent summary shows:")
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

