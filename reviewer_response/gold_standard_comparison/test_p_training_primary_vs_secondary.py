#!/usr/bin/env python3
"""
Test P Training Thresholds: Primary vs Secondary Validation

This script tests whether using training thresholds as PRIMARY validation
(like ECGpuwave) performs better than using them as SECONDARY validation
(current PyHEARTS approach).

Tests on a subset of QTDB subjects and compares:
1. Current: Local MAD primary + Training thresholds secondary (p_use_training_as_primary=False)
2. Proposed: Training thresholds primary + Local MAD secondary (p_use_training_as_primary=True)
"""

import os
import sys
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add pyhearts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyhearts import PyHEARTS
from pyhearts.config import ProcessCycleConfig

# Paths
MANUAL_ANNOTATIONS_DIR = "/Users/morganfitzgerald/Documents/pyhearts/data/qtdb/1.0.0"
OUTPUT_DIR = "/Users/morganfitzgerald/Documents/pyhearts/reviewer_response/gold_standard_comparison/p_training_primary_vs_secondary_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test subjects (subset for testing)
TEST_SUBJECTS = [
    'sel853', 'sel233', 'sel45', 'sel46', 'sel811', 'sel883',
    'sel116', 'sel213', 'sel100', 'sel16539'
]

def load_manual_annotations(record_name: str, annotation_file: str = 'q1c', sampling_rate: float = 250.0) -> List[Dict]:
    """Load manual annotations from QT Database .q1c file."""
    annotation_path = os.path.join(MANUAL_ANNOTATIONS_DIR, f"{record_name}.{annotation_file}")
    
    if not os.path.exists(annotation_path):
        return []
    
    try:
        old_dir = os.getcwd()
        os.chdir(MANUAL_ANNOTATIONS_DIR)
        ann = wfdb.rdann(record_name, annotation_file)
        os.chdir(old_dir)
    except Exception as e:
        try:
            os.chdir(old_dir)
        except:
            pass
        return []
    
    beats = []
    r_peak_indices = []
    for j in range(len(ann.symbol)):
        if ann.symbol[j] == 'N':  # Normal beat
            r_peak_indices.append(j)
    
    for r_idx in r_peak_indices:
        beat = {
            'r_peak': ann.sample[r_idx],
            'p_peak': None,
        }
        
        # Find P peak (before R)
        for j in range(r_idx - 1, max(-1, r_idx - 40), -1):
            if ann.symbol[j] == 'p' or ann.symbol[j] == 'P':
                beat['p_peak'] = ann.sample[j]
                break
        
        beats.append(beat)
    
    return beats

def load_ecg_signal(record_name: str) -> Tuple[Optional[np.ndarray], float]:
    """Load ECG signal from QT Database."""
    try:
        old_dir = os.getcwd()
        os.chdir(MANUAL_ANNOTATIONS_DIR)
        record = wfdb.rdrecord(record_name)
        os.chdir(old_dir)
        
        if record.p_signal is None or record.p_signal.shape[1] == 0:
            return None, record.fs
        
        signal = record.p_signal[:, 0]
        sampling_rate = float(record.fs)
        return signal, sampling_rate
    except Exception as e:
        try:
            os.chdir(old_dir)
        except:
            pass
        return None, 250.0

def analyze_with_validation_method(
    subject: str,
    use_training_as_primary: bool = False
) -> Optional[pd.DataFrame]:
    """
    Analyze P peak detection with specified validation method.
    """
    validation_method = 'training_primary' if use_training_as_primary else 'local_mad_primary'
    print(f"\nAnalyzing {subject} ({validation_method})...")
    
    # Load signal
    signal, sampling_rate = load_ecg_signal(subject)
    if signal is None:
        return None
    
    # Load manual annotations
    manual_beats = load_manual_annotations(subject, 'q1c', sampling_rate)
    if len(manual_beats) == 0:
        return None
    
    # Create config with validation method
    cfg = ProcessCycleConfig(
        p_use_training_phase=True,  # Enable training phase
        p_use_training_as_primary=use_training_as_primary,  # Use training as primary or secondary
    )
    
    # Run PyHEARTS
    try:
        analyzer = PyHEARTS(
            sampling_rate=sampling_rate,
            verbose=False,
            cfg=cfg,
        )
        output_df, epochs_df = analyzer.analyze_ecg(signal)
        
        if output_df is None or len(output_df) == 0:
            return None
        
        r_peaks_global = output_df['R_global_center_idx'].dropna().values
        r_peaks_global = np.array([int(r) for r in r_peaks_global if not np.isnan(r)], dtype=int)
        
    except Exception as e:
        import traceback
        print(f"  Error: {e}")
        traceback.print_exc()
        return None
    
    # Match manual beats to PyHEARTS results
    results = []
    
    for man_beat in manual_beats:
        if man_beat['r_peak'] is None or man_beat['p_peak'] is None:
            continue
        
        man_r = man_beat['r_peak']
        man_p = man_beat['p_peak']
        
        # Find closest PyHEARTS R peak
        distances = np.abs(r_peaks_global - man_r)
        if len(distances) == 0:
            continue
        
        closest_r_idx = np.argmin(distances)
        closest_r = r_peaks_global[closest_r_idx]
        
        # Only analyze if R peaks are close (within 50ms)
        r_distance_ms = abs(closest_r - man_r) * 1000.0 / sampling_rate
        if r_distance_ms > 50.0:
            continue
        
        # Find corresponding P peak in PyHEARTS results
        detected_p = None
        error_p_ms = np.nan
        
        for idx, row in output_df.iterrows():
            if pd.notna(row.get('R_global_center_idx')) and int(row['R_global_center_idx']) == closest_r:
                if pd.notna(row.get('P_global_center_idx')):
                    detected_p = int(row['P_global_center_idx'])
                    error_p_ms = abs(detected_p - man_p) * 1000.0 / sampling_rate
                break
        
        results.append({
            'subject': subject,
            'validation_method': validation_method,
            'manual_r': man_r,
            'manual_p': man_p,
            'pyhearts_r': closest_r,
            'pyhearts_p': detected_p if detected_p is not None else np.nan,
            'error_p_ms': error_p_ms if not np.isnan(error_p_ms) else np.nan,
            'p_detected': detected_p is not None,
        })
    
    if len(results) == 0:
        return None
    
    df = pd.DataFrame(results)
    return df

def main():
    print("="*80)
    print("Test: P Training Thresholds - Primary vs Secondary Validation")
    print("="*80)
    
    all_results = []
    
    # Test both validation methods on each subject
    for subject in TEST_SUBJECTS:
        # Test 1: Local MAD primary (current approach)
        result1 = analyze_with_validation_method(subject, use_training_as_primary=False)
        if result1 is not None:
            all_results.append(result1)
        
        # Test 2: Training primary (proposed approach)
        result2 = analyze_with_validation_method(subject, use_training_as_primary=True)
        if result2 is not None:
            all_results.append(result2)
    
    if len(all_results) == 0:
        print("\nNo results generated!")
        return 1
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save detailed results
    results_file = os.path.join(OUTPUT_DIR, 'p_training_primary_vs_secondary_results.csv')
    combined_df.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")
    
    # Calculate summary statistics
    summary_data = []
    
    for method in ['local_mad_primary', 'training_primary']:
        method_df = combined_df[combined_df['validation_method'] == method]
        
        if len(method_df) == 0:
            continue
        
        detected_df = method_df[method_df['p_detected']]
        all_errors = method_df['error_p_ms'].dropna().values
        
        summary_data.append({
            'validation_method': method,
            'total_beats': len(method_df),
            'p_detected': len(detected_df),
            'p_detection_rate': len(detected_df) / len(method_df) * 100.0 if len(method_df) > 0 else 0.0,
            'mean_error_ms': np.mean(all_errors) if len(all_errors) > 0 else np.nan,
            'median_error_ms': np.median(all_errors) if len(all_errors) > 0 else np.nan,
            'std_error_ms': np.std(all_errors) if len(all_errors) > 0 else np.nan,
            'pct_within_20ms': np.sum(np.abs(all_errors) <= 20.0) / len(all_errors) * 100.0 if len(all_errors) > 0 else 0.0,
            'pct_within_50ms': np.sum(np.abs(all_errors) <= 50.0) / len(all_errors) * 100.0 if len(all_errors) > 0 else 0.0,
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_file = os.path.join(OUTPUT_DIR, 'p_training_primary_vs_secondary_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("P Training Thresholds: Primary vs Secondary Validation Comparison\n")
        f.write("="*80 + "\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write("-"*80 + "\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")
        
        # Calculate differences
        if len(summary_df) == 2:
            local_mad = summary_df[summary_df['validation_method'] == 'local_mad_primary'].iloc[0]
            training = summary_df[summary_df['validation_method'] == 'training_primary'].iloc[0]
            
            f.write("COMPARISON:\n")
            f.write("-"*80 + "\n")
            f.write(f"Detection Rate:\n")
            f.write(f"  Local MAD primary: {local_mad['p_detection_rate']:.1f}%\n")
            f.write(f"  Training primary:  {training['p_detection_rate']:.1f}%\n")
            f.write(f"  Difference:        {training['p_detection_rate'] - local_mad['p_detection_rate']:.1f} percentage points\n\n")
            
            f.write(f"Mean Error:\n")
            f.write(f"  Local MAD primary: {local_mad['mean_error_ms']:.2f} ms\n")
            f.write(f"  Training primary:  {training['mean_error_ms']:.2f} ms\n")
            error_diff = training['mean_error_ms'] - local_mad['mean_error_ms']
            f.write(f"  Difference:        {error_diff:.2f} ms ({error_diff/local_mad['mean_error_ms']*100:.1f}% change)\n\n")
            
            f.write(f"% within ±20ms:\n")
            f.write(f"  Local MAD primary: {local_mad['pct_within_20ms']:.1f}%\n")
            f.write(f"  Training primary:  {training['pct_within_20ms']:.1f}%\n")
            pct_diff = training['pct_within_20ms'] - local_mad['pct_within_20ms']
            f.write(f"  Difference:        {pct_diff:.1f} percentage points\n\n")
            
            # Conclusion
            f.write("CONCLUSION:\n")
            f.write("-"*80 + "\n")
            if error_diff < 0:
                f.write(f"✓ Training primary performs BETTER (lower error by {abs(error_diff):.2f} ms)\n")
            elif error_diff > 0:
                f.write(f"✗ Training primary performs WORSE (higher error by {error_diff:.2f} ms)\n")
            else:
                f.write("= Training primary performs similarly\n")
            
            if pct_diff > 0:
                f.write(f"✓ Training primary has HIGHER % within ±20ms (+{pct_diff:.1f} percentage points)\n")
            elif pct_diff < 0:
                f.write(f"✗ Training primary has LOWER % within ±20ms ({pct_diff:.1f} percentage points)\n")
            else:
                f.write("= Training primary has similar % within ±20ms\n")
    
    print(f"\nSummary saved to: {summary_file}")
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
