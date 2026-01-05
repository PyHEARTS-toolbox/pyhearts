#!/usr/bin/env python3
"""
Test T-Dynamic-Window Improvement on Subjects with Variable RR Intervals

This script tests the T-dynamic-window improvement specifically on subjects
with highly variable RR intervals, where the improvement should have the most impact.
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

from pyhearts import PyHEARTS, ProcessCycleConfig

# Paths
MANUAL_ANNOTATIONS_DIR = "/Users/morganfitzgerald/Documents/pyhearts/data/qtdb/1.0.0"
OUTPUT_DIR = "/Users/morganfitzgerald/Documents/pyhearts/reviewer_response/gold_standard_comparison/t_dynamic_window_variable_rr_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test subjects - will be populated from variability analysis or use default high-variability subjects
# These are subjects known to have variable RR intervals
TEST_SUBJECTS = [
    # Add subjects with high RR variability here after running find_variable_rr_subjects.py
    # For now, using a mix of subjects that might have variability
    'sel100', 'sel102', 'sel103', 'sel114', 'sel116', 'sel117', 'sel123',
    'sel213', 'sel221', 'sel223', 'sel230', 'sel231', 'sel232',
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
            'p_onset': None,
            'p_peak': None,
            't_peak': None,
            't_end': None
        }
        
        # Look backwards from R peak for P wave annotations
        for j in range(r_idx - 1, max(-1, r_idx - 20), -1):
            if ann.symbol[j] == 'p':
                beat['p_peak'] = ann.sample[j]
                for k in range(max(0, j - 5), j):
                    if ann.symbol[k] == '(':
                        if k + 1 == j:
                            beat['p_onset'] = ann.sample[k]
                        else:
                            has_p_between = any(ann.symbol[m] == 'p' for m in range(k + 1, j))
                            if not has_p_between:
                                beat['p_onset'] = ann.sample[k]
                        if beat['p_onset'] is not None:
                            break
                break
        
        # Look forward from R peak for T wave annotations
        for j in range(r_idx + 1, min(len(ann.symbol), r_idx + 20)):
            if ann.symbol[j] in ['t', 'T']:
                beat['t_peak'] = ann.sample[j]
                for k in range(j + 1, min(len(ann.symbol), j + 10)):
                    if ann.symbol[k] == ')':
                        beat['t_end'] = ann.sample[k]
                        break
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

def compute_peak_metrics(
    detected_peaks: np.ndarray,
    manual_peaks: np.ndarray,
    sampling_rate: float,
    max_match_distance_ms: float = 100.0,
) -> Dict[str, float]:
    """Compute performance metrics (MAD, within ±20ms)."""
    if len(detected_peaks) == 0 or len(manual_peaks) == 0:
        return {
            'mean_mad_ms': np.nan,
            'within_20ms_pct': 0.0,
            'n_matched': 0,
            'n_detected': len(detected_peaks),
            'n_manual': len(manual_peaks),
        }
    
    matched_diffs = []
    for det_peak in detected_peaks:
        distances = np.abs(manual_peaks - det_peak)
        closest_idx = np.argmin(distances)
        closest_dist = distances[closest_idx]
        
        max_distance_samples = int(max_match_distance_ms * sampling_rate / 1000.0)
        if closest_dist <= max_distance_samples:
            matched_diffs.append(closest_dist)
    
    if len(matched_diffs) == 0:
        return {
            'mean_mad_ms': np.nan,
            'within_20ms_pct': 0.0,
            'n_matched': 0,
            'n_detected': len(detected_peaks),
            'n_manual': len(manual_peaks),
        }
    
    matched_diffs_ms = np.array(matched_diffs) / sampling_rate * 1000.0
    mean_mad_ms = np.mean(np.abs(matched_diffs_ms))
    within_20ms = np.sum(np.abs(matched_diffs_ms) <= 20.0) / len(matched_diffs_ms) * 100.0
    
    return {
        'mean_mad_ms': mean_mad_ms,
        'within_20ms_pct': within_20ms,
        'n_matched': len(matched_diffs),
        'n_detected': len(detected_peaks),
        'n_manual': len(manual_peaks),
    }

def extract_t_peaks_from_output(output_df: pd.DataFrame) -> np.ndarray:
    """Extract T peak indices from PyHEARTS output DataFrame."""
    if output_df is None or 'T_global_center_idx' not in output_df.columns:
        return np.array([], dtype=int)
    
    t_peaks = output_df['T_global_center_idx'].dropna().values
    return np.array([int(p) for p in t_peaks if not np.isnan(p)], dtype=int)

def test_subject(subject: str, config: ProcessCycleConfig, config_name: str) -> Optional[Dict]:
    """Test a single subject with a given configuration."""
    print(f"\n  Testing {config_name}...", end=' ')
    
    # Load signal
    signal, sampling_rate = load_ecg_signal(subject)
    if signal is None:
        print("FAILED: Could not load signal")
        return None
    
    # Load manual annotations
    manual_beats = load_manual_annotations(subject, 'q1c', sampling_rate)
    if len(manual_beats) == 0:
        print("FAILED: No manual annotations")
        return None
    
    # Extract manual T peaks
    manual_t_peaks = np.array([
        beat['t_peak'] for beat in manual_beats 
        if beat['t_peak'] is not None
    ], dtype=int)
    
    if len(manual_t_peaks) == 0:
        print("FAILED: No manual T peaks")
        return None
    
    # Run PyHEARTS
    try:
        analyzer = PyHEARTS(
            sampling_rate=sampling_rate,
            verbose=False,
            cfg=config,
        )
        output_df, epochs_df = analyzer.analyze_ecg(signal)
        
        # Extract detected T peaks
        detected_t_peaks = extract_t_peaks_from_output(output_df)
        
        # Compute metrics
        metrics = compute_peak_metrics(
            detected_t_peaks,
            manual_t_peaks,
            sampling_rate,
            max_match_distance_ms=100.0,
        )
        
        print(f"MAD={metrics['mean_mad_ms']:.2f}ms, ±20ms={metrics['within_20ms_pct']:.1f}%, matched={metrics['n_matched']}/{metrics['n_manual']}")
        
        return {
            'subject': subject,
            'config': config_name,
            't_mean_mad_ms': metrics['mean_mad_ms'],
            't_within_20ms_pct': metrics['within_20ms_pct'],
            't_n_matched': metrics['n_matched'],
            't_n_detected': metrics['n_detected'],
            't_n_manual': metrics['n_manual'],
        }
    
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def main():
    # Try to load high-variability subjects from analysis
    variability_file = Path(__file__).parent / 'rr_variability_analysis.csv'
    if variability_file.exists():
        try:
            df = pd.read_csv(variability_file)
            high_var = df[(df['cv_rr_pct'] > 5.0) & (df['range_rr_ms'] > 200.0)]
            if len(high_var) > 0:
                recommended = high_var.sort_values('cv_rr_pct', ascending=False).head(10)
                TEST_SUBJECTS = list(recommended['subject'].values)
                print(f"Using high-variability subjects from analysis: {TEST_SUBJECTS}")
            else:
                top5 = df.sort_values('cv_rr_pct', ascending=False).head(5)
                TEST_SUBJECTS = list(top5['subject'].values)
                print(f"Using top 5 by CV: {TEST_SUBJECTS}")
        except Exception as e:
            print(f"Could not load variability analysis: {e}")
            print(f"Using default test subjects: {TEST_SUBJECTS}")
    
    print("="*80)
    print("T-Dynamic-Window Test on Variable RR Interval Subjects")
    print("="*80)
    print(f"\nTest subjects: {len(TEST_SUBJECTS)}")
    print(f"Subjects: {TEST_SUBJECTS}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create configurations
    baseline_config = ProcessCycleConfig(t_use_dynamic_window=False)
    experimental_config = ProcessCycleConfig(t_use_dynamic_window=True)
    
    all_results = []
    
    # Test each subject with both configurations
    for subject in TEST_SUBJECTS:
        print(f"\n{'#'*80}")
        print(f"Processing: {subject}")
        print(f"{'#'*80}")
        
        # Baseline
        baseline_result = test_subject(subject, baseline_config, "baseline")
        if baseline_result:
            all_results.append(baseline_result)
        
        # Experimental
        experimental_result = test_subject(subject, experimental_config, "t_dynamic_window")
        if experimental_result:
            all_results.append(experimental_result)
    
    # Create results DataFrame
    if len(all_results) == 0:
        print("\n\nNo results obtained.")
        return 1
    
    results_df = pd.DataFrame(all_results)
    
    # Save detailed results
    results_file = os.path.join(OUTPUT_DIR, 'variable_rr_test_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\n\nDetailed results saved to: {results_file}")
    
    # Compute summary and comparison
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    
    baseline_results = results_df[results_df['config'] == 'baseline']
    experimental_results = results_df[results_df['config'] == 't_dynamic_window']
    
    if len(baseline_results) > 0 and len(experimental_results) > 0:
        baseline_mad = baseline_results['t_mean_mad_ms'].mean()
        experimental_mad = experimental_results['t_mean_mad_ms'].mean()
        baseline_20ms = baseline_results['t_within_20ms_pct'].mean()
        experimental_20ms = experimental_results['t_within_20ms_pct'].mean()
        
        print("\n" + "="*80)
        print("COMPARISON: Experimental vs Baseline")
        print("="*80)
        print(f"\nT-Peak Mean MAD:")
        print(f"  Baseline:      {baseline_mad:.2f} ms")
        print(f"  Experimental:  {experimental_mad:.2f} ms")
        print(f"  Change:        {experimental_mad - baseline_mad:+.2f} ms ({'IMPROVEMENT' if experimental_mad < baseline_mad else 'REGRESSION'})")
        
        print(f"\nT-Peak Within ±20ms:")
        print(f"  Baseline:      {baseline_20ms:.1f}%")
        print(f"  Experimental:  {experimental_20ms:.1f}%")
        print(f"  Change:        {experimental_20ms - baseline_20ms:+.1f}% ({'IMPROVEMENT' if experimental_20ms > baseline_20ms else 'REGRESSION'})")
        
        # Per-subject comparison
        print("\n" + "="*80)
        print("PER-SUBJECT COMPARISON")
        print("="*80)
        
        comparison_data = []
        for subject in TEST_SUBJECTS:
            baseline_subj = baseline_results[baseline_results['subject'] == subject]
            experimental_subj = experimental_results[experimental_results['subject'] == subject]
            
            if len(baseline_subj) > 0 and len(experimental_subj) > 0:
                bl_mad = baseline_subj['t_mean_mad_ms'].iloc[0]
                exp_mad = experimental_subj['t_mean_mad_ms'].iloc[0]
                bl_20ms = baseline_subj['t_within_20ms_pct'].iloc[0]
                exp_20ms = experimental_subj['t_within_20ms_pct'].iloc[0]
                
                mad_change = exp_mad - bl_mad
                pct_change = exp_20ms - bl_20ms
                
                comparison_data.append({
                    'subject': subject,
                    'baseline_mad_ms': bl_mad,
                    'experimental_mad_ms': exp_mad,
                    'mad_change_ms': mad_change,
                    'baseline_20ms_pct': bl_20ms,
                    'experimental_20ms_pct': exp_20ms,
                    '20ms_change_pct': pct_change,
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print("\nPer-subject changes (negative MAD change = improvement):")
            print(comparison_df.to_string(index=False))
            
            comparison_file = os.path.join(OUTPUT_DIR, 'variable_rr_comparison.csv')
            comparison_df.to_csv(comparison_file, index=False)
            print(f"\nPer-subject comparison saved to: {comparison_file}")
            
            # Count improvements
            improved_mad = (comparison_df['mad_change_ms'] < 0).sum()
            improved_20ms = (comparison_df['20ms_change_pct'] > 0).sum()
            total = len(comparison_df)
            
            print(f"\nSummary:")
            print(f"  Subjects with improved MAD: {improved_mad}/{total} ({improved_mad/total*100:.1f}%)")
            print(f"  Subjects with improved ±20ms: {improved_20ms}/{total} ({improved_20ms/total*100:.1f}%)")
    
    print("\n" + "="*80)
    print("Test complete!")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

