#!/usr/bin/env python3
"""
Test Single vs Double Detrending for T-Peak Detection

This script compares T-peak detection accuracy with:
1. Double detrending (current PyHEARTS approach: detrend in epoch.py + detrend in processcycle.py)
2. Single detrending (detrend only in epoch.py, skip second detrending in processcycle.py)

The test isolates the detrending effect by manually applying the T-detection algorithm
with both approaches and comparing results to manual annotations.
"""

import os
import sys
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.signal import detrend as scipy_detrend
import warnings
warnings.filterwarnings('ignore')

# Add pyhearts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyhearts import PyHEARTS, ProcessCycleConfig
from pyhearts.processing.derivative_t_detection import (
    compute_filtered_derivative,
    detect_t_wave_derivative_based,
)
from pyhearts.processing.detrend import detrend_signal

# Paths
MANUAL_ANNOTATIONS_DIR = "/Users/morganfitzgerald/Documents/pyhearts/data/qtdb/1.0.0"
OUTPUT_DIR = "/Users/morganfitzgerald/Documents/pyhearts/reviewer_response/gold_standard_comparison/single_vs_double_detrending_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_SUBJECTS = [
    'sel853', 'sel233', 'sel45', 'sel46', 'sel811', 'sel883',
    'sel116', 'sel213', 'sel100', 'sel16539',
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
            't_peak': None,
        }
        
        for j in range(r_idx - 1, max(-1, r_idx - 20), -1):
            if ann.symbol[j] == 'p':
                beat['p_peak'] = ann.sample[j]
                break
        
        for j in range(r_idx + 1, min(len(ann.symbol), r_idx + 20)):
            if ann.symbol[j] in ['t', 'T']:
                beat['t_peak'] = ann.sample[j]
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

def test_detrending_effect(subject: str) -> Optional[pd.DataFrame]:
    """Test single vs double detrending effect on T-peak detection."""
    print(f"\nAnalyzing {subject}...")
    
    # Load signal
    signal, sampling_rate = load_ecg_signal(subject)
    if signal is None:
        print(f"  Could not load signal")
        return None
    
    # Load manual annotations
    manual_beats = load_manual_annotations(subject, 'q1c', sampling_rate)
    if len(manual_beats) == 0:
        print(f"  No manual annotations")
        return None
    
    # Run PyHEARTS to get R peaks and cycle structure
    try:
        analyzer = PyHEARTS(
            sampling_rate=sampling_rate,
            verbose=False,
        )
        output_df, epochs_df = analyzer.analyze_ecg(signal)
        
        if output_df is None or len(output_df) == 0:
            print(f"  No PyHEARTS results")
            return None
        
        # Get R peaks
        r_peaks_global = output_df['R_global_center_idx'].dropna().values
        r_peaks_global = np.array([int(r) for r in r_peaks_global if not np.isnan(r)], dtype=int)
        
    except Exception as e:
        print(f"  Error running PyHEARTS: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    results = []
    cfg = ProcessCycleConfig()
    
    # Get cycle segmentation info from epochs_df
    if epochs_df is None or len(epochs_df) == 0:
        print(f"  No epochs data")
        return None
    
    # Group epochs by cycle
    cycles = epochs_df.groupby('cycle')
    
    # For each manual beat, test detection with single vs double detrending
    for man_beat in manual_beats:
        if man_beat['r_peak'] is None or man_beat['t_peak'] is None:
            continue
        
        man_r = man_beat['r_peak']
        man_t = man_beat['t_peak']
        
        # Find closest PyHEARTS R peak
        distances = np.abs(r_peaks_global - man_r)
        closest_r_idx = np.argmin(distances)
        closest_r = r_peaks_global[closest_r_idx]
        
        # Only analyze if R peaks are close (within 50ms)
        r_distance_ms = abs(closest_r - man_r) * 1000.0 / sampling_rate
        if r_distance_ms > 50.0:
            continue
        
        # Find corresponding cycle in epochs_df
        cycle_num = None
        cycle_data = None
        for cycle_id, cycle_group in cycles:
            cycle_global_start = cycle_group['index'].min()
            cycle_global_end = cycle_group['index'].max()
            if cycle_global_start <= closest_r <= cycle_global_end:
                cycle_num = cycle_id
                cycle_data = cycle_group.sort_values('index')
                break
        
        if cycle_data is None or len(cycle_data) == 0:
            continue
        
        # Extract cycle signal (original, not yet detrended)
        cycle_global_indices = cycle_data['index'].values
        cycle_start_global = int(cycle_global_indices[0])
        cycle_end_global = int(cycle_global_indices[-1])
        cycle_signal_original = signal[cycle_start_global:cycle_end_global+1]
        
        if len(cycle_signal_original) < 100:  # Need minimum length
            continue
        
        # Convert global indices to cycle-relative
        r_cycle_idx = closest_r - cycle_start_global
        t_cycle_idx_manual = man_t - cycle_start_global
        
        # Check bounds
        if not (0 <= r_cycle_idx < len(cycle_signal_original) and 0 <= t_cycle_idx_manual < len(cycle_signal_original)):
            continue
        
        # TEST 1: Single detrending (only linear detrend like epoch.py does)
        cycle_single_detrended = scipy_detrend(cycle_signal_original, type='linear')
        
        # TEST 2: Double detrending (linear detrend + PyHEARTS detrend_signal)
        cycle_linear_detrended = scipy_detrend(cycle_signal_original, type='linear')
        xs_rel_idxs = np.arange(len(cycle_linear_detrended))
        cycle_double_detrended, _ = detrend_signal(
            xs_rel_idxs, 
            cycle_linear_detrended, 
            sampling_rate=sampling_rate, 
            window_ms=cfg.detrend_window_ms,
            cycle=0,
            plot=False
        )
        
        # T-detection search window (same for both)
        t_start_offset_ms = 100.0
        t_end_offset_ms = 450.0
        t_start_idx = r_cycle_idx + int(round(t_start_offset_ms * sampling_rate / 1000.0))
        t_end_idx = r_cycle_idx + int(round(t_end_offset_ms * sampling_rate / 1000.0))
        t_start_idx = max(0, min(t_start_idx, len(cycle_signal_original) - 2))
        t_end_idx = max(t_start_idx + 10, min(t_end_idx, len(cycle_signal_original) - 1))
        
        if t_end_idx - t_start_idx < 10:
            continue
        
        # Test T-detection on single-detrended signal
        derivative_single = compute_filtered_derivative(cycle_single_detrended, sampling_rate, lowpass_cutoff=40.0)
        t_peak_single, t_start_single, t_end_single, t_amp_single, morph_single = detect_t_wave_derivative_based(
            signal=cycle_single_detrended,
            derivative=derivative_single,
            search_start=t_start_idx,
            search_end=t_end_idx,
            sampling_rate=sampling_rate,
            verbose=False,
            r_peak_idx=r_cycle_idx,
            r_peak_value=cycle_single_detrended[r_cycle_idx] if 0 <= r_cycle_idx < len(cycle_single_detrended) else None,
        )
        
        # Test T-detection on double-detrended signal
        derivative_double = compute_filtered_derivative(cycle_double_detrended, sampling_rate, lowpass_cutoff=40.0)
        t_peak_double, t_start_double, t_end_double, t_amp_double, morph_double = detect_t_wave_derivative_based(
            signal=cycle_double_detrended,
            derivative=derivative_double,
            search_start=t_start_idx,
            search_end=t_end_idx,
            sampling_rate=sampling_rate,
            verbose=False,
            r_peak_idx=r_cycle_idx,
            r_peak_value=cycle_double_detrended[r_cycle_idx] if 0 <= r_cycle_idx < len(cycle_double_detrended) else None,
        )
        
        # Calculate errors
        error_single_ms = np.nan
        error_double_ms = np.nan
        
        if t_peak_single is not None:
            error_single_ms = abs(t_peak_single - t_cycle_idx_manual) * 1000.0 / sampling_rate
        
        if t_peak_double is not None:
            error_double_ms = abs(t_peak_double - t_cycle_idx_manual) * 1000.0 / sampling_rate
        
        # Check if manual T peak is in search window
        rt_interval_ms = (t_cycle_idx_manual - r_cycle_idx) * 1000.0 / sampling_rate
        in_window = t_start_offset_ms <= rt_interval_ms <= t_end_offset_ms
        
        # Calculate signal characteristics
        cycle_std = np.std(cycle_signal_original)
        baseline_drift = cycle_signal_original[-1] - cycle_signal_original[0]
        
        # Calculate detrending difference magnitude
        detrend_diff = np.std(cycle_double_detrended - cycle_single_detrended)
        
        results.append({
            'subject': subject,
            'manual_t_cycle_idx': t_cycle_idx_manual,
            'rt_interval_ms': rt_interval_ms,
            't_detected_single': t_peak_single is not None,
            't_detected_double': t_peak_double is not None,
            'error_single_ms': error_single_ms,
            'error_double_ms': error_double_ms,
            'error_diff_ms': error_double_ms - error_single_ms if (not np.isnan(error_single_ms) and not np.isnan(error_double_ms)) else np.nan,
            'in_window': in_window,
            'cycle_std': cycle_std,
            'baseline_drift': baseline_drift,
            'detrend_diff': detrend_diff,
            't_peak_single': t_peak_single if t_peak_single is not None else np.nan,
            't_peak_double': t_peak_double if t_peak_double is not None else np.nan,
        })
    
    if len(results) == 0:
        print(f"  No analyzable beats")
        return None
    
    df = pd.DataFrame(results)
    print(f"  Analyzed {len(df)} beats")
    
    # Quick stats
    detected_single = df['t_detected_single'].sum()
    detected_double = df['t_detected_detrended'].sum() if 't_detected_detrended' in df.columns else df['t_detected_double'].sum()
    print(f"  Detected with single detrending: {detected_single}/{len(df)} ({detected_single/len(df)*100:.1f}%)")
    print(f"  Detected with double detrending: {detected_double}/{len(df)} ({detected_double/len(df)*100:.1f}%)")
    
    if detected_single > 0 and detected_double > 0:
        errors_single = df[df['t_detected_single']]['error_single_ms']
        errors_double = df[df['t_detected_double']]['error_double_ms']
        print(f"  Mean error (single): {errors_single.mean():.2f} ms")
        print(f"  Mean error (double): {errors_double.mean():.2f} ms")
    
    return df

def main():
    print("="*80)
    print("Testing Single vs Double Detrending for T-Peak Detection")
    print("="*80)
    print(f"\nTest subjects: {len(TEST_SUBJECTS)}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    all_results = []
    
    for subject in TEST_SUBJECTS:
        result_df = test_detrending_effect(subject)
        if result_df is not None:
            all_results.append(result_df)
    
    if len(all_results) == 0:
        print("\n\nNo results obtained.")
        return 1
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save detailed results
    detailed_file = os.path.join(OUTPUT_DIR, 'single_vs_double_detrending_detailed.csv')
    combined_df.to_csv(detailed_file, index=False)
    print(f"\n\nDetailed results saved to: {detailed_file}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY: SINGLE vs DOUBLE DETRENDING")
    print("="*80)
    
    total_beats = len(combined_df)
    print(f"\nTotal beats analyzed: {total_beats}")
    
    # Detection rates
    detected_single = combined_df['t_detected_single'].sum()
    detected_double = combined_df['t_detected_double'].sum()
    print(f"\nDetection Rates:")
    print(f"  Single detrending: {detected_single}/{total_beats} ({detected_single/total_beats*100:.1f}%)")
    print(f"  Double detrending: {detected_double}/{total_beats} ({detected_double/total_beats*100:.1f}%)")
    
    # Error statistics (only for detected peaks)
    errors_single = combined_df[combined_df['t_detected_single']]['error_single_ms']
    errors_double = combined_df[combined_df['t_detected_double']]['error_double_ms']
    
    if len(errors_single) > 0 and len(errors_double) > 0:
        print(f"\nError Statistics (for detected peaks):")
        print(f"  Single detrending:")
        print(f"    Mean error: {errors_single.mean():.2f} ms")
        print(f"    Median error: {errors_single.median():.2f} ms")
        print(f"    Std error: {errors_single.std():.2f} ms")
        
        print(f"  Double detrending:")
        print(f"    Mean error: {errors_double.mean():.2f} ms")
        print(f"    Median error: {errors_double.median():.2f} ms")
        print(f"    Std error: {errors_double.std():.2f} ms")
        
        # Compare errors for beats detected by both
        both_detected = combined_df[combined_df['t_detected_single'] & combined_df['t_detected_double']]
        if len(both_detected) > 0:
            error_diff = both_detected['error_double_ms'] - both_detected['error_single_ms']
            print(f"\nComparison (beats detected by both methods):")
            print(f"  Beats detected by both: {len(both_detected)}/{total_beats}")
            print(f"  Mean error difference (double - single): {error_diff.mean():.2f} ms")
            print(f"  Median error difference: {error_diff.median():.2f} ms")
            print(f"  Single better (negative diff): {(error_diff < 0).sum()}/{len(error_diff)} ({(error_diff < 0).sum()/len(error_diff)*100:.1f}%)")
            print(f"  Double better (positive diff): {(error_diff > 0).sum()}/{len(error_diff)} ({(error_diff > 0).sum()/len(error_diff)*100:.1f}%)")
            
            # Statistical test (paired t-test)
            from scipy import stats
            t_stat, p_value = stats.ttest_rel(both_detected['error_single_ms'], both_detected['error_double_ms'])
            print(f"\nStatistical test (paired t-test):")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"  -> Statistically significant difference (p < 0.05)")
            else:
                print(f"  -> No statistically significant difference (p >= 0.05)")
    
    # Detrending difference magnitude
    if 'detrend_diff' in combined_df.columns:
        detrend_diffs = combined_df['detrend_diff'].dropna()
        if len(detrend_diffs) > 0:
            print(f"\nDetrending Difference Magnitude:")
            print(f"  Mean std diff between single/double detrended signals: {detrend_diffs.mean():.6f}")
            print(f"  Max std diff: {detrend_diffs.max():.6f}")
            print(f"  Min std diff: {detrend_diffs.min():.6f}")
    
    # Save summary
    summary_file = os.path.join(OUTPUT_DIR, 'single_vs_double_detrending_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Single vs Double Detrending Test Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total beats analyzed: {total_beats}\n\n")
        f.write(f"Detection rates:\n")
        f.write(f"  Single detrending: {detected_single}/{total_beats} ({detected_single/total_beats*100:.1f}%)\n")
        f.write(f"  Double detrending: {detected_double}/{total_beats} ({detected_double/total_beats*100:.1f}%)\n")
        if len(errors_single) > 0 and len(errors_double) > 0:
            f.write(f"\nMean errors:\n")
            f.write(f"  Single detrending: {errors_single.mean():.2f} ms\n")
            f.write(f"  Double detrending: {errors_double.mean():.2f} ms\n")
            if len(both_detected) > 0:
                error_diff = both_detected['error_double_ms'] - both_detected['error_single_ms']
                f.write(f"  Mean difference (double - single): {error_diff.mean():.2f} ms\n")
    print(f"\nSummary saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

