#!/usr/bin/env python3
"""
Test Filtering Artifacts: Full-Signal vs Cycle-Segment Filtering

This script compares T-peak detection accuracy with:
1. Cycle-segment filtering (current PyHEARTS approach: filter each cycle segment individually)
2. Full-signal filtering (filter full signal first, then extract cycle segments)

The hypothesis is that filtering cycle segments individually may create edge artifacts
that affect T-peak detection accuracy, compared to filtering the full continuous signal.
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
OUTPUT_DIR = "/Users/morganfitzgerald/Documents/pyhearts/reviewer_response/gold_standard_comparison/filtering_artifacts_test"
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

def test_filtering_artifacts(subject: str) -> Optional[pd.DataFrame]:
    """Test full-signal filtering vs cycle-segment filtering."""
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
    
    # Prepare full-signal detrending and filtering
    # First, detrend the full signal (simulating what epoch.py does cycle-by-cycle)
    # For full-signal approach, we'll detrend the full signal with a sliding window
    # But to keep it simple, let's detrend each cycle segment as PyHEARTS does,
    # then concatenate and filter the full detrended signal
    
    # Build full detrended signal from cycles (simulating cycle-by-cycle detrending)
    all_cycle_segments = []
    all_cycle_starts = []
    all_cycle_ends = []
    
    for cycle_id, cycle_group in cycles:
        cycle_group = cycle_group.sort_values('index')
        cycle_global_indices = cycle_group['index'].values
        cycle_start_global = int(cycle_global_indices[0])
        cycle_end_global = int(cycle_global_indices[-1])
        cycle_signal = signal[cycle_start_global:cycle_end_global+1]
        
        # Detrend this cycle segment (like epoch.py does)
        cycle_detrended = scipy_detrend(cycle_signal, type='linear')
        
        all_cycle_segments.append((cycle_start_global, cycle_end_global, cycle_detrended))
        all_cycle_starts.append(cycle_start_global)
        all_cycle_ends.append(cycle_end_global)
    
    # For full-signal filtering approach:
    # Build a full detrended signal by concatenating detrended cycles
    # (This simulates having detrended the full signal)
    if len(all_cycle_segments) == 0:
        print(f"  No valid cycles")
        return None
    
    # Sort cycles by start position
    all_cycle_segments.sort(key=lambda x: x[0])
    
    # Create full detrended signal (pad gaps with zeros or interpolate)
    max_end = max(end for _, end, _ in all_cycle_segments)
    full_detrended_signal = np.zeros(max_end + 1)
    
    for cycle_start, cycle_end, cycle_detrended in all_cycle_segments:
        full_detrended_signal[cycle_start:cycle_end+1] = cycle_detrended
    
    # Compute filtered derivative on FULL signal
    derivative_full_signal = compute_filtered_derivative(
        full_detrended_signal, 
        sampling_rate, 
        lowpass_cutoff=40.0
    )
    
    # For each manual beat, test detection with both approaches
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
        
        # Find corresponding cycle
        cycle_num = None
        cycle_data = None
        cycle_segment_info = None
        
        for cycle_id, cycle_group in cycles:
            cycle_group = cycle_group.sort_values('index')
            cycle_global_start = int(cycle_group['index'].min())
            cycle_global_end = int(cycle_group['index'].max())
            if cycle_global_start <= closest_r <= cycle_global_end:
                cycle_num = cycle_id
                cycle_data = cycle_group
                # Find matching segment info
                for start, end, detrended in all_cycle_segments:
                    if start == cycle_global_start and end == cycle_global_end:
                        cycle_segment_info = (start, end, detrended)
                        break
                break
        
        if cycle_data is None or cycle_segment_info is None:
            continue
        
        cycle_start_global, cycle_end_global, cycle_detrended = cycle_segment_info
        
        # Convert global indices to cycle-relative
        r_cycle_idx = closest_r - cycle_start_global
        t_cycle_idx_manual = man_t - cycle_start_global
        
        # Check bounds
        if not (0 <= r_cycle_idx < len(cycle_detrended) and 0 <= t_cycle_idx_manual < len(cycle_detrended)):
            continue
        
        # T-detection search window (same for both)
        t_start_offset_ms = 100.0
        t_end_offset_ms = 450.0
        t_start_idx = r_cycle_idx + int(round(t_start_offset_ms * sampling_rate / 1000.0))
        t_end_idx = r_cycle_idx + int(round(t_end_offset_ms * sampling_rate / 1000.0))
        t_start_idx = max(0, min(t_start_idx, len(cycle_detrended) - 2))
        t_end_idx = max(t_start_idx + 10, min(t_end_idx, len(cycle_detrended) - 1))
        
        if t_end_idx - t_start_idx < 10:
            continue
        
        # TEST 1: Cycle-segment filtering (current PyHEARTS approach)
        # Filter derivative on cycle segment only
        derivative_cycle_segment = compute_filtered_derivative(
            cycle_detrended, 
            sampling_rate, 
            lowpass_cutoff=40.0
        )
        
        t_peak_cycle_segment, t_start_cs, t_end_cs, t_amp_cs, morph_cs = detect_t_wave_derivative_based(
            signal=cycle_detrended,
            derivative=derivative_cycle_segment,
            search_start=t_start_idx,
            search_end=t_end_idx,
            sampling_rate=sampling_rate,
            verbose=False,
            r_peak_idx=r_cycle_idx,
            r_peak_value=cycle_detrended[r_cycle_idx] if 0 <= r_cycle_idx < len(cycle_detrended) else None,
        )
        
        # TEST 2: Full-signal filtering (experimental approach)
        # Extract derivative from full-signal-filtered derivative
        derivative_cycle_from_full = derivative_full_signal[cycle_start_global:cycle_end_global+1]
        
        # Ensure length matches
        if len(derivative_cycle_from_full) != len(cycle_detrended):
            # Trim or pad if needed (shouldn't happen, but handle gracefully)
            min_len = min(len(derivative_cycle_from_full), len(cycle_detrended))
            derivative_cycle_from_full = derivative_cycle_from_full[:min_len]
            cycle_detrended_for_full = cycle_detrended[:min_len]
        else:
            cycle_detrended_for_full = cycle_detrended
        
        # Update indices if we had to trim
        if len(derivative_cycle_from_full) < len(cycle_detrended):
            t_start_idx_adj = min(t_start_idx, len(derivative_cycle_from_full) - 2)
            t_end_idx_adj = min(t_end_idx, len(derivative_cycle_from_full) - 1)
            r_cycle_idx_adj = min(r_cycle_idx, len(derivative_cycle_from_full) - 1)
        else:
            t_start_idx_adj = t_start_idx
            t_end_idx_adj = t_end_idx
            r_cycle_idx_adj = r_cycle_idx
            t_cycle_idx_manual_adj = t_cycle_idx_manual
            if t_cycle_idx_manual_adj >= len(cycle_detrended_for_full):
                continue
        
        t_peak_full_signal, t_start_fs, t_end_fs, t_amp_fs, morph_fs = detect_t_wave_derivative_based(
            signal=cycle_detrended_for_full,
            derivative=derivative_cycle_from_full,
            search_start=t_start_idx_adj,
            search_end=t_end_idx_adj,
            sampling_rate=sampling_rate,
            verbose=False,
            r_peak_idx=r_cycle_idx_adj,
            r_peak_value=cycle_detrended_for_full[r_cycle_idx_adj] if 0 <= r_cycle_idx_adj < len(cycle_detrended_for_full) else None,
        )
        
        # Calculate errors (use adjusted manual index if we trimmed)
        error_cycle_segment_ms = np.nan
        error_full_signal_ms = np.nan
        
        if t_peak_cycle_segment is not None:
            error_cycle_segment_ms = abs(t_peak_cycle_segment - t_cycle_idx_manual) * 1000.0 / sampling_rate
        
        if t_peak_full_signal is not None:
            t_cycle_idx_manual_adj = min(t_cycle_idx_manual, len(cycle_detrended_for_full) - 1)
            error_full_signal_ms = abs(t_peak_full_signal - t_cycle_idx_manual_adj) * 1000.0 / sampling_rate
        
        # Check if manual T peak is in search window
        rt_interval_ms = (t_cycle_idx_manual - r_cycle_idx) * 1000.0 / sampling_rate
        in_window = t_start_offset_ms <= rt_interval_ms <= t_end_offset_ms
        
        # Calculate signal characteristics
        cycle_std = np.std(cycle_detrended)
        
        # Calculate derivative difference magnitude (at edges)
        if len(derivative_cycle_segment) == len(derivative_cycle_from_full):
            # Compare derivatives (especially at edges)
            edge_samples = min(50, len(derivative_cycle_segment) // 10)  # First/last 50 samples or 10% of signal
            edge_diff_start = np.std(derivative_cycle_segment[:edge_samples] - derivative_cycle_from_full[:edge_samples])
            edge_diff_end = np.std(derivative_cycle_segment[-edge_samples:] - derivative_cycle_from_full[-edge_samples:])
            center_diff = np.std(derivative_cycle_segment[edge_samples:-edge_samples] - derivative_cycle_from_full[edge_samples:-edge_samples]) if len(derivative_cycle_segment) > 2 * edge_samples else 0.0
            total_diff = np.std(derivative_cycle_segment - derivative_cycle_from_full)
        else:
            edge_diff_start = edge_diff_end = center_diff = total_diff = np.nan
        
        results.append({
            'subject': subject,
            'manual_t_cycle_idx': t_cycle_idx_manual,
            'rt_interval_ms': rt_interval_ms,
            't_detected_cycle_segment': t_peak_cycle_segment is not None,
            't_detected_full_signal': t_peak_full_signal is not None,
            'error_cycle_segment_ms': error_cycle_segment_ms,
            'error_full_signal_ms': error_full_signal_ms,
            'error_diff_ms': error_full_signal_ms - error_cycle_segment_ms if (not np.isnan(error_cycle_segment_ms) and not np.isnan(error_full_signal_ms)) else np.nan,
            'in_window': in_window,
            'cycle_std': cycle_std,
            'edge_diff_start': edge_diff_start,
            'edge_diff_end': edge_diff_end,
            'center_diff': center_diff,
            'total_derivative_diff': total_diff,
            't_peak_cycle_segment': t_peak_cycle_segment if t_peak_cycle_segment is not None else np.nan,
            't_peak_full_signal': t_peak_full_signal if t_peak_full_signal is not None else np.nan,
        })
    
    if len(results) == 0:
        print(f"  No analyzable beats")
        return None
    
    df = pd.DataFrame(results)
    print(f"  Analyzed {len(df)} beats")
    
    # Quick stats
    detected_cs = df['t_detected_cycle_segment'].sum()
    detected_fs = df['t_detected_full_signal'].sum()
    print(f"  Detected with cycle-segment filtering: {detected_cs}/{len(df)} ({detected_cs/len(df)*100:.1f}%)")
    print(f"  Detected with full-signal filtering: {detected_fs}/{len(df)} ({detected_fs/len(df)*100:.1f}%)")
    
    if detected_cs > 0 and detected_fs > 0:
        errors_cs = df[df['t_detected_cycle_segment']]['error_cycle_segment_ms']
        errors_fs = df[df['t_detected_full_signal']]['error_full_signal_ms']
        print(f"  Mean error (cycle-segment): {errors_cs.mean():.2f} ms")
        print(f"  Mean error (full-signal): {errors_fs.mean():.2f} ms")
    
    return df

def main():
    print("="*80)
    print("Testing Filtering Artifacts: Full-Signal vs Cycle-Segment Filtering")
    print("="*80)
    print(f"\nTest subjects: {len(TEST_SUBJECTS)}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    all_results = []
    
    for subject in TEST_SUBJECTS:
        result_df = test_filtering_artifacts(subject)
        if result_df is not None:
            all_results.append(result_df)
    
    if len(all_results) == 0:
        print("\n\nNo results obtained.")
        return 1
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save detailed results
    detailed_file = os.path.join(OUTPUT_DIR, 'filtering_artifacts_detailed.csv')
    combined_df.to_csv(detailed_file, index=False)
    print(f"\n\nDetailed results saved to: {detailed_file}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY: FULL-SIGNAL vs CYCLE-SEGMENT FILTERING")
    print("="*80)
    
    total_beats = len(combined_df)
    print(f"\nTotal beats analyzed: {total_beats}")
    
    # Detection rates
    detected_cs = combined_df['t_detected_cycle_segment'].sum()
    detected_fs = combined_df['t_detected_full_signal'].sum()
    print(f"\nDetection Rates:")
    print(f"  Cycle-segment filtering (current): {detected_cs}/{total_beats} ({detected_cs/total_beats*100:.1f}%)")
    print(f"  Full-signal filtering (experimental): {detected_fs}/{total_beats} ({detected_fs/total_beats*100:.1f}%)")
    
    # Error statistics (only for detected peaks)
    errors_cs = combined_df[combined_df['t_detected_cycle_segment']]['error_cycle_segment_ms']
    errors_fs = combined_df[combined_df['t_detected_full_signal']]['error_full_signal_ms']
    
    if len(errors_cs) > 0 and len(errors_fs) > 0:
        print(f"\nError Statistics (for detected peaks):")
        print(f"  Cycle-segment filtering:")
        print(f"    Mean error: {errors_cs.mean():.2f} ms")
        print(f"    Median error: {errors_cs.median():.2f} ms")
        print(f"    Std error: {errors_cs.std():.2f} ms")
        
        print(f"  Full-signal filtering:")
        print(f"    Mean error: {errors_fs.mean():.2f} ms")
        print(f"    Median error: {errors_fs.median():.2f} ms")
        print(f"    Std error: {errors_fs.std():.2f} ms")
        
        # Compare errors for beats detected by both
        both_detected = combined_df[combined_df['t_detected_cycle_segment'] & combined_df['t_detected_full_signal']]
        if len(both_detected) > 0:
            error_diff = both_detected['error_full_signal_ms'] - both_detected['error_cycle_segment_ms']
            print(f"\nComparison (beats detected by both methods):")
            print(f"  Beats detected by both: {len(both_detected)}/{total_beats}")
            print(f"  Mean error difference (full-signal - cycle-segment): {error_diff.mean():.2f} ms")
            print(f"  Median error difference: {error_diff.median():.2f} ms")
            print(f"  Cycle-segment better (negative diff): {(error_diff < 0).sum()}/{len(error_diff)} ({(error_diff < 0).sum()/len(error_diff)*100:.1f}%)")
            print(f"  Full-signal better (positive diff): {(error_diff > 0).sum()}/{len(error_diff)} ({(error_diff > 0).sum()/len(error_diff)*100:.1f}%)")
            
            # Statistical test (paired t-test)
            from scipy import stats
            t_stat, p_value = stats.ttest_rel(both_detected['error_cycle_segment_ms'], both_detected['error_full_signal_ms'])
            print(f"\nStatistical test (paired t-test):")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"  -> Statistically significant difference (p < 0.05)")
            else:
                print(f"  -> No statistically significant difference (p >= 0.05)")
    
    # Derivative difference analysis
    if 'edge_diff_start' in combined_df.columns:
        edge_diffs_start = combined_df['edge_diff_start'].dropna()
        edge_diffs_end = combined_df['edge_diff_end'].dropna()
        center_diffs = combined_df['center_diff'].dropna()
        total_diffs = combined_df['total_derivative_diff'].dropna()
        
        if len(edge_diffs_start) > 0:
            print(f"\nDerivative Difference Analysis:")
            print(f"  Start edge difference (std): Mean={edge_diffs_start.mean():.6f}, Max={edge_diffs_start.max():.6f}")
            print(f"  End edge difference (std): Mean={edge_diffs_end.mean():.6f}, Max={edge_diffs_end.max():.6f}")
            print(f"  Center difference (std): Mean={center_diffs.mean():.6f}, Max={center_diffs.max():.6f}")
            print(f"  Total difference (std): Mean={total_diffs.mean():.6f}, Max={total_diffs.max():.6f}")
            
            # Check if edge differences are larger than center differences
            if len(center_diffs) > 0:
                edge_avg = (edge_diffs_start.mean() + edge_diffs_end.mean()) / 2
                center_avg = center_diffs.mean()
                print(f"  Edge vs Center: Edge differences are {edge_avg/center_avg if center_avg > 0 else 'N/A'}x larger than center")
    
    # Save summary
    summary_file = os.path.join(OUTPUT_DIR, 'filtering_artifacts_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Filtering Artifacts Test Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total beats analyzed: {total_beats}\n\n")
        f.write(f"Detection rates:\n")
        f.write(f"  Cycle-segment filtering: {detected_cs}/{total_beats} ({detected_cs/total_beats*100:.1f}%)\n")
        f.write(f"  Full-signal filtering: {detected_fs}/{total_beats} ({detected_fs/total_beats*100:.1f}%)\n")
        if len(errors_cs) > 0 and len(errors_fs) > 0:
            f.write(f"\nMean errors:\n")
            f.write(f"  Cycle-segment filtering: {errors_cs.mean():.2f} ms\n")
            f.write(f"  Full-signal filtering: {errors_fs.mean():.2f} ms\n")
            if len(both_detected) > 0:
                error_diff = both_detected['error_full_signal_ms'] - both_detected['error_cycle_segment_ms']
                f.write(f"  Mean difference (full-signal - cycle-segment): {error_diff.mean():.2f} ms\n")
    print(f"\nSummary saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

