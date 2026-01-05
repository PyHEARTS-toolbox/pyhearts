#!/usr/bin/env python3
"""
Test S Wave End + 20ms Adjustment for T-Peak Detection

This script tests if using S wave end + 20ms (if later than 100ms after R)
improves T-peak detection accuracy, matching ECGpuwave's approach.
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

# Paths
MANUAL_ANNOTATIONS_DIR = "/Users/morganfitzgerald/Documents/pyhearts/data/qtdb/1.0.0"
OUTPUT_DIR = "/Users/morganfitzgerald/Documents/pyhearts/reviewer_response/gold_standard_comparison/s_wave_end_adjustment_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test subjects (same as previous tests)
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

def analyze_subject(subject: str) -> Optional[pd.DataFrame]:
    """Analyze a single subject."""
    print(f"\nAnalyzing {subject}...")
    
    # Load signal
    signal, sampling_rate = load_ecg_signal(subject)
    if signal is None:
        return None
    
    # Load manual annotations
    manual_beats = load_manual_annotations(subject, 'q1c', sampling_rate)
    if len(manual_beats) == 0:
        return None
    
    # Run PyHEARTS (with S wave end adjustment - already implemented in code)
    try:
        analyzer = PyHEARTS(
            sampling_rate=sampling_rate,
            verbose=False,
        )
        output_df, epochs_df = analyzer.analyze_ecg(signal)
        
        if output_df is None or len(output_df) == 0:
            return None
        
        r_peaks_global = output_df['R_global_center_idx'].dropna().values
        r_peaks_global = np.array([int(r) for r in r_peaks_global if not np.isnan(r)], dtype=int)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None
    
    # Match manual beats to PyHEARTS results
    results = []
    
    for man_beat in manual_beats:
        if man_beat['r_peak'] is None or man_beat['t_peak'] is None:
            continue
        
        man_r = man_beat['r_peak']
        man_t = man_beat['t_peak']
        
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
        
        # Find corresponding T peak in PyHEARTS results
        detected_t = None
        error_ms = np.nan
        
        for idx, row in output_df.iterrows():
            if pd.notna(row.get('R_global_center_idx')) and int(row['R_global_center_idx']) == closest_r:
                if pd.notna(row.get('T_global_center_idx')):
                    detected_t = int(row['T_global_center_idx'])
                    error_ms = abs(detected_t - man_t) * 1000.0 / sampling_rate
                break
        
        results.append({
            'subject': subject,
            'manual_r': man_r,
            'manual_t': man_t,
            'pyhearts_r': closest_r,
            'pyhearts_t': detected_t if detected_t is not None else np.nan,
            'error_ms': error_ms,
            't_detected': detected_t is not None,
        })
    
    if len(results) == 0:
        print(f"  No matched beats")
        return None
    
    df = pd.DataFrame(results)
    
    # Calculate statistics
    detected = df['t_detected'].sum()
    errors = df[df['t_detected']]['error_ms']
    
    print(f"  Analyzed {len(df)} beats")
    print(f"  T peaks detected: {detected}/{len(df)} ({detected/len(df)*100:.1f}%)")
    if len(errors) > 0:
        print(f"  Mean error: {errors.mean():.2f} ms")
        print(f"  Median error: {errors.median():.2f} ms")
        print(f"  % within ±20ms: {(errors <= 20).sum()/len(errors)*100:.1f}%")
    
    return df

def main():
    print("="*80)
    print("T-Peak Detection Test: S Wave End + 20ms Adjustment")
    print("="*80)
    print(f"\nTest subjects: {len(TEST_SUBJECTS)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nThis test uses S wave end + 20ms adjustment (like ECGpuwave)")
    print("to determine T search start time.")
    
    all_results = []
    
    for subject in TEST_SUBJECTS:
        result_df = analyze_subject(subject)
        if result_df is not None:
            all_results.append(result_df)
    
    if len(all_results) == 0:
        print("\n\nNo results obtained.")
        return 1
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save detailed results
    detailed_file = os.path.join(OUTPUT_DIR, 's_wave_end_adjustment_results.csv')
    combined_df.to_csv(detailed_file, index=False)
    print(f"\n\nDetailed results saved to: {detailed_file}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY: T-PEAK DETECTION WITH S WAVE END ADJUSTMENT")
    print("="*80)
    
    total_beats = len(combined_df)
    detected = combined_df['t_detected'].sum()
    
    print(f"\nTotal beats analyzed: {total_beats}")
    print(f"T peaks detected: {detected}/{total_beats} ({detected/total_beats*100:.1f}%)")
    
    if detected > 0:
        errors = combined_df[combined_df['t_detected']]['error_ms']
        print(f"\nError Statistics:")
        print(f"  Mean MAD: {errors.mean():.2f} ms")
        print(f"  Median MAD: {errors.median():.2f} ms")
        print(f"  Std MAD: {errors.std():.2f} ms")
        print(f"  % within ±20ms: {(errors <= 20).sum()/len(errors)*100:.1f}%")
        print(f"  % within ±50ms: {(errors <= 50).sum()/len(errors)*100:.1f}%")
        
        # Calculate signed errors for bias analysis
        combined_df['error_signed_ms'] = (combined_df['pyhearts_t'] - combined_df['manual_t']) * (1000.0 / 250.0)
        errors_signed = combined_df[combined_df['t_detected']]['error_signed_ms']
        print(f"\nBias Analysis:")
        print(f"  Mean signed error: {errors_signed.mean():.2f} ms")
        print(f"  Early errors (<0): {(errors_signed < 0).sum()} ({(errors_signed < 0).sum()/len(errors_signed)*100:.1f}%)")
        print(f"  Late errors (>0): {(errors_signed > 0).sum()} ({(errors_signed > 0).sum()/len(errors_signed)*100:.1f}%)")
        
        # ST segment detection analysis
        combined_df['rt_interval_ms'] = (combined_df['pyhearts_t'] - combined_df['manual_r']) * (1000.0 / 250.0)
        st_segment_cases = combined_df[(combined_df['t_detected']) & (combined_df['rt_interval_ms'] < 150.0)]
        print(f"\nST Segment Detection:")
        print(f"  Cases with RT < 150ms: {len(st_segment_cases)} ({len(st_segment_cases)/len(errors)*100:.1f}%)")
        if len(st_segment_cases) > 0:
            print(f"  Mean error for ST segment cases: {st_segment_cases['error_ms'].mean():.2f} ms")
        
        # Compare to baseline (fixed 100ms, no adjustment)
        baseline_mean = 66.07  # From full-signal filtering test
        improvement = baseline_mean - errors.mean()
        improvement_pct = (improvement / baseline_mean) * 100
        
        print(f"\nComparison to Baseline (fixed 100ms, no S wave adjustment):")
        print(f"  Baseline mean error: {baseline_mean:.2f} ms")
        print(f"  New mean error: {errors.mean():.2f} ms")
        print(f"  Improvement: {improvement:.2f} ms ({improvement_pct:.1f}%)")
        
        # Compare to ECGpuwave
        ecgpuwave_mean = 10.15
        gap_to_ecgpuwave = errors.mean() - ecgpuwave_mean
        print(f"\nComparison to ECGpuwave:")
        print(f"  ECGpuwave mean MAD: {ecgpuwave_mean:.2f} ms")
        print(f"  PyHEARTS mean MAD: {errors.mean():.2f} ms")
        print(f"  Remaining gap: {gap_to_ecgpuwave:.2f} ms")
    
    # Save summary
    summary_file = os.path.join(OUTPUT_DIR, 's_wave_end_adjustment_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("T-Peak Detection Results: S Wave End + 20ms Adjustment\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total beats analyzed: {total_beats}\n")
        f.write(f"T peaks detected: {detected}/{total_beats} ({detected/total_beats*100:.1f}%)\n")
        if detected > 0:
            errors = combined_df[combined_df['t_detected']]['error_ms']
            f.write(f"\nMean MAD: {errors.mean():.2f} ms\n")
            f.write(f"Median MAD: {errors.median():.2f} ms\n")
            f.write(f"% within ±20ms: {(errors <= 20).sum()/len(errors)*100:.1f}%\n")
    print(f"\nSummary saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

