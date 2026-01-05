#!/usr/bin/env python3
"""
Test P Peak Detection Accuracy After Q Detection Improvement

This script tests the impact of improved Q detection (42.0% → 55.3%) on P peak detection accuracy.
It compares current performance with the improved Q detection against the baseline.
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
OUTPUT_DIR = "/Users/morganfitzgerald/Documents/pyhearts/reviewer_response/gold_standard_comparison/p_detection_after_q_improvement"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test subjects (same as previous Q/P detection test)
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
            'q_peak': None,
        }
        
        # Find P peak (before R)
        for j in range(r_idx - 1, max(-1, r_idx - 40), -1):
            if ann.symbol[j] == 'p' or ann.symbol[j] == 'P':
                beat['p_peak'] = ann.sample[j]
                break
        
        # Find Q peak (before R)
        for j in range(r_idx - 1, max(-1, r_idx - 20), -1):
            if ann.symbol[j] == 'q' or ann.symbol[j] == 'Q':
                beat['q_peak'] = ann.sample[j]
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
    """Analyze P peak detection for a single subject."""
    print(f"\nAnalyzing {subject}...")
    
    # Load signal
    signal, sampling_rate = load_ecg_signal(subject)
    if signal is None:
        print(f"  Failed to load signal")
        return None
    
    # Load manual annotations
    manual_beats = load_manual_annotations(subject, 'q1c', sampling_rate)
    if len(manual_beats) == 0:
        print(f"  No manual annotations found")
        return None
    
    # Run PyHEARTS (with improved Q detection)
    try:
        analyzer = PyHEARTS(
            sampling_rate=sampling_rate,
            verbose=False,
        )
        output_df, epochs_df = analyzer.analyze_ecg(signal)
        
        if output_df is None or len(output_df) == 0:
            print(f"  No output from PyHEARTS")
            return None
        
        r_peaks_global = output_df['R_global_center_idx'].dropna().values
        r_peaks_global = np.array([int(r) for r in r_peaks_global if not np.isnan(r)], dtype=int)
        
    except Exception as e:
        import traceback
        print(f"  Error running PyHEARTS: {e}")
        traceback.print_exc()
        return None
    
    # Match manual beats to PyHEARTS results
    results = []
    
    for man_beat in manual_beats:
        if man_beat['r_peak'] is None or man_beat['p_peak'] is None:
            continue
        
        man_r = man_beat['r_peak']
        man_p = man_beat['p_peak']
        man_q = man_beat.get('q_peak')
        
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
        
        # Find corresponding P and Q peaks in PyHEARTS results
        detected_p = None
        detected_q = None
        error_p_ms = np.nan
        error_q_ms = np.nan
        
        for idx, row in output_df.iterrows():
            if pd.notna(row.get('R_global_center_idx')) and int(row['R_global_center_idx']) == closest_r:
                if pd.notna(row.get('P_global_center_idx')):
                    detected_p = int(row['P_global_center_idx'])
                    error_p_ms = abs(detected_p - man_p) * 1000.0 / sampling_rate
                
                if pd.notna(row.get('Q_global_center_idx')):
                    detected_q = int(row['Q_global_center_idx'])
                    if man_q is not None:
                        error_q_ms = abs(detected_q - man_q) * 1000.0 / sampling_rate
                break
        
        results.append({
            'subject': subject,
            'manual_r': man_r,
            'manual_p': man_p,
            'manual_q': man_q if man_q is not None else np.nan,
            'pyhearts_r': closest_r,
            'pyhearts_p': detected_p if detected_p is not None else np.nan,
            'pyhearts_q': detected_q if detected_q is not None else np.nan,
            'error_p_ms': error_p_ms if not np.isnan(error_p_ms) else np.nan,
            'error_q_ms': error_q_ms if not np.isnan(error_q_ms) else np.nan,
            'p_detected': detected_p is not None,
            'q_detected': detected_q is not None,
            'q_in_manual': man_q is not None,
        })
    
    if len(results) == 0:
        print(f"  No matched beats")
        return None
    
    df = pd.DataFrame(results)
    
    # Calculate statistics
    total = len(df)
    p_detected = df['p_detected'].sum()
    q_detected = df['q_detected'].sum()
    
    print(f"  Analyzed {total} beats")
    print(f"  P peaks detected: {p_detected}/{total} ({p_detected/total*100:.1f}%)")
    print(f"  Q peaks detected: {q_detected}/{total} ({q_detected/total*100:.1f}%)")
    
    if p_detected > 0:
        errors_p = df[df['p_detected']]['error_p_ms'].dropna()
        if len(errors_p) > 0:
            print(f"  Mean P error: {errors_p.mean():.2f} ms")
    
    return df

def main():
    print("="*80)
    print("P Peak Detection Accuracy After Q Detection Improvement")
    print("="*80)
    print(f"\nTest subjects: {len(TEST_SUBJECTS)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nRunning PyHEARTS with improved Q detection...")
    
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
    detailed_file = os.path.join(OUTPUT_DIR, 'p_detection_results.csv')
    combined_df.to_csv(detailed_file, index=False)
    print(f"\n\nDetailed results saved to: {detailed_file}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY: P PEAK DETECTION AFTER Q IMPROVEMENT")
    print("="*80)
    
    total_beats = len(combined_df)
    p_detected = combined_df['p_detected'].sum()
    q_detected = combined_df['q_detected'].sum()
    
    print(f"\nTotal beats analyzed: {total_beats}")
    print(f"P peaks detected: {p_detected}/{total_beats} ({p_detected/total_beats*100:.1f}%)")
    print(f"Q peaks detected: {q_detected}/{total_beats} ({q_detected/total_beats*100:.1f}%)")
    
    # Overall P error
    errors_p_all = combined_df[combined_df['p_detected']]['error_p_ms'].dropna()
    if len(errors_p_all) > 0:
        print(f"\nOverall P peak detection accuracy:")
        print(f"  Mean error: {errors_p_all.mean():.2f} ms")
        print(f"  Median error: {errors_p_all.median():.2f} ms")
        print(f"  % within ±20ms: {(errors_p_all.abs() <= 20).sum() / len(errors_p_all) * 100:.1f}%")
        print(f"  % within ±50ms: {(errors_p_all.abs() <= 50).sum() / len(errors_p_all) * 100:.1f}%")
    
    # Compare P error with vs without Q detection
    cases_with_q = combined_df[combined_df['q_detected']]
    cases_without_q = combined_df[~combined_df['q_detected']]
    
    print(f"\nP Peak Detection: With Q vs Without Q")
    print("-" * 60)
    
    if len(cases_with_q) > 0:
        errors_p_with_q = cases_with_q[cases_with_q['p_detected']]['error_p_ms'].dropna()
        if len(errors_p_with_q) > 0:
            print(f"\nWhen Q IS detected ({len(cases_with_q)} beats, {len(cases_with_q)/total_beats*100:.1f}%):")
            print(f"  P peaks detected: {cases_with_q['p_detected'].sum()}/{len(cases_with_q)} ({cases_with_q['p_detected'].sum()/len(cases_with_q)*100:.1f}%)")
            print(f"  Mean error: {errors_p_with_q.mean():.2f} ms")
            print(f"  Median error: {errors_p_with_q.median():.2f} ms")
            print(f"  % within ±20ms: {(errors_p_with_q.abs() <= 20).sum() / len(errors_p_with_q) * 100:.1f}%")
            print(f"  % within ±50ms: {(errors_p_with_q.abs() <= 50).sum() / len(errors_p_with_q) * 100:.1f}%")
    
    if len(cases_without_q) > 0:
        errors_p_without_q = cases_without_q[cases_without_q['p_detected']]['error_p_ms'].dropna()
        if len(errors_p_without_q) > 0:
            print(f"\nWhen Q is NOT detected ({len(cases_without_q)} beats, {len(cases_without_q)/total_beats*100:.1f}%):")
            print(f"  P peaks detected: {cases_without_q['p_detected'].sum()}/{len(cases_without_q)} ({cases_without_q['p_detected'].sum()/len(cases_without_q)*100:.1f}%)")
            print(f"  Mean error: {errors_p_without_q.mean():.2f} ms")
            print(f"  Median error: {errors_p_without_q.median():.2f} ms")
            print(f"  % within ±20ms: {(errors_p_without_q.abs() <= 20).sum() / len(errors_p_without_q) * 100:.1f}%")
            print(f"  % within ±50ms: {(errors_p_without_q.abs() <= 50).sum() / len(errors_p_without_q) * 100:.1f}%")
    
    # Compare with baseline (from previous test)
    # Baseline: 117.60 ms overall, 41.40 ms when Q detected, 190.79 ms when Q not detected
    baseline_overall = 117.60
    baseline_with_q = 41.40
    baseline_without_q = 190.79
    baseline_q_rate = 0.487  # 48.7%
    
    print(f"\n" + "="*80)
    print("COMPARISON WITH BASELINE (before Q improvement)")
    print("="*80)
    print(f"\nBaseline (Q detection: {baseline_q_rate*100:.1f}%):")
    print(f"  Overall mean error: {baseline_overall:.2f} ms")
    print(f"  With Q detected: {baseline_with_q:.2f} ms")
    print(f"  Without Q detected: {baseline_without_q:.2f} ms")
    
    if len(errors_p_all) > 0:
        current_overall = errors_p_all.mean()
        improvement_overall = baseline_overall - current_overall
        improvement_pct = (improvement_overall / baseline_overall) * 100
        
        print(f"\nAfter Q improvement (Q detection: {q_detected/total_beats*100:.1f}%):")
        print(f"  Overall mean error: {current_overall:.2f} ms")
        if len(errors_p_with_q) > 0:
            current_with_q = errors_p_with_q.mean()
            improvement_with_q = baseline_with_q - current_with_q
            print(f"  With Q detected: {current_with_q:.2f} ms (improvement: {improvement_with_q:.2f} ms)")
        if len(errors_p_without_q) > 0:
            current_without_q = errors_p_without_q.mean()
            improvement_without_q = baseline_without_q - current_without_q
            print(f"  Without Q detected: {current_without_q:.2f} ms (improvement: {improvement_without_q:.2f} ms)")
        
        print(f"\nOverall improvement: {improvement_overall:.2f} ms ({improvement_pct:.1f}% reduction)")
        print(f"  Q detection rate: {baseline_q_rate*100:.1f}% → {q_detected/total_beats*100:.1f}% (+{(q_detected/total_beats - baseline_q_rate)*100:.1f} percentage points)")
    
    # Save summary
    summary_file = os.path.join(OUTPUT_DIR, 'p_detection_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("P Peak Detection After Q Improvement\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total beats: {total_beats}\n")
        f.write(f"Q detection rate: {q_detected/total_beats*100:.1f}%\n")
        if len(errors_p_all) > 0:
            f.write(f"Overall mean error: {errors_p_all.mean():.2f} ms\n")
            if len(errors_p_with_q) > 0:
                f.write(f"With Q detected: {errors_p_with_q.mean():.2f} ms\n")
            if len(errors_p_without_q) > 0:
                f.write(f"Without Q detected: {errors_p_without_q.mean():.2f} ms\n")
    print(f"\nSummary saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

