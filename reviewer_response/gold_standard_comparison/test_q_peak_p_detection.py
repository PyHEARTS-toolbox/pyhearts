#!/usr/bin/env python3
"""
Test if Q Peak Detection Improves P Peak Detection Performance

This script compares P wave detection performance to see if detecting Q peaks
(now that we've removed the 300 Hz requirement) improves P peak detection accuracy.
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
OUTPUT_DIR = "/Users/morganfitzgerald/Documents/pyhearts/reviewer_response/gold_standard_comparison/q_peak_p_detection_test"
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
            'q_peak': None,
            't_peak': None,
        }
        
        # Find P peak (before R)
        for j in range(r_idx - 1, max(-1, r_idx - 20), -1):
            if ann.symbol[j] == 'p':
                beat['p_peak'] = ann.sample[j]
                break
        
        # Find Q peak (before R, after P if present)
        for j in range(r_idx - 1, max(-1, r_idx - 20), -1):
            if ann.symbol[j] == 'q' or ann.symbol[j] == 'Q':
                beat['q_peak'] = ann.sample[j]
                break
        
        # Find T peak (after R)
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
    
    # Run PyHEARTS (Q peaks are now detected at all sampling rates)
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
        
        for idx, row in output_df.iterrows():
            if pd.notna(row.get('R_global_center_idx')) and int(row['R_global_center_idx']) == closest_r:
                if pd.notna(row.get('P_global_center_idx')):
                    detected_p = int(row['P_global_center_idx'])
                    error_p_ms = abs(detected_p - man_p) * 1000.0 / sampling_rate
                if pd.notna(row.get('Q_global_center_idx')):
                    detected_q = int(row['Q_global_center_idx'])
                break
        
        results.append({
            'subject': subject,
            'manual_r': man_r,
            'manual_p': man_p,
            'manual_q': man_q if man_q is not None else np.nan,
            'pyhearts_r': closest_r,
            'pyhearts_p': detected_p if detected_p is not None else np.nan,
            'pyhearts_q': detected_q if detected_q is not None else np.nan,
            'error_p_ms': error_p_ms,
            'p_detected': detected_p is not None,
            'q_detected': detected_q is not None,
            'q_in_manual': man_q is not None,
        })
    
    if len(results) == 0:
        print(f"  No matched beats")
        return None
    
    df = pd.DataFrame(results)
    
    # Calculate statistics
    detected_p = df['p_detected'].sum()
    errors_p = df[df['p_detected']]['error_p_ms']
    detected_q = df['q_detected'].sum()
    
    print(f"  Analyzed {len(df)} beats")
    print(f"  P peaks detected: {detected_p}/{len(df)} ({detected_p/len(df)*100:.1f}%)")
    print(f"  Q peaks detected: {detected_q}/{len(df)} ({detected_q/len(df)*100:.1f}%)")
    if len(errors_p) > 0:
        print(f"  Mean P error: {errors_p.mean():.2f} ms")
        print(f"  Median P error: {errors_p.median():.2f} ms")
        print(f"  % within ±20ms: {(errors_p <= 20).sum()/len(errors_p)*100:.1f}%")
    
    # Analyze correlation between Q detection and P accuracy
    if detected_q > 0:
        with_q = df[df['q_detected']]
        without_q = df[~df['q_detected']]
        if len(with_q[with_q['p_detected']]) > 0 and len(without_q[without_q['p_detected']]) > 0:
            p_error_with_q = with_q[with_q['p_detected']]['error_p_ms'].mean()
            p_error_without_q = without_q[without_q['p_detected']]['error_p_ms'].mean()
            print(f"  P error with Q detected: {p_error_with_q:.2f} ms")
            print(f"  P error without Q detected: {p_error_without_q:.2f} ms")
    
    return df

def main():
    print("="*80)
    print("P-Peak Detection Test: Impact of Q Peak Detection")
    print("="*80)
    print(f"\nTest subjects: {len(TEST_SUBJECTS)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nThis test analyzes P wave detection performance with Q peaks now detected")
    print("(after removing 300 Hz requirement).")
    
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
    detailed_file = os.path.join(OUTPUT_DIR, 'q_peak_p_detection_results.csv')
    combined_df.to_csv(detailed_file, index=False)
    print(f"\n\nDetailed results saved to: {detailed_file}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY: P-PEAK DETECTION WITH Q PEAK DETECTION")
    print("="*80)
    
    total_beats = len(combined_df)
    detected_p = combined_df['p_detected'].sum()
    detected_q = combined_df['q_detected'].sum()
    
    print(f"\nTotal beats analyzed: {total_beats}")
    print(f"P peaks detected: {detected_p}/{total_beats} ({detected_p/total_beats*100:.1f}%)")
    print(f"Q peaks detected: {detected_q}/{total_beats} ({detected_q/total_beats*100:.1f}%)")
    
    if detected_p > 0:
        errors_p = combined_df[combined_df['p_detected']]['error_p_ms']
        print(f"\nP Peak Error Statistics:")
        print(f"  Mean MAD: {errors_p.mean():.2f} ms")
        print(f"  Median MAD: {errors_p.median():.2f} ms")
        print(f"  Std MAD: {errors_p.std():.2f} ms")
        print(f"  % within ±20ms: {(errors_p <= 20).sum()/len(errors_p)*100:.1f}%")
        print(f"  % within ±50ms: {(errors_p <= 50).sum()/len(errors_p)*100:.1f}%")
        
        # Analyze correlation between Q detection and P accuracy
        if detected_q > 0:
            with_q = combined_df[combined_df['q_detected']]
            without_q = combined_df[~combined_df['q_detected']]
            
            p_detected_with_q = with_q['p_detected'].sum()
            p_detected_without_q = without_q['p_detected'].sum()
            
            print(f"\nQ Peak Impact on P Detection:")
            print(f"  Beats with Q detected: {len(with_q)} ({len(with_q)/total_beats*100:.1f}%)")
            print(f"  Beats without Q detected: {len(without_q)} ({len(without_q)/total_beats*100:.1f}%)")
            print(f"  P detection rate with Q: {p_detected_with_q}/{len(with_q)} ({p_detected_with_q/len(with_q)*100:.1f}%)")
            print(f"  P detection rate without Q: {p_detected_without_q}/{len(without_q)} ({p_detected_without_q/len(without_q)*100:.1f}%)")
            
            if p_detected_with_q > 0 and p_detected_without_q > 0:
                p_error_with_q = with_q[with_q['p_detected']]['error_p_ms']
                p_error_without_q = without_q[without_q['p_detected']]['error_p_ms']
                
                if len(p_error_with_q) > 0 and len(p_error_without_q) > 0:
                    print(f"\nP Accuracy Comparison:")
                    print(f"  Mean error with Q detected: {p_error_with_q.mean():.2f} ms (n={len(p_error_with_q)})")
                    print(f"  Mean error without Q detected: {p_error_without_q.mean():.2f} ms (n={len(p_error_without_q)})")
                    improvement = p_error_without_q.mean() - p_error_with_q.mean()
                    improvement_pct = (improvement / p_error_without_q.mean() * 100) if p_error_without_q.mean() > 0 else 0
                    print(f"  Improvement: {improvement:.2f} ms ({improvement_pct:.1f}%)")
                    
                    print(f"  % within ±20ms with Q: {(p_error_with_q <= 20).sum()/len(p_error_with_q)*100:.1f}%")
                    print(f"  % within ±20ms without Q: {(p_error_without_q <= 20).sum()/len(p_error_without_q)*100:.1f}%")
        
        # Compare to baseline (from ALGORITHM_DIAGNOSIS.md)
        baseline_mean = 54.00  # From ALGORITHM_DIAGNOSIS.md
        improvement_vs_baseline = baseline_mean - errors_p.mean()
        improvement_pct_vs_baseline = (improvement_vs_baseline / baseline_mean) * 100
        
        print(f"\nComparison to Baseline (from ALGORITHM_DIAGNOSIS.md):")
        print(f"  Baseline mean error: {baseline_mean:.2f} ms")
        print(f"  Current mean error: {errors_p.mean():.2f} ms")
        print(f"  Improvement: {improvement_vs_baseline:.2f} ms ({improvement_pct_vs_baseline:.1f}%)")
        
        # Compare to ECGpuwave
        ecgpuwave_mean = 8.04
        gap_to_ecgpuwave = errors_p.mean() - ecgpuwave_mean
        print(f"\nComparison to ECGpuwave:")
        print(f"  ECGpuwave mean MAD: {ecgpuwave_mean:.2f} ms")
        print(f"  PyHEARTS mean MAD: {errors_p.mean():.2f} ms")
        print(f"  Remaining gap: {gap_to_ecgpuwave:.2f} ms")
    
    # Save summary
    summary_file = os.path.join(OUTPUT_DIR, 'q_peak_p_detection_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("P-Peak Detection Results: Impact of Q Peak Detection\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total beats analyzed: {total_beats}\n")
        f.write(f"P peaks detected: {detected_p}/{total_beats} ({detected_p/total_beats*100:.1f}%)\n")
        f.write(f"Q peaks detected: {detected_q}/{total_beats} ({detected_q/total_beats*100:.1f}%)\n")
        if detected_p > 0:
            errors_p = combined_df[combined_df['p_detected']]['error_p_ms']
            f.write(f"\nMean MAD: {errors_p.mean():.2f} ms\n")
            f.write(f"Median MAD: {errors_p.median():.2f} ms\n")
            f.write(f"% within ±20ms: {(errors_p <= 20).sum()/len(errors_p)*100:.1f}%\n")
    print(f"\nSummary saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

