#!/usr/bin/env python3
"""
Evaluate peak detection accuracy using ECGPUWAVE-style detection with PeakAnnotation.

Runs pyhearts with rpeak_method="ecgpuwave_style" on previously poor-performing subjects
and compares detected peaks with ECGPUWAVE annotations.
"""

import os
import sys
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from datetime import datetime
from dataclasses import replace

# Add pyhearts to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyhearts as ph

# Subjects to evaluate
SUBJECTS = ['sel302', 'sele0112', 'sel104', 'sele0114', 'sel100']

# Paths
QTDB_PATH = "/Users/morganfitzgerald/Documents/pyhearts/data/qtdb/1.0.0"
ECGPUWAVE_RESULTS_DIR = "/Users/morganfitzgerald/Documents/pyhearts/results/ecgpuwave_results_20251223_1319"
RESULTS_DIR = "/Users/morganfitzgerald/Documents/pyhearts/results"

def match_peaks_with_double_annotation_handling(
    pyhearts_peaks: np.ndarray, 
    ecgpuwave_peaks: np.ndarray,
    sampling_rate: float, 
    max_match_distance_ms: float = 200.0
):
    """Match peaks between pyhearts and ECGPUWAVE, handling double annotations."""
    if len(pyhearts_peaks) == 0 or len(ecgpuwave_peaks) == 0:
        return [], np.arange(len(pyhearts_peaks)), np.arange(len(ecgpuwave_peaks))
    
    max_match_distance_samples = int(round(max_match_distance_ms * sampling_rate / 1000.0))
    
    matched_pairs = []
    pyhearts_matched_indices = set()
    ecgpuwave_matched_indices = set()
    
    potential_matches = []
    for ph_idx, ph_peak in enumerate(pyhearts_peaks):
        distances = np.abs(ecgpuwave_peaks - ph_peak)
        closest_ecg_indices = np.where(distances <= max_match_distance_samples)[0]
        
        if len(closest_ecg_indices) > 0:
            min_dist_idx_in_closest = np.argmin(distances[closest_ecg_indices])
            ecg_idx = closest_ecg_indices[min_dist_idx_in_closest]
            distance_ms = distances[ecg_idx] * 1000.0 / sampling_rate
            potential_matches.append((distance_ms, ph_idx, ecg_idx))
    
    potential_matches.sort(key=lambda x: x[0])
    
    for dist_ms, ph_idx, ecg_idx in potential_matches:
        if ph_idx not in pyhearts_matched_indices and ecg_idx not in ecgpuwave_matched_indices:
            matched_pairs.append((ph_idx, ecg_idx, dist_ms))
            pyhearts_matched_indices.add(ph_idx)
            ecgpuwave_matched_indices.add(ecg_idx)
    
    pyhearts_unmatched = np.array([i for i in range(len(pyhearts_peaks)) if i not in pyhearts_matched_indices])
    ecgpuwave_unmatched = np.array([i for i in range(len(ecgpuwave_peaks)) if i not in ecgpuwave_matched_indices])
    
    return matched_pairs, pyhearts_unmatched, ecgpuwave_unmatched

def evaluate_subject(record_name: str):
    """Evaluate peak detection accuracy for a single subject."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {record_name}")
    print(f"{'='*80}")
    
    # Load signal
    original_dir = os.getcwd()
    os.chdir(QTDB_PATH)
    try:
        record = wfdb.rdrecord(record_name)
        sampling_rate = record.fs
        signal = record.p_signal[:, 0]
        signal_length = len(signal)
    finally:
        os.chdir(original_dir)
    
    print(f"Signal length: {len(signal)} samples ({len(signal)/sampling_rate:.1f}s)")
    
    # Load ECGPUWAVE annotations
    ecgpuwave_pu = os.path.join(ECGPUWAVE_RESULTS_DIR, f"{record_name}.pu")
    if not os.path.exists(ecgpuwave_pu):
        print(f"  Error: ECGPUWAVE annotations not found")
        return None
    
    os.chdir(os.path.dirname(ecgpuwave_pu))
    try:
        annotation = wfdb.rdann(ecgpuwave_pu.replace('.pu', ''), 'pu')
        
        ecg_r_peaks = np.array([s for i, s in enumerate(annotation.sample) if annotation.symbol[i] == 'N'])
        ecg_t_peaks = np.array([s for i, s in enumerate(annotation.sample) if annotation.symbol[i] in ['t', 'T']])
        ecg_p_peaks = np.array([s for i, s in enumerate(annotation.sample) if annotation.symbol[i] in ['p', 'P']])
        
        ecg_r_peaks = ecg_r_peaks[(ecg_r_peaks >= 0) & (ecg_r_peaks < signal_length)]
        ecg_t_peaks = ecg_t_peaks[(ecg_t_peaks >= 0) & (ecg_t_peaks < signal_length)]
        ecg_p_peaks = ecg_p_peaks[(ecg_p_peaks >= 0) & (ecg_p_peaks < signal_length)]
        
        ecg_r_peaks = np.unique(np.sort(ecg_r_peaks))
        ecg_t_peaks = np.unique(np.sort(ecg_t_peaks))
        ecg_p_peaks = np.unique(np.sort(ecg_p_peaks))
        
    finally:
        os.chdir(original_dir)
    
    print(f"\nECGPUWAVE Reference: R={len(ecg_r_peaks)}, T={len(ecg_t_peaks)}, P={len(ecg_p_peaks)}")
    
    # Run pyhearts with ECGPUWAVE-style detection
    print(f"\nRunning pyhearts with enhanced R-peak detection...")
    try:
        analyzer = ph.PyHEARTS(
            sampling_rate=sampling_rate,
            species="human",
            sensitivity="high"
        )
        
        # Use default method (now has ECGPUWAVE improvements built-in)
        output_df, epochs_df = analyzer.analyze_ecg(signal, verbose=False, plot=False)
        
        ph_r_peaks = output_df['R_global_center_idx'].dropna().values.astype(int)
        ph_t_peaks = output_df['T_global_center_idx'].dropna().values.astype(int)
        ph_p_peaks = output_df['P_global_center_idx'].dropna().values.astype(int)
        
        ph_r_peaks = np.unique(np.sort(ph_r_peaks))
        ph_t_peaks = np.unique(np.sort(ph_t_peaks))
        ph_p_peaks = np.unique(np.sort(ph_p_peaks))
        
        print(f"pyhearts Detected: R={len(ph_r_peaks)}, T={len(ph_t_peaks)}, P={len(ph_p_peaks)}")
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Evaluate R-peak detection
    print(f"\n{'='*80}")
    print(f"R-PEAK EVALUATION")
    print(f"{'='*80}")
    r_matched, _, _ = match_peaks_with_double_annotation_handling(
        ph_r_peaks, ecg_r_peaks, sampling_rate, max_match_distance_ms=150.0
    )
    
    r_recall = (len(r_matched) / len(ecg_r_peaks) * 100) if len(ecg_r_peaks) > 0 else 0.0
    r_precision = (len(r_matched) / len(ph_r_peaks) * 100) if len(ph_r_peaks) > 0 else 0.0
    r_errors_ms = [p[2] for p in r_matched]
    r_mean_error = np.mean(r_errors_ms) if len(r_errors_ms) > 0 else np.nan
    r_median_error = np.median(r_errors_ms) if len(r_errors_ms) > 0 else np.nan
    
    print(f"  Recall: {r_recall:.1f}% ({len(r_matched)}/{len(ecg_r_peaks)})")
    print(f"  Precision: {r_precision:.1f}% ({len(r_matched)}/{len(ph_r_peaks)})")
    print(f"  Mean error: {r_mean_error:.2f} ms")
    print(f"  Median error: {r_median_error:.2f} ms")
    
    # Evaluate T-peak detection
    print(f"\n{'='*80}")
    print(f"T-PEAK EVALUATION")
    print(f"{'='*80}")
    t_matched, _, _ = match_peaks_with_double_annotation_handling(
        ph_t_peaks, ecg_t_peaks, sampling_rate, max_match_distance_ms=200.0
    )
    
    t_recall = (len(t_matched) / len(ecg_t_peaks) * 100) if len(ecg_t_peaks) > 0 else 0.0
    t_precision = (len(t_matched) / len(ph_t_peaks) * 100) if len(ph_t_peaks) > 0 else 0.0
    t_errors_ms = [p[2] for p in t_matched]
    t_mean_error = np.mean(t_errors_ms) if len(t_errors_ms) > 0 else np.nan
    t_median_error = np.median(t_errors_ms) if len(t_errors_ms) > 0 else np.nan
    
    # Calculate distances to nearest ECGPUWAVE peak
    t_distances_to_ecg = []
    for ph_t in ph_t_peaks:
        if len(ecg_t_peaks) > 0:
            distances = np.abs(ecg_t_peaks - ph_t)
            min_dist = np.min(distances)
            t_distances_to_ecg.append(min_dist * 1000.0 / sampling_rate)
    
    t_mean_distance = np.mean(t_distances_to_ecg) if len(t_distances_to_ecg) > 0 else np.nan
    t_median_distance = np.median(t_distances_to_ecg) if len(t_distances_to_ecg) > 0 else np.nan
    
    print(f"  Recall: {t_recall:.1f}% ({len(t_matched)}/{len(ecg_t_peaks)})")
    print(f"  Precision: {t_precision:.1f}% ({len(t_matched)}/{len(ph_t_peaks)})")
    print(f"  Mean error (matched): {t_mean_error:.2f} ms")
    print(f"  Median error (matched): {t_median_error:.2f} ms")
    print(f"  Mean distance to nearest ECGPUWAVE: {t_mean_distance/1000:.2f}s ({t_median_distance:.2f}ms median)")
    
    return {
        'subject': record_name,
        'r_ph_count': len(ph_r_peaks),
        'r_ecg_count': len(ecg_r_peaks),
        'r_matched': len(r_matched),
        'r_recall': r_recall,
        'r_precision': r_precision,
        'r_mean_error_ms': r_mean_error,
        'r_median_error_ms': r_median_error,
        't_ph_count': len(ph_t_peaks),
        't_ecg_count': len(ecg_t_peaks),
        't_matched': len(t_matched),
        't_recall': t_recall,
        't_precision': t_precision,
        't_mean_error_ms': t_mean_error,
        't_median_error_ms': t_median_error,
        't_mean_distance_ms': t_mean_distance,
        't_median_distance_ms': t_median_distance,
    }

def main():
    """Main function."""
    print("="*80)
    print("Evaluating Enhanced R-Peak Detection (ECGPUWAVE-style improvements)")
    print("="*80)
    print(f"\nSubjects: {', '.join(SUBJECTS)}")
    print(f"Method: Enhanced r_peak_detection() with ECGPUWAVE-style improvements")
    
    results = []
    for i, subject in enumerate(SUBJECTS, 1):
        print(f"\n[{i}/{len(SUBJECTS)}]")
        result = evaluate_subject(subject)
        if result:
            results.append(result)
    
    if results:
        df = pd.DataFrame(results)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(RESULTS_DIR, f"enhanced_rpeak_accuracy_{timestamp}.csv")
        df.to_csv(output_file, index=False)
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*80}")
        
        print(f"\nSUMMARY:")
        print(f"  R-peak mean recall: {df['r_recall'].mean():.1f}%")
        print(f"  R-peak mean precision: {df['r_precision'].mean():.1f}%")
        print(f"  T-peak mean recall: {df['t_recall'].mean():.1f}%")
        print(f"  T-peak mean distance: {df['t_mean_distance_ms'].mean()/1000:.2f}s")
        print(f"\nPer-subject:")
        for _, row in df.iterrows():
            print(f"  {row['subject']}: R={row['r_recall']:.1f}%, T={row['t_recall']:.1f}%, T_dist={row['t_mean_distance_ms']/1000:.1f}s")

if __name__ == "__main__":
    main()


