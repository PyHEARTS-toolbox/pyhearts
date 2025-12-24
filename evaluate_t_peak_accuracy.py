#!/usr/bin/env python3
"""
Evaluate pyhearts T peak detection accuracy against ECGPUWAVE for sel104,
excluding beats where ECGPUWAVE detected two T peaks.
"""

import os
import numpy as np
import pandas as pd
import wfdb

# QTDB path
QTDB_PATH = "/Users/morganfitzgerald/Documents/pyhearts/data/qtdb/1.0.0"
RECORD_NAME = "sel104"
DURATION_SECONDS = 15
ECGPUWAVE_RESULTS = "/Users/morganfitzgerald/Documents/pyhearts/results/ecgpuwave_results_20251223_1319/sel104.pu"
PYHEARTS_RESULTS = "/Users/morganfitzgerald/Documents/pyhearts/sel104_output_15s.csv"


def read_ecgpuwave_annotations(pu_file: str, sampling_rate: float, max_samples: int = None):
    """Read ECGPUWAVE annotation file."""
    try:
        annotation = wfdb.rdann(pu_file.replace('.pu', ''), 'pu')
        t_peaks = []
        for i, symbol in enumerate(annotation.symbol):
            if symbol == 't' or symbol == 'T':
                sample = annotation.sample[i]
                if max_samples is None or sample < max_samples:
                    t_peaks.append(sample)
        t_peaks = np.array(t_peaks, dtype=int)
        t_peaks = np.unique(np.sort(t_peaks))
        return t_peaks
    except Exception as e:
        print(f"Error reading ECGPUWAVE annotations: {e}")
        return np.array([], dtype=int)


def read_pyhearts_t_peaks(csv_file: str):
    """Read T peaks from pyhearts output CSV."""
    try:
        df = pd.read_csv(csv_file, index_col=0)
        t_peaks = df['T_global_center_idx'].dropna().values
        t_peaks = t_peaks[np.isfinite(t_peaks)]
        return t_peaks.astype(int)
    except Exception as e:
        print(f"Error reading pyhearts results: {e}")
        return np.array([], dtype=int)


def identify_double_peak_beats(t_peaks: np.ndarray, sampling_rate: float, threshold_ms: float = 500.0):
    """
    Identify beats with multiple T peaks (within threshold_ms of each other).
    
    Returns:
    - set of indices in t_peaks that are part of double-peak beats
    """
    if len(t_peaks) < 2:
        return set()
    
    threshold_samples = int(round(threshold_ms * sampling_rate / 1000.0))
    double_peak_indices = set()
    
    i = 0
    while i < len(t_peaks):
        # Find all peaks within threshold of current peak
        group = [i]
        j = i + 1
        while j < len(t_peaks) and t_peaks[j] - t_peaks[i] <= threshold_samples:
            group.append(j)
            j += 1
        
        # If group has more than one peak, mark all as double-peak
        if len(group) > 1:
            double_peak_indices.update(group)
        
        i = j
    
    return double_peak_indices


def match_peaks(pyhearts_peaks: np.ndarray, ecgpuwave_peaks: np.ndarray, 
                sampling_rate: float, max_match_distance_ms: float = 200.0):
    """
    Match peaks between pyhearts and ECGPUWAVE.
    
    Returns:
    - matched_pairs: list of (pyhearts_idx, ecgpuwave_idx, distance_ms)
    - pyhearts_unmatched: indices in pyhearts not matched
    - ecgpuwave_unmatched: indices in ECGPUWAVE not matched
    """
    if len(pyhearts_peaks) == 0 or len(ecgpuwave_peaks) == 0:
        return [], pyhearts_peaks.copy(), ecgpuwave_peaks.copy()
    
    max_match_distance_samples = int(round(max_match_distance_ms * sampling_rate / 1000.0))
    
    matched_pairs = []
    pyhearts_matched = set()
    ecgpuwave_matched = set()
    
    # For each pyhearts peak, find closest ECGPUWAVE peak
    for ph_idx, ph_peak in enumerate(pyhearts_peaks):
        distances = np.abs(ecgpuwave_peaks - ph_peak)
        closest_idx = np.argmin(distances)
        closest_distance = distances[closest_idx]
        
        if closest_distance <= max_match_distance_samples:
            ecg_idx = closest_idx
            if ecg_idx not in ecgpuwave_matched:
                distance_ms = closest_distance * 1000.0 / sampling_rate
                matched_pairs.append((ph_idx, ecg_idx, distance_ms))
                pyhearts_matched.add(ph_idx)
                ecgpuwave_matched.add(ecg_idx)
    
    pyhearts_unmatched = np.array([i for i in range(len(pyhearts_peaks)) if i not in pyhearts_matched])
    ecgpuwave_unmatched = np.array([i for i in range(len(ecgpuwave_peaks)) if i not in ecgpuwave_matched])
    
    return matched_pairs, pyhearts_unmatched, ecgpuwave_unmatched


def calculate_metrics(matched_pairs: list, pyhearts_peaks: np.ndarray, 
                      ecgpuwave_peaks: np.ndarray, sampling_rate: float):
    """Calculate accuracy metrics."""
    if len(matched_pairs) == 0:
        return {
            'num_pyhearts': len(pyhearts_peaks),
            'num_ecgpuwave': len(ecgpuwave_peaks),
            'num_matched': 0,
            'recall': 0.0,
            'precision': 0.0,
            'mean_error_ms': np.nan,
            'mae_ms': np.nan,
            'rmse_ms': np.nan,
            'median_error_ms': np.nan,
            'std_error_ms': np.nan,
        }
    
    distances_ms = [pair[2] for pair in matched_pairs]
    
    recall = len(matched_pairs) / len(ecgpuwave_peaks) if len(ecgpuwave_peaks) > 0 else 0.0
    precision = len(matched_pairs) / len(pyhearts_peaks) if len(pyhearts_peaks) > 0 else 0.0
    
    mean_error_ms = np.mean(distances_ms)
    mae_ms = np.mean(np.abs(distances_ms))
    rmse_ms = np.sqrt(np.mean(np.array(distances_ms)**2))
    median_error_ms = np.median(distances_ms)
    std_error_ms = np.std(distances_ms)
    
    return {
        'num_pyhearts': len(pyhearts_peaks),
        'num_ecgpuwave': len(ecgpuwave_peaks),
        'num_matched': len(matched_pairs),
        'recall': recall * 100.0,
        'precision': precision * 100.0,
        'mean_error_ms': mean_error_ms,
        'mae_ms': mae_ms,
        'rmse_ms': rmse_ms,
        'median_error_ms': median_error_ms,
        'std_error_ms': std_error_ms,
    }


def main():
    """Main function."""
    print("=" * 80)
    print("T Peak Detection Accuracy Evaluation: pyhearts vs ECGPUWAVE")
    print(f"Subject: {RECORD_NAME} ({DURATION_SECONDS}s)")
    print("=" * 80)
    
    # Load signal to get sampling rate
    original_dir = os.getcwd()
    os.chdir(QTDB_PATH)
    try:
        record = wfdb.rdrecord(RECORD_NAME)
        sampling_rate = record.fs
        signal = record.p_signal[:, 0]
    finally:
        os.chdir(original_dir)
    
    max_samples = int(DURATION_SECONDS * sampling_rate)
    signal = signal[:max_samples]
    
    print(f"\nSignal: {len(signal)} samples at {sampling_rate} Hz")
    
    # Read T peaks
    print("\nLoading T peaks...")
    pyhearts_peaks = read_pyhearts_t_peaks(PYHEARTS_RESULTS)
    print(f"  pyhearts: {len(pyhearts_peaks)} T peaks")
    
    # Read ECGPUWAVE annotations
    ecgpuwave_dir = os.path.dirname(ECGPUWAVE_RESULTS)
    os.chdir(ecgpuwave_dir)
    try:
        ecgpuwave_peaks_all = read_ecgpuwave_annotations(ECGPUWAVE_RESULTS, sampling_rate, max_samples)
    finally:
        os.chdir(original_dir)
    
    print(f"  ECGPUWAVE (all): {len(ecgpuwave_peaks_all)} T peaks")
    
    # Identify double-peak beats
    print("\nIdentifying beats with multiple ECGPUWAVE T peaks...")
    double_peak_indices = identify_double_peak_beats(ecgpuwave_peaks_all, sampling_rate, threshold_ms=500.0)
    print(f"  Found {len(double_peak_indices)} T peaks in double-peak beats")
    
    if len(double_peak_indices) > 0:
        double_peak_times = [ecgpuwave_peaks_all[i] / sampling_rate for i in double_peak_indices]
        print(f"  Double-peak T peaks at times: {[f'{t:.2f}s' for t in sorted(double_peak_times)]}")
    
    # Exclude double-peak beats from ECGPUWAVE peaks
    single_peak_indices = [i for i in range(len(ecgpuwave_peaks_all)) if i not in double_peak_indices]
    ecgpuwave_peaks_single = ecgpuwave_peaks_all[single_peak_indices]
    
    print(f"\n  ECGPUWAVE (single-peak beats only): {len(ecgpuwave_peaks_single)} T peaks")
    
    # Match peaks
    print("\nMatching peaks (max distance: 200ms)...")
    matched_pairs, ph_unmatched, ecg_unmatched = match_peaks(
        pyhearts_peaks, ecgpuwave_peaks_single, sampling_rate, max_match_distance_ms=200.0
    )
    print(f"  Matched: {len(matched_pairs)} pairs")
    print(f"  pyhearts unmatched: {len(ph_unmatched)}")
    print(f"  ECGPUWAVE unmatched: {len(ecg_unmatched)}")
    
    # Calculate metrics
    print("\nCalculating accuracy metrics...")
    metrics = calculate_metrics(matched_pairs, pyhearts_peaks, ecgpuwave_peaks_single, sampling_rate)
    
    print("\n" + "=" * 80)
    print("ACCURACY METRICS (Excluding Double-Peak Beats)")
    print("=" * 80)
    print(f"\nDetection Counts:")
    print(f"  ECGPUWAVE T peaks (single-peak beats): {metrics['num_ecgpuwave']}")
    print(f"  pyhearts T peaks: {metrics['num_pyhearts']}")
    print(f"  Matched pairs: {metrics['num_matched']}")
    
    print(f"\nDetection Performance:")
    print(f"  Recall: {metrics['recall']:.1f}% (pyhearts found {metrics['num_matched']}/{metrics['num_ecgpuwave']} ECGPUWAVE peaks)")
    print(f"  Precision: {metrics['precision']:.1f}% ({metrics['num_matched']}/{metrics['num_pyhearts']} pyhearts peaks matched)")
    
    if len(matched_pairs) > 0:
        distances_ms = [pair[2] for pair in matched_pairs]
        print(f"\nTiming Accuracy (matched peaks):")
        print(f"  Mean Error: {metrics['mean_error_ms']:.2f} ms")
        print(f"  MAE (Mean Absolute Error): {metrics['mae_ms']:.2f} ms")
        print(f"  RMSE (Root Mean Squared Error): {metrics['rmse_ms']:.2f} ms")
        print(f"  Median Error: {metrics['median_error_ms']:.2f} ms")
        print(f"  Std Dev: {metrics['std_error_ms']:.2f} ms")
        print(f"\nDistance Statistics:")
        print(f"  Min: {np.min(distances_ms):.2f} ms")
        print(f"  Max: {np.max(distances_ms):.2f} ms")
        print(f"  25th percentile: {np.percentile(distances_ms, 25):.2f} ms")
        print(f"  75th percentile: {np.percentile(distances_ms, 75):.2f} ms")
        
        # Count peaks within different error thresholds
        print(f"\nPeaks within error thresholds:")
        for threshold in [10, 20, 50, 100]:
            within_threshold = sum(1 for d in distances_ms if d <= threshold)
            pct = (within_threshold / len(distances_ms)) * 100.0
            print(f"  ≤{threshold} ms: {within_threshold}/{len(distances_ms)} ({pct:.1f}%)")
    
    if len(ecg_unmatched) > 0:
        print(f"\nUnmatched ECGPUWAVE T peaks ({len(ecg_unmatched)}):")
        for idx in ecg_unmatched[:10]:  # Show first 10
            peak_idx = ecgpuwave_peaks_single[idx]
            time_s = peak_idx / sampling_rate
            print(f"  Index {peak_idx} ({time_s:.2f}s)")
        if len(ecg_unmatched) > 10:
            print(f"  ... and {len(ecg_unmatched) - 10} more")
    
    if len(ph_unmatched) > 0:
        print(f"\nUnmatched pyhearts T peaks ({len(ph_unmatched)}):")
        for idx in ph_unmatched[:10]:  # Show first 10
            peak_idx = pyhearts_peaks[idx]
            time_s = peak_idx / sampling_rate
            print(f"  Index {peak_idx} ({time_s:.2f}s)")
        if len(ph_unmatched) > 10:
            print(f"  ... and {len(ph_unmatched) - 10} more")
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()


