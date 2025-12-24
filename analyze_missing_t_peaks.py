#!/usr/bin/env python3
"""
Analyze which T peaks pyhearts is missing compared to ECGPUWAVE.
"""

import os
import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt

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

def main():
    # Load signal
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
    
    # Read T peaks
    pyhearts_peaks = read_pyhearts_t_peaks(PYHEARTS_RESULTS)
    ecgpuwave_dir = os.path.dirname(ECGPUWAVE_RESULTS)
    os.chdir(ecgpuwave_dir)
    try:
        ecgpuwave_peaks = read_ecgpuwave_annotations(ECGPUWAVE_RESULTS, sampling_rate, max_samples)
    finally:
        os.chdir(original_dir)
    
    print(f"ECGPUWAVE: {len(ecgpuwave_peaks)} T peaks")
    print(f"pyhearts: {len(pyhearts_peaks)} T peaks")
    
    # Match peaks
    max_match_distance_samples = int(round(200.0 * sampling_rate / 1000.0))
    matched_ecg = set()
    matched_ph = set()
    
    for ph_peak in pyhearts_peaks:
        distances = np.abs(ecgpuwave_peaks - ph_peak)
        closest_idx = np.argmin(distances)
        closest_distance = distances[closest_idx]
        
        if closest_distance <= max_match_distance_samples:
            matched_ecg.add(closest_idx)
            matched_ph.add(ph_peak)
    
    unmatched_ecg = [i for i in range(len(ecgpuwave_peaks)) if i not in matched_ecg]
    unmatched_ecg_peaks = ecgpuwave_peaks[unmatched_ecg]
    
    print(f"\nUnmatched ECGPUWAVE T peaks ({len(unmatched_ecg_peaks)}):")
    for peak in unmatched_ecg_peaks:
        time_s = peak / sampling_rate
        print(f"  Index {peak} ({time_s:.2f}s)")
    
    # Check for beats with two T peaks (within 500ms of each other)
    print(f"\nChecking for beats with multiple T peaks...")
    two_peak_threshold = int(round(500.0 * sampling_rate / 1000.0))
    
    ecg_groups = []
    i = 0
    while i < len(ecgpuwave_peaks):
        group = [ecgpuwave_peaks[i]]
        j = i + 1
        while j < len(ecgpuwave_peaks) and ecgpuwave_peaks[j] - ecgpuwave_peaks[i] <= two_peak_threshold:
            group.append(ecgpuwave_peaks[j])
            j += 1
        ecg_groups.append(group)
        i = j
    
    multi_peak_beats = [g for g in ecg_groups if len(g) > 1]
    print(f"Beats with multiple ECGPUWAVE T peaks: {len(multi_peak_beats)}")
    for group in multi_peak_beats:
        times = [p / sampling_rate for p in group]
        print(f"  {group} ({times})")

if __name__ == "__main__":
    main()


