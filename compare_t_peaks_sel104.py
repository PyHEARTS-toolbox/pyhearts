#!/usr/bin/env python3
"""
Compare T peak detections between pyhearts and ECGPUWAVE for sel104.

This script:
1. Loads pyhearts results from CSV
2. Loads ECGPUWAVE annotations from .pu file
3. Compares T peak positions and calculates accuracy metrics
4. Creates visualization comparing the two methods
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb

# QTDB path
QTDB_PATH = "/Users/morganfitzgerald/Documents/pyhearts/data/qtdb/1.0.0"
RECORD_NAME = "sel104"
DURATION_SECONDS = 15
ECGPUWAVE_RESULTS = "/Users/morganfitzgerald/Documents/pyhearts/results/ecgpuwave_results_20251223_1319/sel104.pu"
PYHEARTS_RESULTS = "/Users/morganfitzgerald/Documents/pyhearts/sel104_output_15s.csv"


def read_ecgpuwave_annotations(pu_file: str, sampling_rate: float, max_samples: int = None):
    """
    Read ECGPUWAVE annotation file (.pu format).
    
    ECGPUWAVE .pu files are WFDB annotation files.
    T peaks are marked with annotation code 't' (lowercase).
    This corresponds to the T wave center/peak point (itpos in ECGPUWAVE source).
    """
    try:
        # Read WFDB annotation file
        annotation = wfdb.rdann(pu_file.replace('.pu', ''), 'pu')
        
        # Extract T peaks (annotation code 't')
        # WFDB annotation codes: 't' = T peak center (itpos in ECGPUWAVE)
        # This is the T wave maximum/minimum point, not the onset or offset
        t_peaks = []
        for i, symbol in enumerate(annotation.symbol):
            if symbol == 't' or symbol == 'T':
                sample = annotation.sample[i]
                if max_samples is None or sample < max_samples:
                    t_peaks.append(sample)
        
        t_peaks = np.array(t_peaks, dtype=int)
        # Sort and remove duplicates
        t_peaks = np.unique(np.sort(t_peaks))
        
        print(f"  Read {len(t_peaks)} ECGPUWAVE T peak annotations (code 't' = T center/peak)")
        return t_peaks
    except Exception as e:
        print(f"Error reading ECGPUWAVE annotations: {e}")
        import traceback
        traceback.print_exc()
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


def match_peaks(pyhearts_peaks: np.ndarray, ecgpuwave_peaks: np.ndarray, 
                sampling_rate: float, max_match_distance_ms: float = 100.0):
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
        }
    
    distances_ms = [pair[2] for pair in matched_pairs]
    
    recall = len(matched_pairs) / len(ecgpuwave_peaks) if len(ecgpuwave_peaks) > 0 else 0.0
    precision = len(matched_pairs) / len(pyhearts_peaks) if len(pyhearts_peaks) > 0 else 0.0
    
    mean_error_ms = np.mean(distances_ms)
    mae_ms = np.mean(np.abs(distances_ms))
    rmse_ms = np.sqrt(np.mean(np.array(distances_ms)**2))
    
    return {
        'num_pyhearts': len(pyhearts_peaks),
        'num_ecgpuwave': len(ecgpuwave_peaks),
        'num_matched': len(matched_pairs),
        'recall': recall * 100.0,  # Percentage
        'precision': precision * 100.0,  # Percentage
        'mean_error_ms': mean_error_ms,
        'mae_ms': mae_ms,
        'rmse_ms': rmse_ms,
    }


def plot_comparison(ecg_signal: np.ndarray, sampling_rate: float,
                   pyhearts_peaks: np.ndarray, ecgpuwave_peaks: np.ndarray,
                   matched_pairs: list, metrics: dict, output_file: str):
    """Create visualization comparing pyhearts and ECGPUWAVE T peak detections."""
    time_axis = np.arange(len(ecg_signal)) / sampling_rate
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    
    # Full view
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time_axis, ecg_signal, color="gray", linewidth=1, alpha=0.6, label="ECG Signal", zorder=1)
    
    # Plot ECGPUWAVE T peaks (T center/peak point from annotation code 't')
    if len(ecgpuwave_peaks) > 0:
        ecg_valid = ecgpuwave_peaks[(ecgpuwave_peaks >= 0) & (ecgpuwave_peaks < len(ecg_signal))]
        if len(ecg_valid) > 0:
            ax1.scatter(
                time_axis[ecg_valid],
                ecg_signal[ecg_valid],
                color="blue",
                s=100,
                marker="^",
                label=f"ECGPUWAVE T Peaks ({len(ecg_valid)})",
                zorder=3,
                edgecolor="white",
                linewidth=1.5,
                alpha=0.5
            )
    
    # Plot pyhearts T peaks (T_global_center_idx = T center/peak point)
    if len(pyhearts_peaks) > 0:
        ph_valid = pyhearts_peaks[(pyhearts_peaks >= 0) & (pyhearts_peaks < len(ecg_signal))]
        if len(ph_valid) > 0:
            ax1.scatter(
                time_axis[ph_valid],
                ecg_signal[ph_valid],
                color="magenta",
                s=100,
                marker="s",
                label=f"pyhearts T Peaks ({len(ph_valid)})",
                zorder=3,
                edgecolor="white",
                linewidth=1.5,
                alpha=0.5
            )
    
    # Draw lines connecting matched peaks
    for ph_idx, ecg_idx, dist_ms in matched_pairs:
        ph_peak = pyhearts_peaks[ph_idx]
        ecg_peak = ecgpuwave_peaks[ecg_idx]
        if 0 <= ph_peak < len(ecg_signal) and 0 <= ecg_peak < len(ecg_signal):
            ax1.plot(
                [time_axis[ph_peak], time_axis[ecg_peak]],
                [ecg_signal[ph_peak], ecg_signal[ecg_peak]],
                color="green",
                linewidth=1,
                alpha=0.5,
                linestyle="--",
                zorder=2
            )
    
    # Add metrics text
    metrics_text = (
        f"Metrics:\n"
        f"  ECGPUWAVE: {metrics['num_ecgpuwave']} T peaks\n"
        f"  pyhearts: {metrics['num_pyhearts']} T peaks\n"
        f"  Matched: {metrics['num_matched']}\n"
        f"  Recall: {metrics['recall']:.1f}%\n"
        f"  Precision: {metrics['precision']:.1f}%\n"
        f"  Mean Error: {metrics['mean_error_ms']:.2f} ms\n"
        f"  MAE: {metrics['mae_ms']:.2f} ms\n"
        f"  RMSE: {metrics['rmse_ms']:.2f} ms"
    )
    ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Amplitude (mV)", fontsize=12)
    ax1.set_title(f"T Peak Detection Comparison: pyhearts vs ECGPUWAVE - {RECORD_NAME} (First {DURATION_SECONDS}s)", 
                  fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=11)
    
    # Zoomed view 1 (0-5s)
    ax2 = fig.add_subplot(gs[1])
    zoom_end = min(5.0, DURATION_SECONDS)
    zoom_end_idx = int(zoom_end * sampling_rate)
    zoom_time = time_axis[:zoom_end_idx]
    zoom_signal = ecg_signal[:zoom_end_idx]
    
    ax2.plot(zoom_time, zoom_signal, color="gray", linewidth=2, alpha=0.6, zorder=1)
    
    # Plot peaks in zoomed view
    if len(ecgpuwave_peaks) > 0:
        ecg_zoom = ecgpuwave_peaks[(ecgpuwave_peaks >= 0) & (ecgpuwave_peaks < zoom_end_idx)]
        if len(ecg_zoom) > 0:
            ax2.scatter(
                time_axis[ecg_zoom],
                ecg_signal[ecg_zoom],
                color="blue",
                s=120,
                marker="^",
                label="ECGPUWAVE",
                zorder=3,
                edgecolor="white",
                linewidth=2,
                alpha=0.5
            )
    
    if len(pyhearts_peaks) > 0:
        ph_zoom = pyhearts_peaks[(pyhearts_peaks >= 0) & (pyhearts_peaks < zoom_end_idx)]
        if len(ph_zoom) > 0:
            ax2.scatter(
                time_axis[ph_zoom],
                ecg_signal[ph_zoom],
                color="magenta",
                s=120,
                marker="s",
                label="pyhearts",
                zorder=3,
                edgecolor="white",
                linewidth=2,
                alpha=0.5
            )
    
    # Draw matched lines
    for ph_idx, ecg_idx, dist_ms in matched_pairs:
        ph_peak = pyhearts_peaks[ph_idx]
        ecg_peak = ecgpuwave_peaks[ecg_idx]
        if 0 <= ph_peak < zoom_end_idx and 0 <= ecg_peak < zoom_end_idx:
            ax2.plot(
                [time_axis[ph_peak], time_axis[ecg_peak]],
                [ecg_signal[ph_peak], ecg_signal[ecg_peak]],
                color="green",
                linewidth=1.5,
                alpha=0.7,
                linestyle="--",
                zorder=2
            )
    
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("Amplitude (mV)", fontsize=11)
    ax2.set_title(f"Zoomed View: 0-{zoom_end}s", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=10)
    
    # Zoomed view 2 (5-10s)
    if DURATION_SECONDS > 5:
        ax3 = fig.add_subplot(gs[2])
        zoom_start = 5.0
        zoom_end = min(10.0, DURATION_SECONDS)
        zoom_start_idx = int(zoom_start * sampling_rate)
        zoom_end_idx = int(zoom_end * sampling_rate)
        zoom_time = time_axis[zoom_start_idx:zoom_end_idx]
        zoom_signal = ecg_signal[zoom_start_idx:zoom_end_idx]
        
        ax3.plot(zoom_time, zoom_signal, color="gray", linewidth=2, alpha=0.6, zorder=1)
        
        # Plot peaks
        if len(ecgpuwave_peaks) > 0:
            ecg_zoom = ecgpuwave_peaks[(ecgpuwave_peaks >= zoom_start_idx) & (ecgpuwave_peaks < zoom_end_idx)]
            if len(ecg_zoom) > 0:
                ax3.scatter(
                    time_axis[ecg_zoom],
                    ecg_signal[ecg_zoom],
                    color="blue",
                    s=120,
                    marker="^",
                    label="ECGPUWAVE",
                    zorder=3,
                    edgecolor="white",
                    linewidth=2,
                    alpha=0.5
                )
        
        if len(pyhearts_peaks) > 0:
            ph_zoom = pyhearts_peaks[(pyhearts_peaks >= zoom_start_idx) & (pyhearts_peaks < zoom_end_idx)]
            if len(ph_zoom) > 0:
                ax3.scatter(
                    time_axis[ph_zoom],
                    ecg_signal[ph_zoom],
                    color="magenta",
                    s=120,
                    marker="s",
                    label="pyhearts",
                    zorder=3,
                    edgecolor="white",
                    linewidth=2,
                    alpha=0.5
                )
        
        # Draw matched lines
        for ph_idx, ecg_idx, dist_ms in matched_pairs:
            ph_peak = pyhearts_peaks[ph_idx]
            ecg_peak = ecgpuwave_peaks[ecg_idx]
            if zoom_start_idx <= ph_peak < zoom_end_idx and zoom_start_idx <= ecg_peak < zoom_end_idx:
                ax3.plot(
                    [time_axis[ph_peak], time_axis[ecg_peak]],
                    [ecg_signal[ph_peak], ecg_signal[ecg_peak]],
                    color="green",
                    linewidth=1.5,
                    alpha=0.7,
                    linestyle="--",
                    zorder=2
                )
        
        ax3.set_xlabel("Time (s)", fontsize=11)
        ax3.set_ylabel("Amplitude (mV)", fontsize=11)
        ax3.set_title(f"Zoomed View: {zoom_start}-{zoom_end}s", fontsize=12, fontweight="bold")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="upper right", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved to {output_file}")
    plt.close()


def main():
    """Main function."""
    print(f"Comparing T peak detections for {RECORD_NAME}...")
    
    # Load signal to get sampling rate
    original_dir = os.getcwd()
    os.chdir(QTDB_PATH)
    try:
        record = wfdb.rdrecord(RECORD_NAME)
        sampling_rate = record.fs
        signal = record.p_signal[:, 0]
    finally:
        os.chdir(original_dir)
    
    # Crop to first 15 seconds
    max_samples = int(DURATION_SECONDS * sampling_rate)
    signal = signal[:max_samples]
    
    print(f"Signal: {len(signal)} samples at {sampling_rate} Hz")
    
    # Read T peaks
    print("\nLoading T peaks...")
    pyhearts_peaks = read_pyhearts_t_peaks(PYHEARTS_RESULTS)
    print(f"  pyhearts: {len(pyhearts_peaks)} T peaks")
    
    # Read ECGPUWAVE annotations (need to change directory to where .pu file is)
    ecgpuwave_dir = os.path.dirname(ECGPUWAVE_RESULTS)
    ecgpuwave_basename = os.path.basename(ECGPUWAVE_RESULTS).replace('.pu', '')
    os.chdir(ecgpuwave_dir)
    try:
        ecgpuwave_peaks = read_ecgpuwave_annotations(ECGPUWAVE_RESULTS, sampling_rate, max_samples)
    finally:
        os.chdir(original_dir)
    
    print(f"  ECGPUWAVE: {len(ecgpuwave_peaks)} T peaks")
    
    # Match peaks (use larger threshold - 200ms to account for potential timing differences)
    print("\nMatching peaks...")
    matched_pairs, ph_unmatched, ecg_unmatched = match_peaks(
        pyhearts_peaks, ecgpuwave_peaks, sampling_rate, max_match_distance_ms=200.0
    )
    print(f"  Matched: {len(matched_pairs)} pairs")
    print(f"  pyhearts unmatched: {len(ph_unmatched)}")
    print(f"  ECGPUWAVE unmatched: {len(ecg_unmatched)}")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(matched_pairs, pyhearts_peaks, ecgpuwave_peaks, sampling_rate)
    
    print("\n=== Accuracy Metrics ===")
    print(f"ECGPUWAVE T peaks: {metrics['num_ecgpuwave']}")
    print(f"pyhearts T peaks: {metrics['num_pyhearts']}")
    print(f"Matched pairs: {metrics['num_matched']}")
    print(f"Recall: {metrics['recall']:.1f}% (pyhearts found {metrics['num_matched']}/{metrics['num_ecgpuwave']} ECGPUWAVE peaks)")
    print(f"Precision: {metrics['precision']:.1f}% ({metrics['num_matched']}/{metrics['num_pyhearts']} pyhearts peaks matched)")
    print(f"Mean Error: {metrics['mean_error_ms']:.2f} ms")
    print(f"MAE (Mean Absolute Error): {metrics['mae_ms']:.2f} ms")
    print(f"RMSE (Root Mean Squared Error): {metrics['rmse_ms']:.2f} ms")
    
    if len(matched_pairs) > 0:
        distances_ms = [pair[2] for pair in matched_pairs]
        print(f"\nDistance Statistics (matched peaks):")
        print(f"  Min: {np.min(distances_ms):.2f} ms")
        print(f"  Max: {np.max(distances_ms):.2f} ms")
        print(f"  Median: {np.median(distances_ms):.2f} ms")
        print(f"  Std: {np.std(distances_ms):.2f} ms")
    
    # Create visualization
    print("\nCreating comparison plot...")
    output_file = "sel104_t_peak_comparison.png"
    plot_comparison(signal, sampling_rate, pyhearts_peaks, ecgpuwave_peaks, 
                   matched_pairs, metrics, output_file)
    
    print(f"\nComparison complete! Plot saved to {output_file}")


if __name__ == "__main__":
    main()

