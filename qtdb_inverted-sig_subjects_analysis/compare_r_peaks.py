#!/usr/bin/env python3
"""
Compare QTDB R-peak annotations with PyHEARTS detected R peaks.

This script analyzes the timing differences between:
- QTDB annotations (ground truth R peaks from .atr files)
- PyHEARTS detected R peaks (from R_global_center_idx in CSV files)

For subjects: sele0121 and sele0211
"""

import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Paths
base_dir = Path(__file__).parent
qtdb_dir = base_dir / "qtdb_raw_files"
results_dir = base_dir / "pyhearts_results_1150"

subjects = ["sele0121", "sele0211"]

def get_preferred_lead_index(signal_names):
    """Select the best lead from available signals."""
    preferred_order = ["ECG2", "ECG1", "MLII", "V5", "V2", "CM5", "CM4", "ML5", "V1"]
    for preferred in preferred_order:
        for idx, name in enumerate(signal_names):
            if name.upper() == preferred.upper():
                return idx
    return 0

print("="*80)
print("COMPARING QTDB ANNOTATIONS WITH PYHEARTS DETECTED R PEAKS")
print("="*80)

for subject in subjects:
    print(f"\n{'='*80}")
    print(f"Subject: {subject}")
    print(f"{'='*80}")
    
    # Load QTDB record and annotations
    record_path = qtdb_dir / subject
    
    try:
        # Load signal to get sampling rate
        signals, fields = wfdb.rdsamp(str(record_path))
        signal_names = fields.get("sig_name", [])
        sampling_rate = fields.get("fs", None)
        
        # Load QTDB annotations
        ann = wfdb.rdann(str(record_path), 'atr')
        qtdb_r_peaks = np.array(ann.sample)
        
        print(f"\nQTDB Annotations:")
        print(f"  Sampling rate: {sampling_rate} Hz")
        print(f"  Number of R peaks: {len(qtdb_r_peaks)}")
        print(f"  First few R peaks: {qtdb_r_peaks[:10]}")
        print(f"  Last few R peaks: {qtdb_r_peaks[-10:]}")
        
        # Load PyHEARTS results
        pyhearts_csv = results_dir / f"{subject}_pyhearts.csv"
        if not pyhearts_csv.exists():
            print(f"\n  ERROR: PyHEARTS results file not found: {pyhearts_csv}")
            continue
        
        df = pd.read_csv(pyhearts_csv)
        
        # Extract R peak indices (R_global_center_idx column)
        if 'R_global_center_idx' not in df.columns:
            print(f"\n  ERROR: R_global_center_idx column not found in CSV")
            continue
        
        pyhearts_r_peaks = df['R_global_center_idx'].dropna().values.astype(int)
        
        print(f"\nPyHEARTS Detections:")
        print(f"  Number of R peaks: {len(pyhearts_r_peaks)}")
        print(f"  First few R peaks: {pyhearts_r_peaks[:10]}")
        print(f"  Last few R peaks: {pyhearts_r_peaks[-10:]}")
        
        # Calculate timing differences
        print(f"\n{'='*80}")
        print(f"TIMING DIFFERENCES (PyHEARTS - QTDB)")
        print(f"{'='*80}")
        
        # Find matching peaks (within a reasonable window)
        max_offset_samples = int(sampling_rate * 0.2)  # 200ms window
        
        matches = []
        pyhearts_matched = set()
        
        for qtdb_idx in qtdb_r_peaks:
            # Find closest PyHEARTS peak
            distances = np.abs(pyhearts_r_peaks - qtdb_idx)
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]
            pyhearts_idx = pyhearts_r_peaks[min_dist_idx]
            
            if min_dist <= max_offset_samples:
                offset_samples = pyhearts_idx - qtdb_idx
                offset_ms = offset_samples / sampling_rate * 1000
                matches.append({
                    'qtdb_idx': qtdb_idx,
                    'pyhearts_idx': pyhearts_idx,
                    'offset_samples': offset_samples,
                    'offset_ms': offset_ms,
                    'distance_samples': min_dist
                })
                pyhearts_matched.add(min_dist_idx)
        
        if len(matches) > 0:
            matches_df = pd.DataFrame(matches)
            
            print(f"\nMatched peaks: {len(matches)} / {len(qtdb_r_peaks)} QTDB peaks")
            print(f"PyHEARTS peaks with matches: {len(pyhearts_matched)} / {len(pyhearts_r_peaks)}")
            
            print(f"\nOffset Statistics (PyHEARTS - QTDB):")
            print(f"  Mean offset: {matches_df['offset_ms'].mean():.2f} ms")
            print(f"  Median offset: {matches_df['offset_ms'].median():.2f} ms")
            print(f"  Std offset: {matches_df['offset_ms'].std():.2f} ms")
            print(f"  Min offset: {matches_df['offset_ms'].min():.2f} ms")
            print(f"  Max offset: {matches_df['offset_ms'].max():.2f} ms")
            print(f"  Mean absolute offset: {matches_df['offset_ms'].abs().mean():.2f} ms")
            print(f"  Median absolute offset: {matches_df['offset_ms'].abs().median():.2f} ms")
            
            # Count positive/negative offsets
            positive_offsets = (matches_df['offset_ms'] > 0).sum()
            negative_offsets = (matches_df['offset_ms'] < 0).sum()
            zero_offsets = (matches_df['offset_ms'] == 0).sum()
            
            print(f"\nOffset Direction:")
            print(f"  PyHEARTS later than QTDB: {positive_offsets} ({100*positive_offsets/len(matches):.1f}%)")
            print(f"  PyHEARTS earlier than QTDB: {negative_offsets} ({100*negative_offsets/len(matches):.1f}%)")
            print(f"  Perfect match: {zero_offsets} ({100*zero_offsets/len(matches):.1f}%)")
            
            # Show first 20 matches in detail
            print(f"\nFirst 20 matched peaks:")
            print(f"{'QTDB idx':<12} {'PyHEARTS idx':<15} {'Offset (ms)':<15} {'Offset (samples)':<20}")
            print("-" * 65)
            for i, match in enumerate(matches[:20]):
                print(f"{match['qtdb_idx']:<12} {match['pyhearts_idx']:<15} {match['offset_ms']:>12.2f} {match['offset_samples']:>15}")
            
            # Create visualization
            print(f"\nCreating visualization...")
            
            # Load signal for visualization
            lead_idx = get_preferred_lead_index(signal_names)
            ecg_signal = signals[:, lead_idx]
            
            # Plot sample region (first 5 seconds)
            sample_duration_sec = 5
            sample_end = int(sampling_rate * sample_duration_sec)
            sample_end = min(sample_end, len(ecg_signal))
            
            time_axis = np.arange(sample_end) / sampling_rate
            
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # Plot 1: Signal with annotations
            ax1 = axes[0]
            ax1.plot(time_axis, ecg_signal[:sample_end], 'k-', linewidth=0.8, alpha=0.7, label='ECG Signal')
            
            # Plot QTDB annotations
            qtdb_in_range = qtdb_r_peaks[qtdb_r_peaks < sample_end]
            if len(qtdb_in_range) > 0:
                qtdb_times = qtdb_in_range / sampling_rate
                qtdb_values = ecg_signal[qtdb_in_range]
                ax1.scatter(qtdb_times, qtdb_values, c='red', marker='v', s=100, 
                          label=f'QTDB ({len(qtdb_in_range)})', zorder=5)
            
            # Plot PyHEARTS detections
            pyhearts_in_range = pyhearts_r_peaks[pyhearts_r_peaks < sample_end]
            if len(pyhearts_in_range) > 0:
                pyhearts_times = pyhearts_in_range / sampling_rate
                pyhearts_values = ecg_signal[pyhearts_in_range]
                ax1.scatter(pyhearts_times, pyhearts_values, c='blue', marker='^', s=100,
                          label=f'PyHEARTS ({len(pyhearts_in_range)})', zorder=5)
            
            ax1.set_xlabel('Time (seconds)', fontsize=12)
            ax1.set_ylabel('Amplitude', fontsize=12)
            ax1.set_title(f'{subject} - R Peak Comparison (First {sample_duration_sec} seconds)', fontsize=14, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Offset distribution
            ax2 = axes[1]
            ax2.hist(matches_df['offset_ms'], bins=50, edgecolor='black', alpha=0.7)
            ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero offset')
            ax2.axvline(matches_df['offset_ms'].mean(), color='green', linestyle='--', 
                       linewidth=2, label=f'Mean: {matches_df["offset_ms"].mean():.2f} ms')
            ax2.axvline(matches_df['offset_ms'].median(), color='orange', linestyle='--',
                       linewidth=2, label=f'Median: {matches_df["offset_ms"].median():.2f} ms')
            ax2.set_xlabel('Offset (PyHEARTS - QTDB) in milliseconds', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title(f'{subject} - R Peak Timing Offset Distribution', fontsize=14, fontweight='bold')
            ax2.legend(loc='upper right', fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            output_path = base_dir / f"{subject}_rpeak_comparison.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  Saved figure to: {output_path}")
            plt.close()
            
            # Save detailed comparison to CSV
            output_csv = base_dir / f"{subject}_rpeak_comparison.csv"
            matches_df.to_csv(output_csv, index=False)
            print(f"  Saved detailed comparison to: {output_csv}")
            
        else:
            print(f"\n  WARNING: No matching peaks found within {max_offset_samples} samples ({max_offset_samples/sampling_rate*1000:.0f} ms)")
            
    except Exception as e:
        print(f"\n  ERROR processing {subject}: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")

