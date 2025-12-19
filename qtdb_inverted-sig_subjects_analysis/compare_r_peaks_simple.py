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
        print(f"  First 10 R peaks: {qtdb_r_peaks[:10]}")
        print(f"  Last 10 R peaks: {qtdb_r_peaks[-10:]}")
        print(f"  Time range: {qtdb_r_peaks[0]/sampling_rate:.2f} to {qtdb_r_peaks[-1]/sampling_rate:.2f} seconds")
        
        # Load PyHEARTS results
        pyhearts_csv = results_dir / f"{subject}_pyhearts.csv"
        if not pyhearts_csv.exists():
            print(f"\n  ERROR: PyHEARTS results file not found: {pyhearts_csv}")
            continue
        
        df = pd.read_csv(pyhearts_csv)
        
        # Extract R peak indices (R_global_center_idx column)
        if 'R_global_center_idx' not in df.columns:
            print(f"\n  ERROR: R_global_center_idx column not found in CSV")
            print(f"  Available columns: {list(df.columns)[:10]}...")
            continue
        
        pyhearts_r_peaks = df['R_global_center_idx'].dropna().values.astype(int)
        
        print(f"\nPyHEARTS Detections:")
        print(f"  Number of R peaks: {len(pyhearts_r_peaks)}")
        print(f"  First 10 R peaks: {pyhearts_r_peaks[:10]}")
        print(f"  Last 10 R peaks: {pyhearts_r_peaks[-10:]}")
        if len(pyhearts_r_peaks) > 0:
            print(f"  Time range: {pyhearts_r_peaks[0]/sampling_rate:.2f} to {pyhearts_r_peaks[-1]/sampling_rate:.2f} seconds")
        
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
            
            print(f"\nMatched peaks: {len(matches)} / {len(qtdb_r_peaks)} QTDB peaks ({100*len(matches)/len(qtdb_r_peaks):.1f}%)")
            print(f"PyHEARTS peaks with matches: {len(pyhearts_matched)} / {len(pyhearts_r_peaks)} ({100*len(pyhearts_matched)/len(pyhearts_r_peaks):.1f}%)")
            
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
            
            # Show first 30 matches in detail
            print(f"\nFirst 30 matched peaks:")
            print(f"{'QTDB idx':<12} {'PyHEARTS idx':<15} {'Offset (ms)':<15} {'Offset (samples)':<20} {'QTDB time (s)':<15}")
            print("-" * 80)
            for i, match in enumerate(matches[:30]):
                qtdb_time = match['qtdb_idx'] / sampling_rate
                print(f"{match['qtdb_idx']:<12} {match['pyhearts_idx']:<15} {match['offset_ms']:>12.2f} {match['offset_samples']:>15} {qtdb_time:>12.3f}")
            
            # Save detailed comparison to CSV
            output_csv = base_dir / f"{subject}_rpeak_comparison.csv"
            matches_df.to_csv(output_csv, index=False)
            print(f"\n  Saved detailed comparison to: {output_csv}")
            
        else:
            print(f"\n  WARNING: No matching peaks found within {max_offset_samples} samples ({max_offset_samples/sampling_rate*1000:.0f} ms)")
            
    except Exception as e:
        print(f"\n  ERROR processing {subject}: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")


