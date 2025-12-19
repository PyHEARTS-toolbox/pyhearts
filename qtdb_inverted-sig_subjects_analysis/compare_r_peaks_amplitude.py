#!/usr/bin/env python3
"""
Compare QTDB R-peak annotations with PyHEARTS detected R peaks.
Analyzes both timing and amplitude differences, with visualizations.

This script analyzes:
- Timing differences (PyHEARTS - QTDB)
- Amplitude differences at detected locations
- Creates visualizations showing signal with both annotations

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
print("(Timing and Amplitude Analysis)")
print("="*80)

for subject in subjects:
    print(f"\n{'='*80}")
    print(f"Subject: {subject}")
    print(f"{'='*80}")
    
    # Load QTDB record and annotations
    record_path = qtdb_dir / subject
    
    try:
        # Load signal to get sampling rate and actual signal values
        signals, fields = wfdb.rdsamp(str(record_path))
        signal_names = fields.get("sig_name", [])
        sampling_rate = fields.get("fs", None)
        
        # Select preferred lead
        lead_idx = get_preferred_lead_index(signal_names)
        lead_name = signal_names[lead_idx] if lead_idx < len(signal_names) else "Unknown"
        ecg_signal = signals[:, lead_idx]
        
        print(f"\nSignal Info:")
        print(f"  Lead: {lead_name} (index {lead_idx})")
        print(f"  Sampling rate: {sampling_rate} Hz")
        print(f"  Signal length: {len(ecg_signal)} samples ({len(ecg_signal)/sampling_rate:.1f} seconds)")
        print(f"  Signal range: [{ecg_signal.min():.3f}, {ecg_signal.max():.3f}]")
        
        # Load QTDB annotations
        ann = wfdb.rdann(str(record_path), 'atr')
        qtdb_r_peaks = np.array(ann.sample)
        qtdb_r_amplitudes = ecg_signal[qtdb_r_peaks]
        
        print(f"\nQTDB Annotations:")
        print(f"  Number of R peaks: {len(qtdb_r_peaks)}")
        print(f"  Amplitude range at peaks: [{qtdb_r_amplitudes.min():.3f}, {qtdb_r_amplitudes.max():.3f}]")
        print(f"  Mean amplitude: {qtdb_r_amplitudes.mean():.3f}")
        
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
        # Make sure indices are within signal bounds
        pyhearts_r_peaks = pyhearts_r_peaks[(pyhearts_r_peaks >= 0) & (pyhearts_r_peaks < len(ecg_signal))]
        pyhearts_r_amplitudes = ecg_signal[pyhearts_r_peaks]
        
        print(f"\nPyHEARTS Detections:")
        print(f"  Number of R peaks: {len(pyhearts_r_peaks)}")
        print(f"  Amplitude range at peaks: [{pyhearts_r_amplitudes.min():.3f}, {pyhearts_r_amplitudes.max():.3f}]")
        print(f"  Mean amplitude: {pyhearts_r_amplitudes.mean():.3f}")
        
        # Calculate timing and amplitude differences
        print(f"\n{'='*80}")
        print(f"TIMING AND AMPLITUDE DIFFERENCES (PyHEARTS - QTDB)")
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
                
                # Get amplitudes at both locations
                qtdb_amp = ecg_signal[qtdb_idx]
                pyhearts_amp = ecg_signal[pyhearts_idx]
                amp_diff = pyhearts_amp - qtdb_amp
                
                matches.append({
                    'qtdb_idx': qtdb_idx,
                    'pyhearts_idx': pyhearts_idx,
                    'qtdb_amplitude': qtdb_amp,
                    'pyhearts_amplitude': pyhearts_amp,
                    'amplitude_diff': amp_diff,
                    'offset_samples': offset_samples,
                    'offset_ms': offset_ms,
                    'distance_samples': min_dist
                })
                pyhearts_matched.add(min_dist_idx)
        
        if len(matches) > 0:
            matches_df = pd.DataFrame(matches)
            
            print(f"\nMatched peaks: {len(matches)} / {len(qtdb_r_peaks)} QTDB peaks ({100*len(matches)/len(qtdb_r_peaks):.1f}%)")
            print(f"PyHEARTS peaks with matches: {len(pyhearts_matched)} / {len(pyhearts_r_peaks)} ({100*len(pyhearts_matched)/len(pyhearts_r_peaks):.1f}%)")
            
            print(f"\n--- TIMING OFFSET STATISTICS (PyHEARTS - QTDB) ---")
            print(f"  Mean offset: {matches_df['offset_ms'].mean():.2f} ms")
            print(f"  Median offset: {matches_df['offset_ms'].median():.2f} ms")
            print(f"  Std offset: {matches_df['offset_ms'].std():.2f} ms")
            print(f"  Min offset: {matches_df['offset_ms'].min():.2f} ms")
            print(f"  Max offset: {matches_df['offset_ms'].max():.2f} ms")
            print(f"  Mean absolute offset: {matches_df['offset_ms'].abs().mean():.2f} ms")
            
            print(f"\n--- AMPLITUDE DIFFERENCES (PyHEARTS - QTDB) ---")
            print(f"  Mean amplitude difference: {matches_df['amplitude_diff'].mean():.3f}")
            print(f"  Median amplitude difference: {matches_df['amplitude_diff'].median():.3f}")
            print(f"  Std amplitude difference: {matches_df['amplitude_diff'].std():.3f}")
            print(f"  Min amplitude difference: {matches_df['amplitude_diff'].min():.3f}")
            print(f"  Max amplitude difference: {matches_df['amplitude_diff'].max():.3f}")
            print(f"  Mean absolute amplitude difference: {matches_df['amplitude_diff'].abs().mean():.3f}")
            
            # Count direction of amplitude differences
            positive_amp_diff = (matches_df['amplitude_diff'] > 0).sum()
            negative_amp_diff = (matches_df['amplitude_diff'] < 0).sum()
            zero_amp_diff = (matches_df['amplitude_diff'] == 0).sum()
            
            print(f"\nAmplitude Difference Direction:")
            print(f"  PyHEARTS higher than QTDB: {positive_amp_diff} ({100*positive_amp_diff/len(matches):.1f}%)")
            print(f"  PyHEARTS lower than QTDB: {negative_amp_diff} ({100*negative_amp_diff/len(matches):.1f}%)")
            print(f"  Equal: {zero_amp_diff} ({100*zero_amp_diff/len(matches):.1f}%)")
            
            # Show first 20 matches in detail
            print(f"\nFirst 20 matched peaks (showing amplitude differences):")
            print(f"{'QTDB idx':<12} {'PyHEARTS idx':<15} {'QTDB amp':<12} {'PyHEARTS amp':<15} {'Amp diff':<12} {'Offset (ms)':<15}")
            print("-" * 90)
            for i, match in enumerate(matches[:20]):
                print(f"{match['qtdb_idx']:<12} {match['pyhearts_idx']:<15} "
                      f"{match['qtdb_amplitude']:>10.3f} {match['pyhearts_amplitude']:>13.3f} "
                      f"{match['amplitude_diff']:>10.3f} {match['offset_ms']:>13.2f}")
            
            # Save detailed comparison to CSV
            output_csv = base_dir / f"{subject}_rpeak_comparison_amplitude.csv"
            matches_df.to_csv(output_csv, index=False)
            print(f"\n  Saved detailed comparison to: {output_csv}")
            
            # Create visualizations
            print(f"\nCreating visualizations...")
            
            # Select 3 interesting regions to visualize
            signal_duration_sec = len(ecg_signal) / sampling_rate
            regions_to_plot = [
                (0, min(5, signal_duration_sec), "First 5 seconds"),
                (signal_duration_sec/3, signal_duration_sec/3 + 5, "Middle region"),
                (signal_duration_sec - 5, signal_duration_sec, "Last 5 seconds")
            ]
            
            fig, axes = plt.subplots(len(regions_to_plot), 1, figsize=(16, 4*len(regions_to_plot)))
            if len(regions_to_plot) == 1:
                axes = [axes]
            
            for region_idx, (start_sec, end_sec, title) in enumerate(regions_to_plot):
                start_sample = int(start_sec * sampling_rate)
                end_sample = int(end_sec * sampling_rate)
                end_sample = min(end_sample, len(ecg_signal))
                
                region_signal = ecg_signal[start_sample:end_sample]
                time_axis = np.arange(start_sample, end_sample) / sampling_rate
                
                ax = axes[region_idx]
                
                # Plot signal
                ax.plot(time_axis, region_signal, 'k-', linewidth=0.8, alpha=0.6, label='ECG Signal')
                
                # Plot QTDB annotations in this region
                qtdb_mask = (qtdb_r_peaks >= start_sample) & (qtdb_r_peaks < end_sample)
                qtdb_in_region = qtdb_r_peaks[qtdb_mask]
                if len(qtdb_in_region) > 0:
                    qtdb_times = qtdb_in_region / sampling_rate
                    qtdb_values = ecg_signal[qtdb_in_region]
                    ax.scatter(qtdb_times, qtdb_values, c='red', marker='v', s=150, 
                              label=f'QTDB ({len(qtdb_in_region)})', zorder=5, edgecolors='darkred', linewidths=1)
                
                # Plot PyHEARTS detections in this region
                pyhearts_mask = (pyhearts_r_peaks >= start_sample) & (pyhearts_r_peaks < end_sample)
                pyhearts_in_region = pyhearts_r_peaks[pyhearts_mask]
                if len(pyhearts_in_region) > 0:
                    pyhearts_times = pyhearts_in_region / sampling_rate
                    pyhearts_values = ecg_signal[pyhearts_in_region]
                    ax.scatter(pyhearts_times, pyhearts_values, c='blue', marker='^', s=150,
                              label=f'PyHEARTS ({len(pyhearts_in_region)})', zorder=5, edgecolors='darkblue', linewidths=1)
                
                # Draw connecting lines for matched peaks
                for match in matches:
                    qtdb_idx = match['qtdb_idx']
                    pyhearts_idx = match['pyhearts_idx']
                    if start_sample <= qtdb_idx < end_sample or start_sample <= pyhearts_idx < end_sample:
                        qtdb_time = qtdb_idx / sampling_rate
                        pyhearts_time = pyhearts_idx / sampling_rate
                        qtdb_val = ecg_signal[qtdb_idx]
                        pyhearts_val = ecg_signal[pyhearts_idx]
                        ax.plot([qtdb_time, pyhearts_time], [qtdb_val, pyhearts_val], 
                               'g--', alpha=0.4, linewidth=1, zorder=3)
                
                ax.set_xlabel('Time (seconds)', fontsize=11)
                ax.set_ylabel('Amplitude', fontsize=11)
                ax.set_title(f'{subject} - {title} (Mean offset: {matches_df["offset_ms"].mean():.1f} ms, '
                           f'Mean amp diff: {matches_df["amplitude_diff"].mean():.3f})', 
                           fontsize=12, fontweight='bold')
                ax.legend(loc='upper right', fontsize=9)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            output_path = base_dir / f"{subject}_rpeak_comparison_amplitude.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  Saved visualization to: {output_path}")
            plt.close()
            
            # Create histogram plots for offset and amplitude distributions
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Timing offset histogram
            ax1 = axes[0, 0]
            ax1.hist(matches_df['offset_ms'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
            ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero offset')
            ax1.axvline(matches_df['offset_ms'].mean(), color='green', linestyle='--', 
                       linewidth=2, label=f'Mean: {matches_df["offset_ms"].mean():.2f} ms')
            ax1.axvline(matches_df['offset_ms'].median(), color='orange', linestyle='--',
                       linewidth=2, label=f'Median: {matches_df["offset_ms"].median():.2f} ms')
            ax1.set_xlabel('Timing Offset (PyHEARTS - QTDB) in milliseconds', fontsize=11)
            ax1.set_ylabel('Frequency', fontsize=11)
            ax1.set_title(f'{subject} - R Peak Timing Offset Distribution', fontsize=12, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # Amplitude difference histogram
            ax2 = axes[0, 1]
            ax2.hist(matches_df['amplitude_diff'], bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
            ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
            ax2.axvline(matches_df['amplitude_diff'].mean(), color='green', linestyle='--', 
                       linewidth=2, label=f'Mean: {matches_df["amplitude_diff"].mean():.3f}')
            ax2.axvline(matches_df['amplitude_diff'].median(), color='orange', linestyle='--',
                       linewidth=2, label=f'Median: {matches_df["amplitude_diff"].median():.3f}')
            ax2.set_xlabel('Amplitude Difference (PyHEARTS - QTDB)', fontsize=11)
            ax2.set_ylabel('Frequency', fontsize=11)
            ax2.set_title(f'{subject} - R Peak Amplitude Difference Distribution', fontsize=12, fontweight='bold')
            ax2.legend(loc='upper right', fontsize=9)
            ax2.grid(True, alpha=0.3)
            
            # Scatter plot: Timing vs Amplitude difference
            ax3 = axes[1, 0]
            scatter = ax3.scatter(matches_df['offset_ms'], matches_df['amplitude_diff'], 
                                 alpha=0.5, s=20, c=matches_df['qtdb_amplitude'], 
                                 cmap='viridis', edgecolors='black', linewidths=0.3)
            ax3.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax3.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax3.set_xlabel('Timing Offset (ms)', fontsize=11)
            ax3.set_ylabel('Amplitude Difference', fontsize=11)
            ax3.set_title(f'{subject} - Timing vs Amplitude Differences', fontsize=12, fontweight='bold')
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('QTDB Amplitude', fontsize=9)
            ax3.grid(True, alpha=0.3)
            
            # Box plot comparison
            ax4 = axes[1, 1]
            box_data = [matches_df['qtdb_amplitude'].values, matches_df['pyhearts_amplitude'].values]
            bp = ax4.boxplot(box_data, tick_labels=['QTDB', 'PyHEARTS'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightcoral')
            bp['boxes'][1].set_facecolor('lightblue')
            ax4.set_ylabel('Amplitude', fontsize=11)
            ax4.set_title(f'{subject} - Amplitude Distribution Comparison', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            # Save histogram figure
            output_path_hist = base_dir / f"{subject}_rpeak_stats.png"
            plt.savefig(output_path_hist, dpi=150, bbox_inches='tight')
            print(f"  Saved statistics plots to: {output_path_hist}")
            plt.close()
            
        else:
            print(f"\n  WARNING: No matching peaks found within {max_offset_samples} samples ({max_offset_samples/sampling_rate*1000:.0f} ms)")
            
    except Exception as e:
        print(f"\n  ERROR processing {subject}: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")

