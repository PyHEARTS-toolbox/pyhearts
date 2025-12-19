#!/usr/bin/env python3
"""
Visualize ECG signals with QTDB and PyHEARTS R peak annotations overlaid.

Creates detailed visualizations showing:
1. Multiple signal segments with both annotations
2. Zoomed views showing timing and amplitude differences
3. Clear markers distinguishing QTDB vs PyHEARTS peaks
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
print("VISUALIZING R PEAK DIFFERENCES")
print("="*80)

for subject in subjects:
    print(f"\nProcessing {subject}...")
    
    record_path = qtdb_dir / subject
    
    try:
        # Load signal
        signals, fields = wfdb.rdsamp(str(record_path))
        signal_names = fields.get("sig_name", [])
        sampling_rate = fields.get("fs", None)
        
        lead_idx = get_preferred_lead_index(signal_names)
        lead_name = signal_names[lead_idx] if lead_idx < len(signal_names) else "Unknown"
        ecg_signal = signals[:, lead_idx]
        
        # Load QTDB annotations
        ann = wfdb.rdann(str(record_path), 'atr')
        qtdb_r_peaks = np.array(ann.sample)
        qtdb_r_amplitudes = ecg_signal[qtdb_r_peaks]
        
        # Load PyHEARTS results
        pyhearts_csv = results_dir / f"{subject}_pyhearts.csv"
        df = pd.read_csv(pyhearts_csv)
        pyhearts_r_peaks = df['R_global_center_idx'].dropna().values.astype(int)
        pyhearts_r_peaks = pyhearts_r_peaks[(pyhearts_r_peaks >= 0) & (pyhearts_r_peaks < len(ecg_signal))]
        pyhearts_r_amplitudes = ecg_signal[pyhearts_r_peaks]
        
        # Find matches for connecting lines
        max_offset_samples = int(sampling_rate * 0.2)
        matches = []
        for qtdb_idx in qtdb_r_peaks:
            distances = np.abs(pyhearts_r_peaks - qtdb_idx)
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]
            if min_dist <= max_offset_samples:
                matches.append({
                    'qtdb_idx': qtdb_idx,
                    'pyhearts_idx': pyhearts_r_peaks[min_dist_idx]
                })
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Define regions to plot (start time in seconds, duration in seconds)
        regions = [
            (10, 3, "Early signal (10-13s)"),
            (60, 3, "Mid signal (60-63s)"),
            (120, 3, "Mid signal (120-123s)"),
        ]
        
        # Plot 3 overview regions
        for idx, (start_sec, duration, title) in enumerate(regions):
            start_sample = int(start_sec * sampling_rate)
            end_sample = int((start_sec + duration) * sampling_rate)
            end_sample = min(end_sample, len(ecg_signal))
            
            time_axis = np.arange(start_sample, end_sample) / sampling_rate
            region_signal = ecg_signal[start_sample:end_sample]
            
            ax = fig.add_subplot(gs[0, idx])
            
            # Plot signal
            ax.plot(time_axis, region_signal, 'k-', linewidth=1.0, alpha=0.7, label='ECG Signal')
            
            # Plot QTDB annotations
            qtdb_mask = (qtdb_r_peaks >= start_sample) & (qtdb_r_peaks < end_sample)
            qtdb_in_region = qtdb_r_peaks[qtdb_mask]
            if len(qtdb_in_region) > 0:
                qtdb_times = qtdb_in_region / sampling_rate
                qtdb_vals = ecg_signal[qtdb_in_region]
                ax.scatter(qtdb_times, qtdb_vals, c='red', marker='v', s=200, 
                          label='QTDB', zorder=5, edgecolors='darkred', linewidths=1.5, alpha=0.8)
            
            # Plot PyHEARTS detections
            pyhearts_mask = (pyhearts_r_peaks >= start_sample) & (pyhearts_r_peaks < end_sample)
            pyhearts_in_region = pyhearts_r_peaks[pyhearts_mask]
            if len(pyhearts_in_region) > 0:
                pyhearts_times = pyhearts_in_region / sampling_rate
                pyhearts_vals = ecg_signal[pyhearts_in_region]
                ax.scatter(pyhearts_times, pyhearts_vals, c='blue', marker='^', s=200,
                          label='PyHEARTS', zorder=5, edgecolors='darkblue', linewidths=1.5, alpha=0.8)
            
            # Draw connecting lines for matched peaks
            for match in matches:
                q_idx = match['qtdb_idx']
                p_idx = match['pyhearts_idx']
                if start_sample <= q_idx < end_sample or start_sample <= p_idx < end_sample:
                    q_time = q_idx / sampling_rate
                    p_time = p_idx / sampling_rate
                    q_val = ecg_signal[q_idx]
                    p_val = ecg_signal[p_idx]
                    ax.plot([q_time, p_time], [q_val, p_val], 
                           'g--', alpha=0.5, linewidth=1.5, zorder=3)
            
            ax.set_xlabel('Time (seconds)', fontsize=10)
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Create zoomed-in views showing individual beats
        # Find a few interesting beats to zoom into
        zoom_regions = []
        for i in range(3):
            if i < len(matches):
                match = matches[i * 100]  # Sample every 100th match
                center_idx = match['qtdb_idx']
                # Show 0.5 seconds around this beat
                start_sample = max(0, center_idx - int(0.25 * sampling_rate))
                end_sample = min(len(ecg_signal), center_idx + int(0.25 * sampling_rate))
                zoom_regions.append((start_sample, end_sample, f"Beat {i+1} (zoom)"))
        
        for idx, (start_sample, end_sample, title) in enumerate(zoom_regions):
            time_axis = np.arange(start_sample, end_sample) / sampling_rate
            region_signal = ecg_signal[start_sample:end_sample]
            
            ax = fig.add_subplot(gs[1, idx])
            
            # Plot signal with thicker line
            ax.plot(time_axis, region_signal, 'k-', linewidth=1.5, alpha=0.8, label='ECG Signal')
            
            # Plot QTDB annotations
            qtdb_mask = (qtdb_r_peaks >= start_sample) & (qtdb_r_peaks < end_sample)
            qtdb_in_region = qtdb_r_peaks[qtdb_mask]
            if len(qtdb_in_region) > 0:
                qtdb_times = qtdb_in_region / sampling_rate
                qtdb_vals = ecg_signal[qtdb_in_region]
                ax.scatter(qtdb_times, qtdb_vals, c='red', marker='v', s=300, 
                          label='QTDB (QRS onset)', zorder=6, edgecolors='darkred', 
                          linewidths=2, alpha=0.9)
            
            # Plot PyHEARTS detections
            pyhearts_mask = (pyhearts_r_peaks >= start_sample) & (pyhearts_r_peaks < end_sample)
            pyhearts_in_region = pyhearts_r_peaks[pyhearts_mask]
            if len(pyhearts_in_region) > 0:
                pyhearts_times = pyhearts_in_region / sampling_rate
                pyhearts_vals = ecg_signal[pyhearts_in_region]
                ax.scatter(pyhearts_times, pyhearts_vals, c='blue', marker='^', s=300,
                          label='PyHEARTS (R peak)', zorder=6, edgecolors='darkblue', 
                          linewidths=2, alpha=0.9)
            
            # Draw connecting lines with annotation
            for match in matches:
                q_idx = match['qtdb_idx']
                p_idx = match['pyhearts_idx']
                if start_sample <= q_idx < end_sample:
                    q_time = q_idx / sampling_rate
                    p_time = p_idx / sampling_rate
                    q_val = ecg_signal[q_idx]
                    p_val = ecg_signal[p_idx]
                    
                    # Draw line
                    ax.plot([q_time, p_time], [q_val, p_val], 
                           'g-', alpha=0.7, linewidth=2, zorder=4)
                    
                    # Add text annotation showing offset
                    offset_ms = (p_idx - q_idx) / sampling_rate * 1000
                    mid_time = (q_time + p_time) / 2
                    mid_val = (q_val + p_val) / 2
                    ax.annotate(f'{offset_ms:.0f}ms', 
                              xy=(mid_time, mid_val),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, color='green', fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                      edgecolor='green', alpha=0.7))
            
            ax.set_xlabel('Time (seconds)', fontsize=10)
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Create single-beat detailed view
        if len(matches) > 0:
            # Pick a representative match
            match = matches[len(matches) // 2]
            center_idx = match['qtdb_idx']
            
            # Show wider view around this beat (1 second)
            start_sample = max(0, center_idx - int(0.5 * sampling_rate))
            end_sample = min(len(ecg_signal), center_idx + int(0.5 * sampling_rate))
            
            time_axis = np.arange(start_sample, end_sample) / sampling_rate
            region_signal = ecg_signal[start_sample:end_sample]
            
            ax = fig.add_subplot(gs[2, :])  # Full width
            
            # Plot signal
            ax.plot(time_axis, region_signal, 'k-', linewidth=2.0, alpha=0.8, label='ECG Signal')
            
            # Highlight the QRS complex region
            qrs_start = match['qtdb_idx'] - int(0.05 * sampling_rate)
            qrs_end = match['pyhearts_idx'] + int(0.05 * sampling_rate)
            qrs_time_start = qrs_start / sampling_rate
            qrs_time_end = qrs_end / sampling_rate
            if qrs_start >= start_sample and qrs_end <= end_sample:
                ax.axvspan(qrs_time_start, qrs_time_end, alpha=0.2, color='yellow', 
                          label='QRS Complex Region')
            
            # Plot QTDB annotation with large marker
            if start_sample <= match['qtdb_idx'] < end_sample:
                q_time = match['qtdb_idx'] / sampling_rate
                q_val = ecg_signal[match['qtdb_idx']]
                ax.scatter([q_time], [q_val], c='red', marker='v', s=500, 
                          label=f'QTDB (QRS onset)\nAmplitude: {q_val:.3f}', 
                          zorder=7, edgecolors='darkred', linewidths=2.5, alpha=0.9)
            
            # Plot PyHEARTS detection with large marker
            if start_sample <= match['pyhearts_idx'] < end_sample:
                p_time = match['pyhearts_idx'] / sampling_rate
                p_val = ecg_signal[match['pyhearts_idx']]
                ax.scatter([p_time], [p_val], c='blue', marker='^', s=500,
                          label=f'PyHEARTS (R peak)\nAmplitude: {p_val:.3f}', 
                          zorder=7, edgecolors='darkblue', linewidths=2.5, alpha=0.9)
                
                # Draw prominent connecting line
                ax.plot([match['qtdb_idx']/sampling_rate, p_time], 
                       [q_val, p_val], 
                       'g-', alpha=0.8, linewidth=3, zorder=5, label='Offset')
                
                # Add detailed annotation
                offset_ms = (match['pyhearts_idx'] - match['qtdb_idx']) / sampling_rate * 1000
                amp_diff = p_val - q_val
                mid_time = (match['qtdb_idx']/sampling_rate + p_time) / 2
                mid_val = (q_val + p_val) / 2
                
                ax.annotate(f'Timing: {offset_ms:.1f} ms\nAmplitude diff: {amp_diff:.3f}', 
                          xy=(mid_time, mid_val),
                          xytext=(10, 10), textcoords='offset points',
                          fontsize=11, color='green', fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                  edgecolor='green', linewidth=2, alpha=0.9))
            
            ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
            ax.set_title(f'{subject} - Detailed Single Beat View (QTDB vs PyHEARTS)', 
                        fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add summary statistics text
        ax_text = fig.add_subplot(gs[3, :])
        ax_text.axis('off')
        
        # Calculate statistics
        matches_df = pd.DataFrame(matches)
        if len(matches_df) > 0:
            timing_stats = {
                'mean': (matches_df['pyhearts_idx'] - matches_df['qtdb_idx']).mean() / sampling_rate * 1000,
                'median': (matches_df['pyhearts_idx'] - matches_df['qtdb_idx']).median() / sampling_rate * 1000,
                'std': (matches_df['pyhearts_idx'] - matches_df['qtdb_idx']).std() / sampling_rate * 1000,
            }
            
            amp_diffs = [ecg_signal[m['pyhearts_idx']] - ecg_signal[m['qtdb_idx']] for m in matches]
            amplitude_stats = {
                'mean': np.mean(amp_diffs),
                'median': np.median(amp_diffs),
                'std': np.std(amp_diffs),
            }
            
            summary_text = f"""
            Summary Statistics for {subject}:
            Lead: {lead_name} | Sampling Rate: {sampling_rate} Hz
            
            Timing Offset (PyHEARTS - QTDB):
            • Mean: {timing_stats['mean']:.2f} ms
            • Median: {timing_stats['median']:.2f} ms  
            • Std Dev: {timing_stats['std']:.2f} ms
            
            Amplitude Difference (PyHEARTS - QTDB):
            • Mean: {amplitude_stats['mean']:.3f}
            • Median: {amplitude_stats['median']:.3f}
            • Std Dev: {amplitude_stats['std']:.3f}
            
            Total Peaks: QTDB={len(qtdb_r_peaks)}, PyHEARTS={len(pyhearts_r_peaks)}, Matched={len(matches)}
            """
            ax_text.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                        verticalalignment='center', bbox=dict(boxstyle='round', 
                        facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'{subject} - R Peak Comparison: QTDB vs PyHEARTS', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save figure
        output_path = base_dir / f"{subject}_rpeak_visualization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to: {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"  ERROR processing {subject}: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print("VISUALIZATION COMPLETE")
print(f"{'='*80}")


