git #!/usr/bin/env python3
"""
Generate comparison plot showing:
1. Inverted signal with PyHEARTS detected peaks
2. Normal signal with PyHEARTS detected peaks
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pyhearts.processing import r_peak_detection
from pyhearts.config import ProcessCycleConfig


def generate_simple_ecg(sampling_rate=1000, duration=10, heart_rate=60):
    """
    Generate synthetic ECG with known P, Q, R, S, T waves.
    """
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    signal = np.zeros(n_samples)
    
    rr_interval = 60 / heart_rate
    peak_times = np.arange(0.5, duration - 0.5, rr_interval)
    
    for peak_time in peak_times:
        # R-peak: tall, narrow Gaussian (positive)
        r_peak = 1.0 * np.exp(-((t - peak_time) ** 2) / (2 * 0.01 ** 2))
        
        # P-wave: small positive, before R
        p_wave = 0.15 * np.exp(-((t - (peak_time - 0.16)) ** 2) / (2 * 0.02 ** 2))
        
        # Q-wave: small negative, just before R
        q_wave = -0.1 * np.exp(-((t - (peak_time - 0.04)) ** 2) / (2 * 0.01 ** 2))
        
        # S-wave: negative, just after R
        s_wave = -0.2 * np.exp(-((t - (peak_time + 0.04)) ** 2) / (2 * 0.01 ** 2))
        
        # T-wave: positive, after QRS
        t_wave = 0.3 * np.exp(-((t - (peak_time + 0.25)) ** 2) / (2 * 0.04 ** 2))
        
        signal += r_peak + p_wave + q_wave + s_wave + t_wave
    
    return signal, t, peak_times


def main():
    print("Generating comparison plot...")
    
    cfg = ProcessCycleConfig()
    sampling_rate = 1000.0
    duration = 10  # seconds
    heart_rate = 60  # bpm
    
    # Generate normal signal
    signal_normal, time, true_peak_times = generate_simple_ecg(
        sampling_rate=sampling_rate,
        duration=duration,
        heart_rate=heart_rate
    )
    
    # Generate inverted signal
    signal_inverted = -signal_normal
    
    # Detect peaks in both signals
    print("Detecting peaks in normal signal...")
    peaks_normal = r_peak_detection(signal_normal, sampling_rate, cfg=cfg, sensitivity="standard")
    
    print("Detecting peaks in inverted signal...")
    peaks_inverted = r_peak_detection(
        signal_inverted, 
        sampling_rate, 
        cfg=cfg, 
        sensitivity="standard",
        raw_ecg=signal_inverted
    )
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Inverted signal
    ax1.plot(time, signal_inverted, 'b-', linewidth=1.5, label='Inverted ECG Signal', alpha=0.7)
    if len(peaks_inverted) > 0:
        ax1.scatter(
            time[peaks_inverted], 
            signal_inverted[peaks_inverted], 
            color='red', 
            marker='x', 
            s=100, 
            linewidths=2,
            zorder=5,
            label=f'PyHEARTS R-peaks ({len(peaks_inverted)})'
        )
    ax1.set_ylabel('Amplitude (mV)', fontsize=12)
    ax1.set_title('Inverted Signal with PyHEARTS Detected R-peaks', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Plot 2: Normal signal
    ax2.plot(time, signal_normal, 'b-', linewidth=1.5, label='Normal ECG Signal', alpha=0.7)
    if len(peaks_normal) > 0:
        ax2.scatter(
            time[peaks_normal], 
            signal_normal[peaks_normal], 
            color='red', 
            marker='x', 
            s=100, 
            linewidths=2,
            zorder=5,
            label=f'PyHEARTS R-peaks ({len(peaks_normal)})'
        )
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Amplitude (mV)', fontsize=12)
    ax2.set_title('Normal Signal with PyHEARTS Detected R-peaks', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Add summary text
    summary_text = (
        f"Normal signal: {len(peaks_normal)} peaks detected, "
        f"{np.sum(signal_normal[peaks_normal] > 0) if len(peaks_normal) > 0 else 0} positive\n"
        f"Inverted signal: {len(peaks_inverted)} peaks detected, "
        f"{np.sum(signal_inverted[peaks_inverted] < 0) if len(peaks_inverted) > 0 else 0} negative\n"
        f"Expected: {len(true_peak_times)} peaks"
    )
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    # Save figure
    output_path = project_root / "benchmark" / "normal_vs_inverted_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Also show zoomed view of first few seconds
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Zoom to first 3 seconds
    zoom_end = 3.0
    zoom_mask = time <= zoom_end
    
    # Plot 1: Inverted signal (zoomed)
    ax3.plot(time[zoom_mask], signal_inverted[zoom_mask], 'b-', linewidth=1.5, label='Inverted ECG Signal', alpha=0.7)
    peaks_in_zoom_inv = peaks_inverted[time[peaks_inverted] <= zoom_end]
    if len(peaks_in_zoom_inv) > 0:
        ax3.scatter(
            time[peaks_in_zoom_inv], 
            signal_inverted[peaks_in_zoom_inv], 
            color='red', 
            marker='x', 
            s=150, 
            linewidths=3,
            zorder=5,
            label=f'PyHEARTS R-peaks ({len(peaks_in_zoom_inv)})'
        )
    ax3.set_ylabel('Amplitude (mV)', fontsize=12)
    ax3.set_title('Inverted Signal (Zoomed: 0-3 seconds)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Plot 2: Normal signal (zoomed)
    ax4.plot(time[zoom_mask], signal_normal[zoom_mask], 'b-', linewidth=1.5, label='Normal ECG Signal', alpha=0.7)
    peaks_in_zoom_norm = peaks_normal[time[peaks_normal] <= zoom_end]
    if len(peaks_in_zoom_norm) > 0:
        ax4.scatter(
            time[peaks_in_zoom_norm], 
            signal_normal[peaks_in_zoom_norm], 
            color='red', 
            marker='x', 
            s=150, 
            linewidths=3,
            zorder=5,
            label=f'PyHEARTS R-peaks ({len(peaks_in_zoom_norm)})'
        )
    ax4.set_xlabel('Time (seconds)', fontsize=12)
    ax4.set_ylabel('Amplitude (mV)', fontsize=12)
    ax4.set_title('Normal Signal (Zoomed: 0-3 seconds)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=10)
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    
    # Save zoomed figure
    output_path_zoom = project_root / "benchmark" / "normal_vs_inverted_zoomed.png"
    plt.savefig(output_path_zoom, dpi=150, bbox_inches='tight')
    print(f"Zoomed plot saved to: {output_path_zoom}")
    
    print("\nPlot generation complete!")
    print(f"  - Full view: {output_path}")
    print(f"  - Zoomed view: {output_path_zoom}")


if __name__ == "__main__":
    main()

