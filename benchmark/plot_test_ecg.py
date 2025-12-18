#!/usr/bin/env python3
"""
Plot the simulated ECG test signal to verify it looks like Lead II.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/morganfitzgerald/Documents/pyhearts')


def generate_test_ecg(sampling_rate=1000, duration=10, heart_rate=60):
    """
    Generate synthetic ECG with known P, Q, R, S, T waves.
    This is the same signal used in conftest.py for testing.
    """
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    signal = np.zeros(n_samples)
    
    rr_interval = 60 / heart_rate
    peak_times = np.arange(0.5, duration - 0.5, rr_interval)
    
    for peak_time in peak_times:
        # R-peak: tall, narrow Gaussian (amplitude = 1.0 mV)
        r_peak = 1.0 * np.exp(-((t - peak_time) ** 2) / (2 * 0.01 ** 2))
        
        # P-wave: small positive, ~160ms before R (amplitude = 0.15 mV)
        p_wave = 0.15 * np.exp(-((t - (peak_time - 0.16)) ** 2) / (2 * 0.02 ** 2))
        
        # Q-wave: small negative, ~40ms before R (amplitude = -0.1 mV)
        q_wave = -0.1 * np.exp(-((t - (peak_time - 0.04)) ** 2) / (2 * 0.01 ** 2))
        
        # S-wave: negative, ~40ms after R (amplitude = -0.2 mV)
        s_wave = -0.2 * np.exp(-((t - (peak_time + 0.04)) ** 2) / (2 * 0.01 ** 2))
        
        # T-wave: positive, ~250ms after R (amplitude = 0.3 mV)
        t_wave = 0.3 * np.exp(-((t - (peak_time + 0.25)) ** 2) / (2 * 0.04 ** 2))
        
        signal += r_peak + p_wave + q_wave + s_wave + t_wave
    
    return signal, t, sampling_rate


def main():
    # Generate test signal
    signal, time, fs = generate_test_ecg(duration=10, heart_rate=60)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Full signal
    ax1 = axes[0]
    ax1.plot(time, signal, 'b-', linewidth=0.8)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (mV)')
    ax1.set_title('Full Simulated ECG Signal (10 seconds, 60 BPM)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 10])
    
    # Plot 2: Two cardiac cycles (zoomed)
    ax2 = axes[1]
    start_t, end_t = 1.0, 3.2  # ~2 cycles
    mask = (time >= start_t) & (time <= end_t)
    ax2.plot(time[mask], signal[mask], 'b-', linewidth=1.2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude (mV)')
    ax2.set_title('Two Cardiac Cycles (Zoomed)')
    ax2.grid(True, alpha=0.3)
    
    # Annotate the waves on the second cycle
    cycle_time = 1.5  # Center of first visible cycle
    annotations = [
        ('P', cycle_time - 0.16, 0.15, 'green'),
        ('Q', cycle_time - 0.04, -0.1, 'orange'),
        ('R', cycle_time, 1.0, 'red'),
        ('S', cycle_time + 0.04, -0.2, 'purple'),
        ('T', cycle_time + 0.25, 0.3, 'brown'),
    ]
    for label, t_pos, amp, color in annotations:
        ax2.annotate(label, xy=(t_pos, amp), xytext=(t_pos, amp + 0.15),
                    fontsize=12, fontweight='bold', color=color,
                    ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    
    # Plot 3: Single cardiac cycle with wave labels
    ax3 = axes[2]
    start_t, end_t = 1.3, 2.1  # Single cycle
    mask = (time >= start_t) & (time <= end_t)
    ax3.plot(time[mask], signal[mask], 'b-', linewidth=1.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude (mV)')
    ax3.set_title('Single Cardiac Cycle with PQRST Labels')
    ax3.grid(True, alpha=0.3)
    
    # Add shaded regions for each wave
    cycle_time = 1.5
    wave_regions = [
        ('P-wave', cycle_time - 0.22, cycle_time - 0.10, 'lightgreen', 0.3),
        ('QRS', cycle_time - 0.06, cycle_time + 0.06, 'lightyellow', 0.5),
        ('T-wave', cycle_time + 0.15, cycle_time + 0.35, 'lightcoral', 0.3),
    ]
    for label, t_start, t_end, color, alpha in wave_regions:
        ax3.axvspan(t_start, t_end, alpha=alpha, color=color, label=label)
    ax3.legend(loc='upper right')
    
    # Add amplitude reference lines
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=0.15, color='green', linestyle=':', alpha=0.5, label='P height')
    ax3.axhline(y=0.3, color='brown', linestyle=':', alpha=0.5, label='T height')
    
    plt.tight_layout()
    
    # Save and show
    output_path = '/Users/morganfitzgerald/Documents/pyhearts/benchmark/test_ecg_plot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()
    
    # Print wave characteristics
    print("\n" + "="*60)
    print("SIMULATED ECG WAVE CHARACTERISTICS")
    print("="*60)
    print(f"Sampling rate: {fs} Hz")
    print(f"Heart rate: 60 BPM (RR interval = 1000 ms)")
    print()
    print("Wave amplitudes (relative to baseline):")
    print(f"  P-wave:  +0.15 mV (15% of R)")
    print(f"  Q-wave:  -0.10 mV (10% of R)")
    print(f"  R-wave:  +1.00 mV (reference)")
    print(f"  S-wave:  -0.20 mV (20% of R)")
    print(f"  T-wave:  +0.30 mV (30% of R)")
    print()
    print("Wave timing (relative to R-peak):")
    print(f"  P-wave center:  -160 ms")
    print(f"  Q-wave center:  -40 ms")
    print(f"  R-wave center:  0 ms")
    print(f"  S-wave center:  +40 ms")
    print(f"  T-wave center:  +250 ms")
    print()
    print("Wave widths (σ of Gaussian):")
    print(f"  P-wave:  20 ms (FWHM ≈ 47 ms)")
    print(f"  Q-wave:  10 ms (FWHM ≈ 24 ms)")
    print(f"  R-wave:  10 ms (FWHM ≈ 24 ms)")
    print(f"  S-wave:  10 ms (FWHM ≈ 24 ms)")
    print(f"  T-wave:  40 ms (FWHM ≈ 94 ms)")
    print()
    print("Typical Lead II ECG characteristics:")
    print("  - P-wave: 0.1-0.25 mV, 80-120 ms duration")
    print("  - QRS: 0.5-2.0 mV, 80-120 ms duration")
    print("  - T-wave: 0.1-0.5 mV, 120-200 ms duration")
    print("  - PR interval: 120-200 ms")
    print("  - QT interval: 350-450 ms")


if __name__ == "__main__":
    main()

