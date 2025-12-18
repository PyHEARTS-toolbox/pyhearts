#!/usr/bin/env python3
"""
Debug script to trace exactly why P and T waves aren't being detected
on a clean synthetic ECG signal.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/morganfitzgerald/Documents/pyhearts')

from pyhearts import PyHEARTS
from pyhearts.config import ProcessCycleConfig
from pyhearts.processing import r_peak_detection, epoch_ecg
import pandas as pd


def generate_test_ecg(sampling_rate=1000, duration=10, heart_rate=60):
    """Generate synthetic ECG with known P, Q, R, S, T waves."""
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    signal = np.zeros(n_samples)
    
    rr_interval = 60 / heart_rate
    peak_times = np.arange(0.5, duration - 0.5, rr_interval)
    
    for peak_time in peak_times:
        r_peak = 1.0 * np.exp(-((t - peak_time) ** 2) / (2 * 0.01 ** 2))
        p_wave = 0.15 * np.exp(-((t - (peak_time - 0.16)) ** 2) / (2 * 0.02 ** 2))
        q_wave = -0.1 * np.exp(-((t - (peak_time - 0.04)) ** 2) / (2 * 0.01 ** 2))
        s_wave = -0.2 * np.exp(-((t - (peak_time + 0.04)) ** 2) / (2 * 0.01 ** 2))
        t_wave = 0.3 * np.exp(-((t - (peak_time + 0.25)) ** 2) / (2 * 0.04 ** 2))
        signal += r_peak + p_wave + q_wave + s_wave + t_wave
    
    return signal, t, sampling_rate, peak_times


def main():
    print("="*70)
    print("DEBUGGING P/T WAVE DETECTION ON CLEAN SYNTHETIC ECG")
    print("="*70)
    
    # Generate signal
    signal, time, fs, true_r_times = generate_test_ecg(duration=10, heart_rate=60)
    cfg = ProcessCycleConfig()
    
    print(f"\n[1] SIGNAL CHARACTERISTICS")
    print(f"    Duration: {len(signal)/fs:.1f}s, Samples: {len(signal)}, Fs: {fs} Hz")
    print(f"    True R-peak times: {true_r_times[:5]}... ({len(true_r_times)} total)")
    print(f"    Signal range: [{signal.min():.3f}, {signal.max():.3f}] mV")
    
    # Step 1: R-peak detection
    print(f"\n[2] R-PEAK DETECTION")
    r_peaks = r_peak_detection(signal, fs, cfg=cfg)
    print(f"    Detected R-peaks: {len(r_peaks)}")
    print(f"    First 5 R-peak indices: {r_peaks[:5]}")
    print(f"    First 5 R-peak times: {r_peaks[:5]/fs}")
    
    # Check if R-peaks match true R-peaks
    true_r_indices = (true_r_times * fs).astype(int)
    print(f"    True R-peak indices: {true_r_indices[:5]}")
    
    # Why so many R-peaks? (87 detected vs 9 expected)
    print(f"\n    ⚠️  Expected ~{len(true_r_times)} R-peaks, detected {len(r_peaks)}")
    if len(r_peaks) > len(true_r_times) * 2:
        print(f"    → PROBLEM: Detecting too many peaks (likely P/T waves as R-peaks)")
        
        # Analyze what we're detecting
        peak_heights = signal[r_peaks]
        print(f"\n    Peak height distribution:")
        print(f"      Max: {peak_heights.max():.3f} mV (should be R-peaks)")
        print(f"      Min: {peak_heights.min():.3f} mV")
        print(f"      Mean: {peak_heights.mean():.3f} mV")
        
        # Categorize peaks by height
        r_like = peak_heights > 0.5  # R-peaks should be ~1.0 mV
        t_like = (peak_heights > 0.2) & (peak_heights < 0.5)  # T-waves ~0.3 mV
        p_like = (peak_heights > 0.1) & (peak_heights < 0.2)  # P-waves ~0.15 mV
        other = peak_heights <= 0.1
        
        print(f"\n    Peak categories by amplitude:")
        print(f"      R-like (>0.5 mV): {r_like.sum()} peaks")
        print(f"      T-like (0.2-0.5 mV): {t_like.sum()} peaks")
        print(f"      P-like (0.1-0.2 mV): {p_like.sum()} peaks")
        print(f"      Other (<0.1 mV): {other.sum()} peaks")
    
    # Step 2: Epoching
    print(f"\n[3] EPOCHING")
    epochs_df, energy = epoch_ecg(signal, r_peaks, fs, cfg=cfg, verbose=True)
    
    if epochs_df.empty:
        print("    ❌ Epoching returned empty DataFrame!")
        return
    
    n_cycles = epochs_df['cycle'].nunique()
    print(f"    Epochs created: {n_cycles}")
    
    # Step 3: Analyze a single epoch
    print(f"\n[4] ANALYZING SINGLE EPOCH")
    cycle_0 = epochs_df[epochs_df['cycle'] == epochs_df['cycle'].unique()[0]]
    print(f"    Cycle 0 shape: {cycle_0.shape}")
    print(f"    Cycle 0 index range: {cycle_0['index'].min()} - {cycle_0['index'].max()}")
    
    cycle_signal = cycle_0['signal_y'].values
    print(f"    Cycle signal range: [{cycle_signal.min():.3f}, {cycle_signal.max():.3f}]")
    
    # Check if P and T waves are visible in the epoch
    r_idx_in_cycle = np.argmax(cycle_signal)
    r_height = cycle_signal[r_idx_in_cycle]
    print(f"\n    R-peak in cycle: index={r_idx_in_cycle}, height={r_height:.3f}")
    
    # Look for P-wave (before R)
    pre_r = cycle_signal[:r_idx_in_cycle]
    if len(pre_r) > 0:
        p_idx = np.argmax(pre_r)
        p_height = pre_r[p_idx]
        print(f"    P-wave candidate: index={p_idx}, height={p_height:.3f}")
        print(f"    P/R ratio: {p_height/r_height:.3f} (expected ~0.15)")
    else:
        print(f"    ❌ No samples before R-peak in epoch!")
    
    # Look for T-wave (after R)
    post_r = cycle_signal[r_idx_in_cycle:]
    if len(post_r) > 50:  # Need some samples after R
        # Skip first 50ms (S-wave region)
        t_region = post_r[50:]
        if len(t_region) > 0:
            t_idx = np.argmax(t_region) + r_idx_in_cycle + 50
            t_height = cycle_signal[t_idx]
            print(f"    T-wave candidate: index={t_idx}, height={t_height:.3f}")
            print(f"    T/R ratio: {t_height/r_height:.3f} (expected ~0.30)")
    else:
        print(f"    ❌ Not enough samples after R-peak for T-wave!")
        print(f"    Post-R samples: {len(post_r)}")
    
    # Step 4: Run full analysis
    print(f"\n[5] RUNNING FULL PyHEARTS ANALYSIS")
    analyzer = PyHEARTS(sampling_rate=fs, cfg=cfg, verbose=True, plot=False)
    output_df, epochs_df = analyzer.analyze_ecg(signal)
    
    print(f"\n[6] RESULTS SUMMARY")
    print(f"    Output DataFrame shape: {output_df.shape}")
    
    if not output_df.empty:
        for wave in ['P', 'R', 'T']:
            col = f'{wave}_global_center_idx'
            if col in output_df.columns:
                valid = output_df[col].notna().sum()
                total = len(output_df)
                print(f"    {wave}-waves detected: {valid}/{total} ({100*valid/total:.1f}%)")
                
                # Show some values
                if valid > 0:
                    heights = output_df[f'{wave}_gauss_height'].dropna()
                    if len(heights) > 0:
                        print(f"      Heights: mean={heights.mean():.3f}, std={heights.std():.3f}")
    
    # Step 5: Check SNR gate
    print(f"\n[7] SNR GATE ANALYSIS")
    print(f"    cfg.snr_mad_multiplier: {cfg.snr_mad_multiplier}")
    print(f"    cfg.amp_min_ratio: {cfg.amp_min_ratio}")
    
    # Manually check if P and T would pass SNR gate
    if len(pre_r) > 0:
        mad = 1.4826 * np.median(np.abs(pre_r - np.median(pre_r)))
        k = cfg.snr_mad_multiplier.get('P', 2.0)
        threshold = k * mad
        print(f"\n    P-wave SNR check:")
        print(f"      P height: {p_height:.4f}")
        print(f"      MAD of pre-R region: {mad:.4f}")
        print(f"      Threshold (k={k}): {threshold:.4f}")
        print(f"      PASSES SNR gate: {p_height >= threshold}")
    
    # Check amplitude ratio
    print(f"\n    Amplitude ratio check:")
    print(f"      P/R ratio: {p_height/r_height:.3f}, min required: {cfg.amp_min_ratio.get('P', 0.03)}")
    print(f"      T/R ratio (expected): 0.30, min required: {cfg.amp_min_ratio.get('T', 0.05)}")


if __name__ == "__main__":
    main()

