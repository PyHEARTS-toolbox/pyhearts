#!/usr/bin/env python3
"""
Test script to verify that inverted signal detection doesn't degrade
performance on normal (non-inverted) simulated signals.

Compares R-peak detection accuracy on:
1. Normal simulated ECG signals
2. Inverted versions of the same signals
3. Verifies both detect the correct number of peaks at correct locations
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pyhearts.processing import r_peak_detection
from pyhearts.config import ProcessCycleConfig


def generate_test_ecg(sampling_rate=1000, duration=10, heart_rate=60):
    """
    Generate synthetic ECG with known P, Q, R, S, T waves.
    Returns signal, time array, sampling rate, and true R-peak times.
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
    
    return signal, t, sampling_rate, peak_times


def evaluate_peak_detection(detected_peaks, true_peak_times, sampling_rate, tolerance_ms=20):
    """
    Evaluate peak detection accuracy.
    
    Returns:
        - n_detected: number of detected peaks
        - n_expected: number of expected peaks
        - matches: number of peaks within tolerance
        - mean_offset_ms: mean timing offset in ms
        - std_offset_ms: std of timing offset in ms
    """
    n_expected = len(true_peak_times)
    n_detected = len(detected_peaks)
    
    if n_detected == 0:
        return n_detected, n_expected, 0, np.nan, np.nan
    
    # Convert detected peaks to times
    detected_times = detected_peaks / sampling_rate
    
    # Find matches: for each true peak, find closest detected peak
    tolerance_s = tolerance_ms / 1000.0
    matches = 0
    offsets = []
    
    for true_time in true_peak_times:
        # Find closest detected peak
        distances = np.abs(detected_times - true_time)
        min_dist = np.min(distances)
        closest_idx = np.argmin(distances)
        
        if min_dist <= tolerance_s:
            matches += 1
            offset_ms = (detected_times[closest_idx] - true_time) * 1000
            offsets.append(offset_ms)
    
    if len(offsets) > 0:
        mean_offset_ms = np.mean(offsets)
        std_offset_ms = np.std(offsets)
    else:
        mean_offset_ms = np.nan
        std_offset_ms = np.nan
    
    return n_detected, n_expected, matches, mean_offset_ms, std_offset_ms


def main():
    print("="*80)
    print("TESTING INVERTED SIGNAL DETECTION PERFORMANCE")
    print("="*80)
    
    cfg = ProcessCycleConfig()
    sampling_rate = 1000.0
    duration = 10  # seconds
    heart_rate = 60  # bpm
    
    # Generate normal signal
    print(f"\n[1] Generating normal simulated ECG signal")
    print(f"    Duration: {duration}s, Sampling rate: {sampling_rate} Hz, Heart rate: {heart_rate} bpm")
    signal_normal, t, fs, true_r_times = generate_test_ecg(
        sampling_rate=sampling_rate, 
        duration=duration, 
        heart_rate=heart_rate
    )
    
    n_expected = len(true_r_times)
    print(f"    Expected R-peaks: {n_expected}")
    print(f"    Signal range: [{signal_normal.min():.3f}, {signal_normal.max():.3f}]")
    
    # Test normal signal
    print(f"\n[2] Testing R-peak detection on NORMAL signal")
    peaks_normal = r_peak_detection(signal_normal, fs, cfg=cfg)
    
    n_detected, n_exp, matches, mean_offset, std_offset = evaluate_peak_detection(
        peaks_normal, true_r_times, fs, tolerance_ms=20
    )
    
    print(f"    Detected peaks: {n_detected}")
    print(f"    Expected peaks: {n_exp}")
    print(f"    Matches (within 20ms): {matches}/{n_exp} ({100*matches/n_exp:.1f}%)")
    if not np.isnan(mean_offset):
        print(f"    Mean offset: {mean_offset:.2f} ms (std: {std_offset:.2f} ms)")
    
    # Generate inverted signal (negate the normal signal)
    print(f"\n[3] Generating INVERTED signal (negated)")
    signal_inverted = -signal_normal
    print(f"    Signal range: [{signal_inverted.min():.3f}, {signal_inverted.max():.3f}]")
    print(f"    Note: R-peaks are now at negative nadirs (minima)")
    
    # True R-peak times remain the same (they're just inverted)
    true_r_times_inverted = true_r_times.copy()
    
    # Test inverted signal
    print(f"\n[4] Testing R-peak detection on INVERTED signal")
    peaks_inverted = r_peak_detection(signal_inverted, fs, cfg=cfg)
    
    n_detected_inv, n_exp_inv, matches_inv, mean_offset_inv, std_offset_inv = evaluate_peak_detection(
        peaks_inverted, true_r_times_inverted, fs, tolerance_ms=20
    )
    
    print(f"    Detected peaks: {n_detected_inv}")
    print(f"    Expected peaks: {n_exp_inv}")
    print(f"    Matches (within 20ms): {matches_inv}/{n_exp_inv} ({100*matches_inv/n_exp_inv:.1f}%)")
    if not np.isnan(mean_offset_inv):
        print(f"    Mean offset: {mean_offset_inv:.2f} ms (std: {std_offset_inv:.2f} ms)")
    
    # Comparison summary
    print(f"\n[5] COMPARISON SUMMARY")
    print(f"    {'Metric':<30} {'Normal':<15} {'Inverted':<15} {'Status':<15}")
    print(f"    {'-'*30} {'-'*15} {'-'*15} {'-'*15}")
    
    # Detection rate
    detection_rate_normal = matches / n_exp if n_exp > 0 else 0
    detection_rate_inverted = matches_inv / n_exp_inv if n_exp_inv > 0 else 0
    status_detection = "✓ PASS" if abs(detection_rate_normal - detection_rate_inverted) < 0.1 else "⚠ CHECK"
    print(f"    {'Detection rate':<30} {detection_rate_normal*100:>6.1f}%{'':<8} {detection_rate_inverted*100:>6.1f}%{'':<8} {status_detection:<15}")
    
    # Number of peaks
    status_peaks = "✓ PASS" if abs(n_detected - n_detected_inv) <= 2 else "⚠ CHECK"
    print(f"    {'Peaks detected':<30} {n_detected:>6}{'':<8} {n_detected_inv:>6}{'':<8} {status_peaks:<15}")
    
    # Timing accuracy
    if not np.isnan(mean_offset) and not np.isnan(mean_offset_inv):
        status_timing = "✓ PASS" if abs(mean_offset) < 10 and abs(mean_offset_inv) < 10 else "⚠ CHECK"
        print(f"    {'Mean offset (ms)':<30} {mean_offset:>6.2f}{'':<8} {mean_offset_inv:>6.2f}{'':<8} {status_timing:<15}")
    
    # Overall assessment
    print(f"\n[6] OVERALL ASSESSMENT")
    if detection_rate_normal >= 0.9 and detection_rate_inverted >= 0.9:
        print(f"    ✓ PASS: Both normal and inverted signals detected with >90% accuracy")
    elif detection_rate_normal >= 0.9:
        print(f"    ⚠ WARNING: Normal signals work well ({detection_rate_normal*100:.1f}%), but inverted detection needs improvement ({detection_rate_inverted*100:.1f}%)")
    elif detection_rate_inverted >= 0.9:
        print(f"    ⚠ WARNING: Inverted signals work well ({detection_rate_inverted*100:.1f}%), but normal detection degraded ({detection_rate_normal*100:.1f}%)")
    else:
        print(f"    ✗ FAIL: Both detection rates below 90%")
    
    # Verify inverted detection is working
    if n_detected_inv > 0:
        # Check if detected peaks are at minima (nadirs) in inverted signal
        peak_values = signal_inverted[peaks_inverted]
        min_value = np.min(peak_values)
        max_value = np.max(peak_values)
        
        print(f"\n[7] INVERTED SIGNAL VERIFICATION")
        print(f"    Peak values in inverted signal: min={min_value:.3f}, max={max_value:.3f}")
        if min_value < -0.5:  # Should be negative and large magnitude
            print(f"    ✓ PASS: Detected peaks are at negative nadirs (inverted R-peaks)")
        else:
            print(f"    ⚠ WARNING: Peak values don't look like inverted R-peaks")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

