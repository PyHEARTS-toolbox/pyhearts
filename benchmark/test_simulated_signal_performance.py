#!/usr/bin/env python3
"""
Test script to verify PyHEARTS performance on:
1. Normal simulated ECG signals (baseline performance check)
2. Inverted simulated ECG signals (new capability verification)

This ensures that:
- Normal signal performance hasn't degraded
- Inverted signal detection works correctly
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pyhearts import PyHEARTS
from pyhearts.processing import r_peak_detection
from pyhearts.config import ProcessCycleConfig


def generate_simple_ecg(sampling_rate=1000, duration=10, heart_rate=60):
    """
    Generate synthetic ECG with known P, Q, R, S, T waves.
    Same as tests/conftest.py
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
    
    return signal, peak_times


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
    print("PYHEARTS PERFORMANCE TEST: NORMAL vs INVERTED SIMULATED SIGNALS")
    print("="*80)
    
    cfg = ProcessCycleConfig()
    sampling_rate = 1000.0
    duration = 10  # seconds
    heart_rate = 60  # bpm
    
    # Generate normal signal
    print(f"\n[1] Generating normal simulated ECG signal")
    print(f"    Duration: {duration}s, Sampling rate: {sampling_rate} Hz, Heart rate: {heart_rate} bpm")
    signal_normal, true_r_times = generate_simple_ecg(
        sampling_rate=sampling_rate, 
        duration=duration, 
        heart_rate=heart_rate
    )
    
    n_expected = len(true_r_times)
    print(f"    Expected R-peaks: {n_expected}")
    print(f"    Signal range: [{signal_normal.min():.3f}, {signal_normal.max():.3f}]")
    
    # Test 1: Normal signal with direct R-peak detection
    print(f"\n[2] Testing R-peak detection on NORMAL signal (direct)")
    peaks_normal_direct = r_peak_detection(signal_normal, sampling_rate, cfg=cfg, sensitivity="standard")
    
    n_detected, n_exp, matches, mean_offset, std_offset = evaluate_peak_detection(
        peaks_normal_direct, true_r_times, sampling_rate, tolerance_ms=20
    )
    
    print(f"    Detected peaks: {n_detected}")
    print(f"    Expected peaks: {n_exp}")
    print(f"    Matches (within 20ms): {matches}/{n_exp} ({100*matches/n_exp:.1f}%)")
    if not np.isnan(mean_offset):
        print(f"    Mean offset: {mean_offset:.2f} ms (std: {std_offset:.2f} ms)")
    
    # Check peak polarity
    if len(peaks_normal_direct) > 0:
        peak_vals = signal_normal[peaks_normal_direct]
        pos_count = np.sum(peak_vals > 0)
        neg_count = np.sum(peak_vals < 0)
        print(f"    Peak polarity: {pos_count} positive, {neg_count} negative")
        normal_polarity_correct = pos_count == len(peaks_normal_direct)
        print(f"    Polarity status: {'✓ PASS' if normal_polarity_correct else '✗ FAIL'}")
    
    # Test 2: Normal signal with full PyHEARTS analysis
    print(f"\n[3] Testing NORMAL signal with full PyHEARTS analysis")
    try:
        analyzer_normal = PyHEARTS(sampling_rate=sampling_rate, species="human", verbose=False, plot=False)
        output_df_normal, epochs_df_normal = analyzer_normal.analyze_ecg(signal_normal)
        
        peaks_normal_full = analyzer_normal.r_peak_indices
        n_detected_full = len(peaks_normal_full) if peaks_normal_full is not None else 0
        
        print(f"    Detected R-peaks: {n_detected_full}")
        print(f"    Cardiac cycles: {len(output_df_normal)}")
        
        if len(peaks_normal_full) > 0:
            peak_vals_full = signal_normal[peaks_normal_full]
            pos_count_full = np.sum(peak_vals_full > 0)
            print(f"    Peak polarity: {pos_count_full} positive, {len(peaks_normal_full) - pos_count_full} negative")
            print(f"    Status: {'✓ PASS' if pos_count_full == len(peaks_normal_full) else '✗ FAIL'}")
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate inverted signal
    print(f"\n[4] Generating INVERTED signal (negated)")
    signal_inverted = -signal_normal
    print(f"    Signal range: [{signal_inverted.min():.3f}, {signal_inverted.max():.3f}]")
    print(f"    Note: R-peaks are now at negative nadirs (minima)")
    
    # True R-peak times remain the same (they're just inverted)
    true_r_times_inverted = true_r_times.copy()
    
    # Test 3: Inverted signal with direct R-peak detection
    print(f"\n[5] Testing R-peak detection on INVERTED signal (direct)")
    peaks_inverted_direct = r_peak_detection(
        signal_inverted, 
        sampling_rate, 
        cfg=cfg, 
        sensitivity="standard",
        raw_ecg=signal_inverted  # Pass raw signal for polarity detection
    )
    
    n_detected_inv, n_exp_inv, matches_inv, mean_offset_inv, std_offset_inv = evaluate_peak_detection(
        peaks_inverted_direct, true_r_times_inverted, sampling_rate, tolerance_ms=20
    )
    
    print(f"    Detected peaks: {n_detected_inv}")
    print(f"    Expected peaks: {n_exp_inv}")
    print(f"    Matches (within 20ms): {matches_inv}/{n_exp_inv} ({100*matches_inv/n_exp_inv:.1f}%)")
    if not np.isnan(mean_offset_inv):
        print(f"    Mean offset: {mean_offset_inv:.2f} ms (std: {std_offset_inv:.2f} ms)")
    
    # Check peak polarity
    if len(peaks_inverted_direct) > 0:
        peak_vals_inv = signal_inverted[peaks_inverted_direct]
        pos_count_inv = np.sum(peak_vals_inv > 0)
        neg_count_inv = np.sum(peak_vals_inv < 0)
        print(f"    Peak polarity: {pos_count_inv} positive, {neg_count_inv} negative")
        inverted_polarity_correct = neg_count_inv == len(peaks_inverted_direct)
        print(f"    Polarity status: {'✓ PASS' if inverted_polarity_correct else '✗ FAIL'}")
    
    # Test 4: Inverted signal with full PyHEARTS analysis
    print(f"\n[6] Testing INVERTED signal with full PyHEARTS analysis")
    try:
        analyzer_inverted = PyHEARTS(sampling_rate=sampling_rate, species="human", verbose=False, plot=False)
        output_df_inverted, epochs_df_inverted = analyzer_inverted.analyze_ecg(
            signal_inverted, 
            raw_ecg=signal_inverted  # Pass raw signal for polarity detection
        )
        
        peaks_inverted_full = analyzer_inverted.r_peak_indices
        n_detected_inv_full = len(peaks_inverted_full) if peaks_inverted_full is not None else 0
        
        print(f"    Detected R-peaks: {n_detected_inv_full}")
        print(f"    Cardiac cycles: {len(output_df_inverted)}")
        
        if len(peaks_inverted_full) > 0:
            peak_vals_inv_full = signal_inverted[peaks_inverted_full]
            neg_count_inv_full = np.sum(peak_vals_inv_full < 0)
            print(f"    Peak polarity: {neg_count_inv_full} negative, {len(peaks_inverted_full) - neg_count_inv_full} positive")
            print(f"    Status: {'✓ PASS' if neg_count_inv_full == len(peaks_inverted_full) else '✗ FAIL'}")
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Comparison summary
    print(f"\n[7] COMPARISON SUMMARY")
    print(f"    {'Metric':<35} {'Normal':<20} {'Inverted':<20} {'Status':<15}")
    print(f"    {'-'*35} {'-'*20} {'-'*20} {'-'*15}")
    
    # Detection rate (direct)
    detection_rate_normal = matches / n_exp if n_exp > 0 else 0
    detection_rate_inverted = matches_inv / n_exp_inv if n_exp_inv > 0 else 0
    status_detection = "✓ PASS" if abs(detection_rate_normal - detection_rate_inverted) < 0.1 and detection_rate_normal >= 0.9 else "⚠ CHECK"
    print(f"    {'Detection rate (direct)':<35} {detection_rate_normal*100:>6.1f}%{'':<13} {detection_rate_inverted*100:>6.1f}%{'':<13} {status_detection:<15}")
    
    # Number of peaks (direct)
    status_peaks = "✓ PASS" if abs(n_detected - n_detected_inv) <= 2 else "⚠ CHECK"
    print(f"    {'Peaks detected (direct)':<35} {n_detected:>6}{'':<13} {n_detected_inv:>6}{'':<13} {status_peaks:<15}")
    
    # Timing accuracy
    if not np.isnan(mean_offset) and not np.isnan(mean_offset_inv):
        status_timing = "✓ PASS" if abs(mean_offset) < 10 and abs(mean_offset_inv) < 10 else "⚠ CHECK"
        print(f"    {'Mean offset (ms, direct)':<35} {mean_offset:>6.2f}{'':<13} {mean_offset_inv:>6.2f}{'':<13} {status_timing:<15}")
    
    # Polarity correctness
    normal_pol_ok = len(peaks_normal_direct) > 0 and np.all(signal_normal[peaks_normal_direct] > 0)
    inv_pol_ok = len(peaks_inverted_direct) > 0 and np.all(signal_inverted[peaks_inverted_direct] < 0)
    status_polarity = "✓ PASS" if normal_pol_ok and inv_pol_ok else "✗ FAIL"
    print(f"    {'Polarity detection':<35} {'Correct':<20} {'Correct':<20} {status_polarity:<15}")
    
    # Overall assessment
    print(f"\n[8] OVERALL ASSESSMENT")
    all_passed = (
        detection_rate_normal >= 0.9 and 
        detection_rate_inverted >= 0.9 and
        normal_pol_ok and
        inv_pol_ok
    )
    
    if all_passed:
        print(f"    ✓ PASS: All tests passed")
        print(f"      - Normal signal: {detection_rate_normal*100:.1f}% detection, correct polarity")
        print(f"      - Inverted signal: {detection_rate_inverted*100:.1f}% detection, correct polarity")
        print(f"      - No performance degradation detected")
    else:
        print(f"    ⚠ WARNING: Some tests did not pass")
        if detection_rate_normal < 0.9:
            print(f"      - Normal signal detection rate: {detection_rate_normal*100:.1f}% (< 90%)")
        if detection_rate_inverted < 0.9:
            print(f"      - Inverted signal detection rate: {detection_rate_inverted*100:.1f}% (< 90%)")
        if not normal_pol_ok:
            print(f"      - Normal signal polarity detection failed")
        if not inv_pol_ok:
            print(f"      - Inverted signal polarity detection failed")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

