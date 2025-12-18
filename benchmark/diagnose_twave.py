#!/usr/bin/env python3
"""
Diagnostic script to identify T-wave detection regression.
Compares old vs new config settings on simulated ECG.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/morganfitzgerald/Documents/pyhearts')

from pyhearts import PyHEARTS
from pyhearts.config import ProcessCycleConfig
from dataclasses import replace

# Generate a simple test ECG signal (similar to conftest.py)
def generate_test_ecg(sampling_rate=1000, duration=10, heart_rate=60):
    """Generate synthetic ECG with known P, Q, R, S, T waves."""
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    signal = np.zeros(n_samples)
    
    rr_interval = 60 / heart_rate
    peak_times = np.arange(0.5, duration - 0.5, rr_interval)
    
    for peak_time in peak_times:
        # R-peak: tall, narrow Gaussian
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
    
    return signal, sampling_rate, len(peak_times)


def count_detected_waves(output_df):
    """Count how many P, R, T waves were detected."""
    counts = {}
    for wave in ['P', 'R', 'T']:
        col = f'{wave}_global_center_idx'
        if col in output_df.columns:
            valid = output_df[col].notna().sum()
            counts[wave] = valid
        else:
            counts[wave] = 0
    return counts


def run_test(config_name, cfg, signal, sampling_rate, expected_beats):
    """Run PyHEARTS with given config and report results."""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"{'='*60}")
    
    analyzer = PyHEARTS(sampling_rate=sampling_rate, cfg=cfg, verbose=False, plot=False)
    output_df, epochs_df = analyzer.analyze_ecg(signal)
    
    if output_df.empty:
        print(f"  ❌ No output - analysis failed")
        return {'P': 0, 'R': 0, 'T': 0}
    
    counts = count_detected_waves(output_df)
    
    print(f"  Expected beats: {expected_beats}")
    print(f"  Detected R-peaks: {len(analyzer.r_peak_indices) if hasattr(analyzer, 'r_peak_indices') else 'N/A'}")
    print(f"  Valid epochs: {len(output_df)}")
    print(f"  P-waves detected: {counts['P']} ({100*counts['P']/max(1,len(output_df)):.1f}%)")
    print(f"  R-waves detected: {counts['R']} ({100*counts['R']/max(1,len(output_df)):.1f}%)")
    print(f"  T-waves detected: {counts['T']} ({100*counts['T']/max(1,len(output_df)):.1f}%)")
    
    return counts


def main():
    print("="*60)
    print("T-WAVE DETECTION DIAGNOSTIC")
    print("="*60)
    
    # Generate test signal
    signal, sampling_rate, expected_beats = generate_test_ecg(duration=30, heart_rate=60)
    print(f"\nGenerated ECG: {len(signal)} samples, {expected_beats} expected beats")
    
    # Config 1: OLD defaults (before my changes)
    old_config = replace(
        ProcessCycleConfig(),
        rpeak_prominence_multiplier=3.0,  # old default
        rpeak_rr_frac_second_pass=0.55,   # old default
        epoch_corr_thresh=0.80,           # old default
        epoch_var_thresh=5.0,             # old default
        threshold_fraction=0.30,          # old default
        snr_mad_multiplier={"P": 2.0, "T": 2.0},  # old default
        wavelet_guard_cap_ms=120,         # old default
    )
    old_results = run_test("OLD Defaults (pre-changes)", old_config, signal, sampling_rate, expected_beats)
    
    # Config 2: NEW defaults (my changes)
    new_config = ProcessCycleConfig()  # Current defaults
    new_results = run_test("NEW Defaults (post-changes)", new_config, signal, sampling_rate, expected_beats)
    
    # Config 3: NEW human preset
    human_config = ProcessCycleConfig.for_human()
    human_results = run_test("NEW Human Preset", human_config, signal, sampling_rate, expected_beats)
    
    # Config 4: Test with ONLY threshold_fraction change
    only_threshold = replace(
        ProcessCycleConfig(),
        rpeak_prominence_multiplier=3.0,
        rpeak_rr_frac_second_pass=0.55,
        epoch_corr_thresh=0.80,
        epoch_var_thresh=5.0,
        threshold_fraction=0.15,  # ONLY this changed
        snr_mad_multiplier={"P": 2.0, "T": 2.0},
        wavelet_guard_cap_ms=120,
    )
    threshold_results = run_test("OLD + threshold_fraction=0.15", only_threshold, signal, sampling_rate, expected_beats)
    
    # Config 5: Test with ONLY SNR change
    only_snr = replace(
        ProcessCycleConfig(),
        rpeak_prominence_multiplier=3.0,
        rpeak_rr_frac_second_pass=0.55,
        epoch_corr_thresh=0.80,
        epoch_var_thresh=5.0,
        threshold_fraction=0.30,
        snr_mad_multiplier={"P": 1.5, "T": 1.5},  # ONLY this changed
        wavelet_guard_cap_ms=120,
    )
    snr_results = run_test("OLD + snr_mad_multiplier=1.5", only_snr, signal, sampling_rate, expected_beats)
    
    # Config 6: Test with ONLY wavelet guard cap change
    only_wavelet = replace(
        ProcessCycleConfig(),
        rpeak_prominence_multiplier=3.0,
        rpeak_rr_frac_second_pass=0.55,
        epoch_corr_thresh=0.80,
        epoch_var_thresh=5.0,
        threshold_fraction=0.30,
        snr_mad_multiplier={"P": 2.0, "T": 2.0},
        wavelet_guard_cap_ms=100,  # ONLY this changed
    )
    wavelet_results = run_test("OLD + wavelet_guard_cap_ms=100", only_wavelet, signal, sampling_rate, expected_beats)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - T-wave detection count by config:")
    print("="*60)
    print(f"  OLD Defaults:              {old_results['T']} T-waves")
    print(f"  NEW Defaults:              {new_results['T']} T-waves")
    print(f"  NEW Human Preset:          {human_results['T']} T-waves")
    print(f"  OLD + threshold=0.15:      {threshold_results['T']} T-waves")
    print(f"  OLD + snr_mad=1.5:         {snr_results['T']} T-waves")
    print(f"  OLD + wavelet_cap=100:     {wavelet_results['T']} T-waves")
    
    # Identify the culprit
    print("\n" + "="*60)
    print("DIAGNOSIS:")
    print("="*60)
    if new_results['T'] < old_results['T'] * 0.5:
        print("❌ NEW defaults have significant T-wave regression!")
        if threshold_results['T'] < old_results['T'] * 0.5:
            print("  → threshold_fraction=0.15 is causing T-wave loss")
        if snr_results['T'] < old_results['T'] * 0.5:
            print("  → snr_mad_multiplier=1.5 is causing T-wave loss")
        if wavelet_results['T'] < old_results['T'] * 0.5:
            print("  → wavelet_guard_cap_ms=100 is causing T-wave loss")
    else:
        print("✓ T-wave detection looks OK on simulated data")
        print("  Issue may be specific to real QTDB data")


if __name__ == "__main__":
    main()

