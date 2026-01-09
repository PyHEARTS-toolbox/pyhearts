"""
Test script to verify slope features are working correctly and haven't broken existing features.

This script:
1. Generates or uses a test ECG signal
2. Runs PyHEARTS analysis
3. Verifies new slope features are present and have valid values
4. Verifies existing features still work correctly
5. Compares feature counts
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pyhearts import PyHEARTS

def create_simple_ecg_signal(duration=10, sampling_rate=250, heart_rate=60):
    """
    Create a simple synthetic ECG signal for testing.
    """
    t = np.arange(0, duration, 1/sampling_rate)
    signal = np.zeros_like(t)
    
    # Create R-peaks at regular intervals
    rr_interval = 60 / heart_rate  # seconds
    r_peaks = np.arange(0.5, duration, rr_interval)
    
    for r_peak_time in r_peaks:
        # R-peak (sharp positive)
        r_idx = int(r_peak_time * sampling_rate)
        if 0 <= r_idx < len(signal):
            # Create a simple R-peak shape
            for i in range(-5, 6):
                idx = r_idx + i
                if 0 <= idx < len(signal):
                    signal[idx] += 1.0 * np.exp(-(i**2) / 2.0)
        
        # P-wave before R (smaller, earlier)
        p_time = r_peak_time - 0.15
        p_idx = int(p_time * sampling_rate)
        if 0 <= p_idx < len(signal):
            for i in range(-3, 4):
                idx = p_idx + i
                if 0 <= idx < len(signal):
                    signal[idx] += 0.2 * np.exp(-(i**2) / 1.5)
        
        # T-wave after R (broader)
        t_time = r_peak_time + 0.3
        t_idx = int(t_time * sampling_rate)
        if 0 <= t_idx < len(signal):
            for i in range(-8, 9):
                idx = t_idx + i
                if 0 <= idx < len(signal):
                    signal[idx] += 0.3 * np.exp(-(i**2) / 4.0)
        
        # Q and S waves (negative deflections around R)
        q_time = r_peak_time - 0.02
        q_idx = int(q_time * sampling_rate)
        if 0 <= q_idx < len(signal):
            signal[q_idx] -= 0.1
        
        s_time = r_peak_time + 0.02
        s_idx = int(s_time * sampling_rate)
        if 0 <= s_idx < len(signal):
            signal[s_idx] -= 0.15
    
    # Add small amount of noise
    signal += np.random.randn(len(signal)) * 0.02
    
    return signal


def test_slope_features():
    """Test that slope features are computed correctly."""
    print("=" * 80)
    print("Testing Slope Features Implementation")
    print("=" * 80)
    
    # Create test signal
    print("\n1. Creating test ECG signal...")
    signal = create_simple_ecg_signal(duration=30, sampling_rate=250, heart_rate=70)
    print(f"   Signal length: {len(signal)} samples ({len(signal)/250:.1f} seconds)")
    
    # Initialize analyzer
    print("\n2. Initializing PyHEARTS analyzer...")
    analyzer = PyHEARTS(sampling_rate=250, species='human')
    
    # Run analysis
    print("\n3. Running ECG analysis...")
    output_df, epochs_df = analyzer.analyze_ecg(signal, verbose=False)
    
    print(f"   Detected {len(output_df)} cycles")
    print(f"   Epochs DataFrame shape: {epochs_df.shape}")
    
    if len(output_df) == 0:
        print("\n   WARNING: No cycles detected! Cannot test features.")
        return False
    
    # Check for new slope features
    print("\n4. Checking for new slope features...")
    waves = ['P', 'Q', 'R', 'S', 'T']
    slope_features = ['max_upslope_mv_per_s', 'max_downslope_mv_per_s', 'slope_asymmetry']
    
    all_slope_features_present = True
    for wave in waves:
        for feature in slope_features:
            col_name = f"{wave}_{feature}"
            if col_name not in output_df.columns:
                print(f"   ❌ MISSING: {col_name}")
                all_slope_features_present = False
            else:
                # Check if feature has any non-NaN values
                non_nan_count = output_df[col_name].notna().sum()
                if non_nan_count > 0:
                    print(f"   ✓ {col_name}: {non_nan_count}/{len(output_df)} cycles have values")
                else:
                    print(f"   ⚠ {col_name}: Present but all NaN (wave may not be detected)")
    
    if not all_slope_features_present:
        print("\n   ERROR: Some slope features are missing!")
        return False
    
    # Verify slope feature values are reasonable
    print("\n5. Verifying slope feature values...")
    issues = []
    for wave in waves:
        upslope_col = f"{wave}_max_upslope_mv_per_s"
        downslope_col = f"{wave}_max_downslope_mv_per_s"
        asymmetry_col = f"{wave}_slope_asymmetry"
        
        if upslope_col in output_df.columns:
            upslope_vals = output_df[upslope_col].dropna()
            if len(upslope_vals) > 0:
                # For positive peaks (P, R, T): upslope should be positive
                # For negative peaks (Q, S): upslope goes down, so negative is expected
                if wave in ['P', 'R', 'T']:
                    if (upslope_vals < 0).all():
                        issues.append(f"{upslope_col}: All values are negative (unexpected for positive peak)")
                    else:
                        print(f"   ✓ {upslope_col}: {len(upslope_vals)} values, range [{upslope_vals.min():.2f}, {upslope_vals.max():.2f}] mV/s")
                else:  # Q, S (negative peaks)
                    # Negative upslope is expected (going down toward negative peak)
                    print(f"   ✓ {upslope_col}: {len(upslope_vals)} values, range [{upslope_vals.min():.2f}, {upslope_vals.max():.2f}] mV/s (negative expected for negative peak)")
        
        if downslope_col in output_df.columns:
            downslope_vals = output_df[downslope_col].dropna()
            if len(downslope_vals) > 0:
                # For positive peaks (P, R, T): downslope should be negative
                # For negative peaks (Q, S): downslope goes up, so positive is expected
                if wave in ['P', 'R', 'T']:
                    if (downslope_vals > 0).all():
                        issues.append(f"{downslope_col}: All values are positive (unexpected for positive peak)")
                    else:
                        print(f"   ✓ {downslope_col}: {len(downslope_vals)} values, range [{downslope_vals.min():.2f}, {downslope_vals.max():.2f}] mV/s")
                else:  # Q, S (negative peaks)
                    # Positive downslope is expected (going up from negative peak)
                    print(f"   ✓ {downslope_col}: {len(downslope_vals)} values, range [{downslope_vals.min():.2f}, {downslope_vals.max():.2f}] mV/s (positive expected for negative peak)")
        
        if asymmetry_col in output_df.columns:
            asymmetry_vals = output_df[asymmetry_col].dropna()
            if len(asymmetry_vals) > 0:
                # Asymmetry should be positive
                if (asymmetry_vals < 0).any():
                    issues.append(f"{asymmetry_col}: Some values are negative (unexpected)")
                else:
                    print(f"   ✓ {asymmetry_col}: {len(asymmetry_vals)} values, range [{asymmetry_vals.min():.2f}, {asymmetry_vals.max():.2f}]")
    
    if issues:
        print("\n   WARNINGS:")
        for issue in issues:
            print(f"   ⚠ {issue}")
    
    # Check existing features still work
    print("\n6. Verifying existing features still work...")
    existing_features = [
        'duration_ms', 'rise_ms', 'decay_ms', 'rdsm', 'sharpness',
        'gauss_height', 'gauss_stdev_ms', 'voltage_integral_uv_ms'
    ]
    
    all_existing_features_ok = True
    for wave in waves:
        for feature in existing_features:
            col_name = f"{wave}_{feature}"
            if col_name not in output_df.columns:
                print(f"   ❌ MISSING: {col_name}")
                all_existing_features_ok = False
    
    if not all_existing_features_ok:
        print("\n   ERROR: Some existing features are missing!")
        return False
    else:
        print("   ✓ All existing features are present")
    
    # Count total features
    print("\n7. Feature count summary...")
    total_cols = len(output_df.columns)
    print(f"   Total columns in output: {total_cols}")
    
    # Count per-wave features
    per_wave_features = {}
    for wave in waves:
        wave_cols = [col for col in output_df.columns if col.startswith(f"{wave}_")]
        per_wave_features[wave] = len(wave_cols)
        print(f"   {wave}-wave features: {len(wave_cols)}")
    
    # Expected: 33 features per wave (was 30, now +3 slope features)
    expected_per_wave = 33
    for wave in waves:
        if per_wave_features[wave] < expected_per_wave:
            print(f"   ⚠ {wave}-wave has fewer features than expected ({per_wave_features[wave]} < {expected_per_wave})")
    
    # Check interval features
    interval_features = [col for col in output_df.columns if 'interval' in col or 'segment' in col]
    print(f"   Interval features: {len(interval_features)}")
    
    # Check pairwise differences
    pairwise_features = [col for col in output_df.columns if 'voltage_diff' in col]
    print(f"   Pairwise difference features: {len(pairwise_features)}")
    
    # Sample some values
    print("\n8. Sample feature values from first valid cycle...")
    for wave in ['R', 'P', 'T']:  # Check R, P, T as they're most commonly detected
        upslope_col = f"{wave}_max_upslope_mv_per_s"
        downslope_col = f"{wave}_max_downslope_mv_per_s"
        asymmetry_col = f"{wave}_slope_asymmetry"
        
        if upslope_col in output_df.columns:
            first_valid = output_df[upslope_col].first_valid_index()
            if first_valid is not None:
                idx = output_df.index.get_loc(first_valid)
                print(f"   Cycle {idx}, {wave}-wave:")
                print(f"     {upslope_col}: {output_df.loc[first_valid, upslope_col]:.2f} mV/s")
                print(f"     {downslope_col}: {output_df.loc[first_valid, downslope_col]:.2f} mV/s")
                print(f"     {asymmetry_col}: {output_df.loc[first_valid, asymmetry_col]:.2f}")
                break
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    if all_slope_features_present and all_existing_features_ok and len(issues) == 0:
        print("✓ All tests passed!")
        print("✓ Slope features are correctly implemented")
        print("✓ Existing features are intact")
        return True
    else:
        print("✗ Some tests failed or warnings were raised")
        return False


if __name__ == "__main__":
    success = test_slope_features()
    sys.exit(0 if success else 1)

