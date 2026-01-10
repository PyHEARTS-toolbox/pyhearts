"""
Test script to verify ST segment features (elevation, slope, deviation) are working correctly.

This script:
1. Tests the ST segment calculation functions with known values
2. Verifies ST segment features are computed correctly
3. Tests integration with PyHEARTS
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pyhearts import PyHEARTS
from pyhearts.feature.st_segment import extract_st_segment_features


def create_simple_ecg_signal(duration=10, sampling_rate=250, heart_rate=60):
    """Create a simple synthetic ECG signal for testing."""
    t = np.arange(0, duration, 1/sampling_rate)
    signal = np.zeros_like(t)
    
    # Create R-peaks at regular intervals with some variability
    rr_interval = 60 / heart_rate  # seconds
    r_peaks = np.arange(0.5, duration, rr_interval)
    
    # Add variability to RR intervals
    np.random.seed(42)
    for i, r_peak_time in enumerate(r_peaks):
        # Add some variability (±5%)
        variability = np.random.normal(0, 0.02) * rr_interval
        r_peak_time += variability
        
        r_idx = int(r_peak_time * sampling_rate)
        if 0 <= r_idx < len(signal):
            for j in range(-5, 6):
                idx = r_idx + j
                if 0 <= idx < len(signal):
                    signal[idx] += 1.0 * np.exp(-(j**2) / 2.0)
    
    signal += np.random.randn(len(signal)) * 0.02
    return signal


def test_st_segment_calculations():
    """Test ST segment calculation functions with known values."""
    print("=" * 80)
    print("Testing ST Segment Calculation Functions")
    print("=" * 80)
    
    # Test case 1: Simple ST segment with elevation
    print("\n1. Testing with simple ST segment with elevation...")
    sampling_rate = 250.0
    signal = np.zeros(100)
    
    # Create a simple ST segment: flat baseline, then elevation
    # S end at index 40, T start at index 60
    s_ri_idx = 40
    t_le_idx = 60
    p_ri_idx = 20
    q_le_idx = 35
    
    # PR segment (baseline): flat at 0.0 mV
    signal[p_ri_idx:q_le_idx+1] = 0.0
    
    # ST segment: elevated at 0.1 mV
    signal[s_ri_idx:t_le_idx+1] = 0.1
    
    st_features = extract_st_segment_features(
        signal=signal,
        s_ri_idx=s_ri_idx,
        t_le_idx=t_le_idx,
        p_ri_idx=p_ri_idx,
        q_le_idx=q_le_idx,
        sampling_rate=sampling_rate,
        j_point_offset_ms=60.0,
        verbose=False,
    )
    
    # Check that features are computed
    assert not np.isnan(st_features['ST_elevation_mv']), "ST elevation should be computed"
    assert not np.isnan(st_features['ST_deviation_mv']), "ST deviation should be computed"
    
    # ST elevation should be around 0.1 mV (at J+60ms)
    # J+60ms = 40 + (60ms * 250Hz/1000) = 40 + 15 = 55
    # But if 55 > 60 (T start), it will be clamped to 60
    # So elevation should be at index 60 (T start) = 0.1 mV
    assert abs(st_features['ST_elevation_mv'] - 0.1) < 0.01, \
        f"ST elevation should be ~0.1 mV, got {st_features['ST_elevation_mv']}"
    
    # ST deviation should be elevation - baseline = 0.1 - 0.0 = 0.1 mV
    assert abs(st_features['ST_deviation_mv'] - 0.1) < 0.01, \
        f"ST deviation should be ~0.1 mV, got {st_features['ST_deviation_mv']}"
    
    print(f"   ✓ ST elevation: {st_features['ST_elevation_mv']:.4f} mV")
    print(f"   ✓ ST deviation: {st_features['ST_deviation_mv']:.4f} mV")
    print(f"   ✓ ST slope: {st_features['ST_slope_mv_per_s']:.4f} mV/s")
    
    # Test case 2: ST segment with slope
    print("\n2. Testing with ST segment with slope...")
    signal2 = np.zeros(100)
    
    # PR segment: flat at 0.0 mV
    signal2[p_ri_idx:q_le_idx+1] = 0.0
    
    # ST segment: linear slope from 0.0 to 0.2 mV
    st_length = t_le_idx - s_ri_idx + 1
    st_slope_values = np.linspace(0.0, 0.2, st_length)
    signal2[s_ri_idx:t_le_idx+1] = st_slope_values
    
    st_features2 = extract_st_segment_features(
        signal=signal2,
        s_ri_idx=s_ri_idx,
        t_le_idx=t_le_idx,
        p_ri_idx=p_ri_idx,
        q_le_idx=q_le_idx,
        sampling_rate=sampling_rate,
        j_point_offset_ms=60.0,
        verbose=False,
    )
    
    # ST slope should be positive (increasing)
    assert st_features2['ST_slope_mv_per_s'] > 0, \
        f"ST slope should be positive, got {st_features2['ST_slope_mv_per_s']}"
    
    print(f"   ✓ ST elevation: {st_features2['ST_elevation_mv']:.4f} mV")
    print(f"   ✓ ST slope: {st_features2['ST_slope_mv_per_s']:.4f} mV/s (positive)")
    print(f"   ✓ ST deviation: {st_features2['ST_deviation_mv']:.4f} mV")
    
    # Test case 3: Edge cases
    print("\n3. Testing edge cases...")
    
    # Missing indices
    st_features3 = extract_st_segment_features(
        signal=signal,
        s_ri_idx=None,
        t_le_idx=t_le_idx,
        p_ri_idx=p_ri_idx,
        q_le_idx=q_le_idx,
        sampling_rate=sampling_rate,
        verbose=False,
    )
    assert np.isnan(st_features3['ST_elevation_mv']), "Missing S_ri_idx should return NaN"
    print("   ✓ Missing S_ri_idx handled correctly")
    
    # Invalid ST segment (S end after T start)
    st_features4 = extract_st_segment_features(
        signal=signal,
        s_ri_idx=60,
        t_le_idx=40,  # Invalid: T start before S end
        p_ri_idx=p_ri_idx,
        q_le_idx=q_le_idx,
        sampling_rate=sampling_rate,
        verbose=False,
    )
    assert np.isnan(st_features4['ST_elevation_mv']), "Invalid ST segment should return NaN"
    print("   ✓ Invalid ST segment handled correctly")
    
    return True


def test_st_segment_integration():
    """Test ST segment features integration with PyHEARTS."""
    print("\n" + "=" * 80)
    print("Testing ST Segment Features Integration with PyHEARTS")
    print("=" * 80)
    
    # Create test signal
    print("\n1. Creating test ECG signal...")
    signal = create_simple_ecg_signal(duration=60, sampling_rate=250, heart_rate=70)
    print(f"   Signal length: {len(signal)} samples ({len(signal)/250:.1f} seconds)")
    
    # Initialize analyzer
    print("\n2. Initializing PyHEARTS analyzer...")
    analyzer = PyHEARTS(sampling_rate=250, species='human')
    
    # Run analysis
    print("\n3. Running ECG analysis...")
    output_df, epochs_df = analyzer.analyze_ecg(signal, verbose=False)
    
    print(f"   Detected {len(output_df)} cycles")
    
    if len(output_df) == 0:
        print("\n   WARNING: No cycles detected! Cannot test ST segment features.")
        return False
    
    # Check for ST segment features
    print("\n4. Checking for ST segment features...")
    expected_features = ['ST_elevation_mv', 'ST_slope_mv_per_s', 'ST_deviation_mv']
    
    all_features_present = True
    for feature in expected_features:
        if feature not in output_df.columns:
            print(f"   ❌ MISSING: {feature}")
            all_features_present = False
        else:
            # Check if any non-NaN values exist
            non_nan_count = output_df[feature].notna().sum()
            if non_nan_count > 0:
                mean_val = output_df[feature].mean()
                print(f"   ✓ {feature}: {non_nan_count} non-NaN values, mean={mean_val:.4f}")
            else:
                print(f"   ⚠ {feature}: Present but all NaN")
    
    if not all_features_present:
        print("\n   ERROR: Some ST segment features are missing!")
        return False
    
    # Verify values are reasonable
    print("\n5. Verifying values are reasonable...")
    issues = []
    
    for feature in expected_features:
        if feature in output_df.columns:
            values = output_df[feature].dropna()
            if len(values) > 0:
                # Check for extreme values
                if feature == 'ST_elevation_mv' or feature == 'ST_deviation_mv':
                    # ST elevation/deviation typically in range -0.5 to +0.5 mV
                    extreme = values[(values < -1.0) | (values > 1.0)]
                    if len(extreme) > 0:
                        issues.append(f"{feature}: {len(extreme)} extreme values (<-1.0 or >1.0 mV)")
                elif feature == 'ST_slope_mv_per_s':
                    # ST slope typically in range -1.0 to +1.0 mV/s
                    extreme = values[(values < -5.0) | (values > 5.0)]
                    if len(extreme) > 0:
                        issues.append(f"{feature}: {len(extreme)} extreme values (<-5.0 or >5.0 mV/s)")
    
    if issues:
        print("\n   WARNINGS:")
        for issue in issues:
            print(f"   ⚠ {issue}")
    else:
        print("   ✓ All values are reasonable")
    
    # Sample some values
    print("\n6. Sample ST segment features...")
    for feature in expected_features:
        if feature in output_df.columns:
            values = output_df[feature].dropna()
            if len(values) > 0:
                print(f"   {feature}:")
                print(f"     Mean: {values.mean():.4f}")
                print(f"     Std: {values.std():.4f}")
                print(f"     Range: [{values.min():.4f}, {values.max():.4f}]")
                break
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    if all_features_present and len(issues) == 0:
        print("✓ All tests passed!")
        print("✓ ST segment features (elevation, slope, deviation) are correctly implemented")
        return True
    else:
        print("✗ Some tests failed or warnings were raised")
        if issues:
            print("\nIssues found:")
            for issue in issues:
                print(f"  - {issue}")
        return False


if __name__ == "__main__":
    print("Testing ST Segment Calculation Functions...")
    test1_passed = test_st_segment_calculations()
    
    print("\n\nTesting ST Segment Features Integration...")
    test2_passed = test_st_segment_integration()
    
    success = test1_passed and test2_passed
    sys.exit(0 if success else 1)


