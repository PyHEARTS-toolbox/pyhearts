"""
Test script to verify additional HRV metrics (pNN50, SD1, SD2) are working correctly.

This script:
1. Tests the HRV calculation functions with known values
2. Verifies new HRV metrics are computed correctly
3. Tests integration with PyHEARTS
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pyhearts import PyHEARTS
from pyhearts.feature.hrv import calc_hrv_metrics


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


def test_hrv_calculations():
    """Test HRV calculation functions with known values."""
    print("=" * 80)
    print("Testing HRV Calculation Functions")
    print("=" * 80)
    
    # Test case 1: Constant RR intervals (no variability)
    print("\n1. Testing with constant RR intervals (no variability)...")
    constant_rr = np.full(100, 1000.0)  # All 1000ms = 60 bpm
    hr, sdnn, rmssd, nn50, pnn50, sd1, sd2 = calc_hrv_metrics(constant_rr)
    
    assert hr == 60, f"Expected HR=60, got {hr}"
    assert sdnn == 0, f"Expected SDNN=0, got {sdnn}"
    assert rmssd == 0, f"Expected RMSSD=0, got {rmssd}"
    assert nn50 == 0, f"Expected NN50=0, got {nn50}"
    assert pnn50 == 0.0, f"Expected pNN50=0.0, got {pnn50}"
    assert sd1 == 0.0, f"Expected SD1=0.0, got {sd1}"
    assert sd2 == 0, f"Expected SD2=0, got {sd2}"
    print("   ✓ All metrics correct for constant intervals")
    
    # Test case 2: Variable RR intervals
    print("\n2. Testing with variable RR intervals...")
    np.random.seed(42)
    variable_rr = 1000 + np.random.normal(0, 50, 100)  # Mean 1000ms, std 50ms
    variable_rr = np.clip(variable_rr, 600, 1400)  # Keep in reasonable range
    
    hr, sdnn, rmssd, nn50, pnn50, sd1, sd2 = calc_hrv_metrics(variable_rr)
    
    # Calculate raw values before rounding for verification
    # (SD1 and SD2 are calculated from raw values, then rounded)
    raw_rmssd = np.sqrt(np.mean(np.diff(variable_rr) ** 2))
    raw_sdnn = np.std(variable_rr, ddof=1)
    raw_sd1 = raw_rmssd / np.sqrt(2.0)
    raw_sd2 = np.sqrt(2.0 * (raw_sdnn ** 2) - (raw_sd1 ** 2))
    
    # Verify relationships
    # SD1 should equal round(RMSSD / sqrt(2), 2) where RMSSD is the raw (unrounded) value
    expected_sd1 = round(raw_sd1, 2)
    if sd1 is not None:
        assert abs(sd1 - expected_sd1) < 0.01, f"SD1 should equal round(raw_RMSSD/√2, 2): {sd1} vs {expected_sd1}"
    
    # SD2 should equal round(sqrt(2 * SDNN^2 - SD1^2), 2) where SDNN and SD1 are raw values
    expected_sd2 = round(raw_sd2, 2)
    if sd2 is not None:
        assert abs(sd2 - expected_sd2) < 0.01, f"SD2 calculation mismatch: {sd2} vs {expected_sd2}"
    
    # pNN50 should equal (NN50 / (N-1)) * 100 (rounded to 2 decimals)
    expected_pnn50 = round((nn50 / (len(variable_rr) - 1)) * 100.0, 2) if nn50 is not None else None
    if expected_pnn50 is not None and pnn50 is not None:
        assert abs(pnn50 - expected_pnn50) < 0.01, f"pNN50 calculation mismatch: {pnn50} vs {expected_pnn50}"
    
    print(f"   ✓ HR={hr} bpm, SDNN={sdnn} ms, RMSSD={rmssd} ms")
    print(f"   ✓ NN50={nn50}, pNN50={pnn50:.2f}%")
    print(f"   ✓ SD1={sd1:.2f} ms, SD2={sd2:.2f} ms")
    print(f"   ✓ Relationships verified: SD1=RMSSD/√2, SD2=√(2×SDNN²-SD1²), pNN50=NN50/(N-1)×100")
    
    # Test case 3: Edge cases
    print("\n3. Testing edge cases...")
    
    # Single interval
    single_rr = np.array([1000.0])
    hr, sdnn, rmssd, nn50, pnn50, sd1, sd2 = calc_hrv_metrics(single_rr)
    assert hr == 60, "Single interval should give HR=60"
    assert sdnn is None, "SDNN should be None for single interval"
    assert pnn50 is None, "pNN50 should be None for single interval"
    print("   ✓ Single interval handled correctly")
    
    # All NaN
    all_nan = np.array([np.nan, np.nan, np.nan])
    hr, sdnn, rmssd, nn50, pnn50, sd1, sd2 = calc_hrv_metrics(all_nan)
    assert hr is None or np.isnan(hr), "All NaN should return None/NaN"
    print("   ✓ All NaN handled correctly")
    
    # Two intervals
    two_rr = np.array([1000.0, 1050.0])
    hr, sdnn, rmssd, nn50, pnn50, sd1, sd2 = calc_hrv_metrics(two_rr)
    assert hr is not None, "Two intervals should give HR"
    assert sdnn is not None, "Two intervals should give SDNN"
    assert pnn50 is not None, "Two intervals should give pNN50"
    # Difference is 50ms, so NN50 should be 1 if >50, or 0 if <=50
    # Actually, abs(1050-1000) = 50, so it's not >50, so NN50 should be 0
    assert nn50 == 0, f"NN50 should be 0 for 50ms difference, got {nn50}"
    assert pnn50 == 0.0, f"pNN50 should be 0.0 for 50ms difference, got {pnn50}"
    print("   ✓ Two intervals handled correctly")
    
    return True


def test_hrv_integration():
    """Test HRV metrics integration with PyHEARTS."""
    print("\n" + "=" * 80)
    print("Testing HRV Metrics Integration with PyHEARTS")
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
        print("\n   WARNING: No cycles detected! Cannot test HRV.")
        return False
    
    # Compute HRV metrics
    print("\n4. Computing HRV metrics...")
    analyzer.compute_hrv_metrics()
    
    # Check for new HRV metrics
    print("\n5. Checking for new HRV metrics...")
    expected_metrics = ['average_heart_rate', 'sdnn', 'rmssd', 'nn50', 'pnn50', 'sd1', 'sd2']
    
    all_metrics_present = True
    for metric in expected_metrics:
        if metric not in analyzer.hrv_metrics:
            print(f"   ❌ MISSING: {metric}")
            all_metrics_present = False
        else:
            value = analyzer.hrv_metrics[metric]
            print(f"   ✓ {metric}: {value}")
    
    if not all_metrics_present:
        print("\n   ERROR: Some HRV metrics are missing!")
        return False
    
    # Verify relationships
    print("\n6. Verifying metric relationships...")
    sdnn = analyzer.hrv_metrics.get('sdnn')
    rmssd = analyzer.hrv_metrics.get('rmssd')
    nn50 = analyzer.hrv_metrics.get('nn50')
    pnn50 = analyzer.hrv_metrics.get('pnn50')
    sd1 = analyzer.hrv_metrics.get('sd1')
    sd2 = analyzer.hrv_metrics.get('sd2')
    
    issues = []
    
    # SD1 should equal RMSSD / sqrt(2)
    # Note: RMSSD is rounded to int, so allow tolerance
    if rmssd is not None and sd1 is not None:
        expected_sd1 = rmssd / np.sqrt(2.0)
        if abs(sd1 - expected_sd1) > 1.0:
            issues.append(f"SD1 ({sd1:.2f}) should equal RMSSD/√2 ({expected_sd1:.2f})")
        else:
            print(f"   ✓ SD1 = RMSSD/√2: {sd1:.2f} ≈ {rmssd}/{np.sqrt(2.0):.3f} = {expected_sd1:.2f}")
    
    # SD2 should equal sqrt(2 * SDNN^2 - SD1^2)
    # Note: SDNN is rounded to int, so allow tolerance
    if sdnn is not None and sd1 is not None and sd2 is not None:
        expected_sd2 = np.sqrt(2.0 * (sdnn ** 2) - (sd1 ** 2))
        if expected_sd2 >= 0 and abs(sd2 - expected_sd2) > 1.0:
            issues.append(f"SD2 ({sd2:.2f}) should equal √(2×SDNN²-SD1²) ({expected_sd2:.2f})")
        else:
            print(f"   ✓ SD2 = √(2×SDNN²-SD1²): {sd2:.2f} ≈ {expected_sd2:.2f}")
    
    # pNN50 should be percentage
    if pnn50 is not None:
        if pnn50 < 0 or pnn50 > 100:
            issues.append(f"pNN50 ({pnn50:.2f}) should be between 0 and 100")
        else:
            print(f"   ✓ pNN50 is in valid range: {pnn50:.2f}%")
    
    if issues:
        print("\n   WARNINGS:")
        for issue in issues:
            print(f"   ⚠ {issue}")
    
    # Verify values are reasonable
    print("\n7. Verifying values are reasonable...")
    if sdnn is not None and sdnn < 0:
        issues.append(f"SDNN should be non-negative, got {sdnn}")
    if rmssd is not None and rmssd < 0:
        issues.append(f"RMSSD should be non-negative, got {rmssd}")
    if sd1 is not None and sd1 < 0:
        issues.append(f"SD1 should be non-negative, got {sd1}")
    if sd2 is not None and sd2 < 0:
        issues.append(f"SD2 should be non-negative, got {sd2}")
    
    if not issues:
        print("   ✓ All values are reasonable")
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    if all_metrics_present and len(issues) == 0:
        print("✓ All tests passed!")
        print("✓ New HRV metrics (pNN50, SD1, SD2) are correctly implemented")
        print("✓ Metric relationships are correct")
        return True
    else:
        print("✗ Some tests failed or warnings were raised")
        if issues:
            print("\nIssues found:")
            for issue in issues:
                print(f"  - {issue}")
        return False


if __name__ == "__main__":
    print("Testing HRV Calculation Functions...")
    test1_passed = test_hrv_calculations()
    
    print("\n\nTesting HRV Metrics Integration...")
    test2_passed = test_hrv_integration()
    
    success = test1_passed and test2_passed
    sys.exit(0 if success else 1)

