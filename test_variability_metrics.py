"""
Test script to verify beat-to-beat variability metrics are working correctly.

This script:
1. Tests the variability calculation functions with known values
2. Verifies variability metrics are computed correctly
3. Tests integration with PyHEARTS
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pyhearts import PyHEARTS
from pyhearts.feature.variability import compute_variability_metrics, compute_beat_to_beat_variability


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


def test_variability_calculations():
    """Test variability calculation functions with known values."""
    print("=" * 80)
    print("Testing Variability Calculation Functions")
    print("=" * 80)
    
    # Test case 1: Constant values (no variability)
    print("\n1. Testing with constant values (no variability)...")
    constant_values = np.full(100, 400.0)  # All 400ms
    metrics = compute_variability_metrics(constant_values, "test_feature")
    
    assert metrics["test_feature_std"] == 0.0, f"Expected std=0, got {metrics['test_feature_std']}"
    assert metrics["test_feature_cv"] == 0.0, f"Expected CV=0, got {metrics['test_feature_cv']}"
    assert metrics["test_feature_iqr"] == 0.0, f"Expected IQR=0, got {metrics['test_feature_iqr']}"
    assert metrics["test_feature_mad"] == 0.0, f"Expected MAD=0, got {metrics['test_feature_mad']}"
    assert metrics["test_feature_range"] == 0.0, f"Expected range=0, got {metrics['test_feature_range']}"
    print("   ✓ All metrics correct for constant values")
    
    # Test case 2: Variable values
    print("\n2. Testing with variable values...")
    np.random.seed(42)
    variable_values = 400 + np.random.normal(0, 20, 100)  # Mean 400ms, std 20ms
    
    metrics = compute_variability_metrics(variable_values, "test_feature")
    
    # Verify relationships
    # CV should equal std / mean
    mean_val = np.mean(variable_values)
    expected_cv = np.std(variable_values, ddof=1) / abs(mean_val)
    assert abs(metrics["test_feature_cv"] - expected_cv) < 0.01, \
        f"CV mismatch: {metrics['test_feature_cv']} vs {expected_cv}"
    
    # IQR should be 75th - 25th percentile
    expected_iqr = np.percentile(variable_values, 75) - np.percentile(variable_values, 25)
    assert abs(metrics["test_feature_iqr"] - expected_iqr) < 0.01, \
        f"IQR mismatch: {metrics['test_feature_iqr']} vs {expected_iqr}"
    
    # Range should be max - min
    expected_range = np.max(variable_values) - np.min(variable_values)
    assert abs(metrics["test_feature_range"] - expected_range) < 0.01, \
        f"Range mismatch: {metrics['test_feature_range']} vs {expected_range}"
    
    print(f"   ✓ std={metrics['test_feature_std']:.2f}, CV={metrics['test_feature_cv']:.4f}")
    print(f"   ✓ IQR={metrics['test_feature_iqr']:.2f}, MAD={metrics['test_feature_mad']:.2f}, range={metrics['test_feature_range']:.2f}")
    
    # Test case 3: Edge cases
    print("\n3. Testing edge cases...")
    
    # Single value
    single_value = np.array([400.0])
    metrics = compute_variability_metrics(single_value, "test")
    assert np.isnan(metrics["test_std"]), "Single value should return NaN for std"
    assert np.isnan(metrics["test_cv"]), "Single value should return NaN for CV"
    print("   ✓ Single value handled correctly")
    
    # All NaN
    all_nan = np.array([np.nan, np.nan, np.nan])
    metrics = compute_variability_metrics(all_nan, "test")
    assert np.isnan(metrics["test_std"]), "All NaN should return NaN"
    print("   ✓ All NaN handled correctly")
    
    # Two values
    two_values = np.array([400.0, 450.0])
    metrics = compute_variability_metrics(two_values, "test")
    assert not np.isnan(metrics["test_std"]), "Two values should give std"
    assert metrics["test_range"] == 50.0, f"Range should be 50, got {metrics['test_range']}"
    print("   ✓ Two values handled correctly")
    
    # Zero mean (CV should be NaN)
    zero_mean = np.array([-10.0, 10.0])  # Mean = 0
    metrics = compute_variability_metrics(zero_mean, "test")
    assert np.isnan(metrics["test_cv"]), "Zero mean should give NaN CV"
    assert not np.isnan(metrics["test_std"]), "Zero mean should still give std"
    print("   ✓ Zero mean handled correctly")
    
    return True


def test_variability_integration():
    """Test variability metrics integration with PyHEARTS."""
    print("\n" + "=" * 80)
    print("Testing Variability Metrics Integration with PyHEARTS")
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
        print("\n   WARNING: No cycles detected! Cannot test variability.")
        return False
    
    # Check if variability metrics were computed automatically
    print("\n4. Checking for variability metrics...")
    
    if not analyzer.variability_metrics:
        print("   ⚠ Variability metrics not computed automatically, computing now...")
        analyzer.compute_variability_metrics()
    
    if not analyzer.variability_metrics:
        print("\n   ERROR: Variability metrics are empty!")
        return False
    
    # Check for expected variability metrics
    print("\n5. Verifying variability metrics are present...")
    priority_features = [
        "QT_interval_ms", "QRS_interval_ms", "PR_interval_ms", "RR_interval_ms",
        "QTc_Bazett_ms", "QTc_Fridericia_ms",
        "R_gauss_height", "P_gauss_height", "T_gauss_height",
    ]
    
    expected_metrics_per_feature = ["_std", "_cv", "_iqr", "_mad", "_range"]
    
    found_metrics = 0
    for feature in priority_features:
        if feature in analyzer.output_dict:
            for metric_suffix in expected_metrics_per_feature:
                metric_name = f"{feature}{metric_suffix}"
                if metric_name in analyzer.variability_metrics:
                    found_metrics += 1
                    value = analyzer.variability_metrics[metric_name]
                    if not np.isnan(value):
                        print(f"   ✓ {metric_name}: {value:.4f}")
    
    if found_metrics == 0:
        print("\n   ERROR: No variability metrics found!")
        return False
    
    print(f"\n   Found {found_metrics} variability metrics")
    
    # Verify metric values are reasonable
    print("\n6. Verifying metric values are reasonable...")
    issues = []
    
    for metric_name, value in analyzer.variability_metrics.items():
        if np.isnan(value):
            continue
        
        # Check for negative values (std, cv, iqr, mad, range should be non-negative)
        if any(suffix in metric_name for suffix in ["_std", "_cv", "_iqr", "_mad", "_range"]):
            if value < 0:
                issues.append(f"{metric_name}: Negative value ({value:.4f})")
        
        # CV should typically be < 1.0 for most features (but can be higher)
        if "_cv" in metric_name and value > 10.0:
            issues.append(f"{metric_name}: Very high CV ({value:.4f}), may indicate issue")
    
    if issues:
        print("\n   WARNINGS:")
        for issue in issues:
            print(f"   ⚠ {issue}")
    else:
        print("   ✓ All values are reasonable")
    
    # Sample some values
    print("\n7. Sample variability metrics...")
    sample_features = ["QT_interval_ms", "R_gauss_height", "QRS_interval_ms"]
    for feature in sample_features:
        if f"{feature}_std" in analyzer.variability_metrics:
            std = analyzer.variability_metrics[f"{feature}_std"]
            cv = analyzer.variability_metrics.get(f"{feature}_cv", np.nan)
            iqr = analyzer.variability_metrics.get(f"{feature}_iqr", np.nan)
            if not np.isnan(std):
                print(f"   {feature}:")
                print(f"     std: {std:.4f}")
                if not np.isnan(cv):
                    print(f"     CV: {cv:.4f}")
                if not np.isnan(iqr):
                    print(f"     IQR: {iqr:.4f}")
                break
    
    # Verify relationships
    print("\n8. Verifying metric relationships...")
    for feature in priority_features[:3]:  # Check first 3 features
        if f"{feature}_std" in analyzer.variability_metrics:
            std = analyzer.variability_metrics[f"{feature}_std"]
            cv = analyzer.variability_metrics.get(f"{feature}_cv", np.nan)
            
            # Get mean from output_df
            if feature in output_df.columns:
                mean_val = output_df[feature].mean()
                if not np.isnan(mean_val) and abs(mean_val) > 1e-6:
                    expected_cv = std / abs(mean_val)
                    if not np.isnan(cv) and abs(cv - expected_cv) > 0.01:
                        issues.append(f"{feature}_cv: Mismatch with std/mean ({cv:.4f} vs {expected_cv:.4f})")
                    else:
                        print(f"   ✓ {feature}_cv relationship verified")
            break
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    if found_metrics > 0 and len(issues) == 0:
        print("✓ All tests passed!")
        print("✓ Variability metrics are correctly implemented")
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
    print("Testing Variability Calculation Functions...")
    test1_passed = test_variability_calculations()
    
    print("\n\nTesting Variability Metrics Integration...")
    test2_passed = test_variability_integration()
    
    success = test1_passed and test2_passed
    sys.exit(0 if success else 1)

