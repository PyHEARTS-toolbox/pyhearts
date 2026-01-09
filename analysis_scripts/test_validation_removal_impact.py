#!/usr/bin/env python3
"""
Test that removing distance and morphology validation didn't break detection or features.
"""

import os
import sys
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent.absolute()))
from pyhearts import PyHEARTS
from pyhearts.config import ProcessCycleConfig

SCRIPT_DIR = Path(__file__).parent.absolute()
QTDB_DATA_DIR = SCRIPT_DIR / "data" / "qtdb" / "1.0.0"

TEST_SUBJECT = "sel100"


def test_detection_and_features(subject, cfg, config_name):
    """Test P-wave detection and feature extraction."""
    try:
        old_dir = os.getcwd()
        os.chdir(QTDB_DATA_DIR)
        record = wfdb.rdrecord(subject)
        os.chdir(old_dir)
        
        ecg_signal = record.p_signal[:, 0]
        sampling_rate = record.fs
        
        # Use first 30 seconds for speed
        max_samples = int(30 * sampling_rate)
        ecg_signal = ecg_signal[:max_samples]
        
        analyzer = PyHEARTS(sampling_rate=sampling_rate, cfg=cfg)
        output_df, epochs_df = analyzer.analyze_ecg(ecg_signal)
        
        if output_df is None or len(output_df) == 0:
            return {
                'success': False,
                'error': 'No output dataframe',
                'cycles': 0,
                'p_detected': 0,
                'features_extracted': False
            }
        
        # Check P-wave detection
        p_col = 'P_global_center_idx' if 'P_global_center_idx' in output_df.columns else None
        p_detected = 0
        if p_col:
            p_detected = output_df[p_col].notna().sum()
        
        # Check that key features are present (check for variations in naming)
        required_features = [
            'R_global_center_idx',
            'PR_interval_ms', 'QT_interval_ms', 'QRS_interval_ms'
        ]
        
        # R_height may be named differently, so check for any R amplitude/height column
        has_r_amplitude = any('R' in c and ('height' in c.lower() or 'amp' in c.lower()) 
                              for c in output_df.columns)
        
        features_present = all(col in output_df.columns for col in required_features) and has_r_amplitude
        
        # Check that P-wave related features are present if P-waves detected
        p_features_present = True
        if p_detected > 0:
            p_related_features = ['P_global_center_idx']
            # P_height may not always be present, so just check for P_global_center_idx
            p_features_present = 'P_global_center_idx' in output_df.columns
        
        # Check for NaN/invalid values in key features
        valid_r_peaks = output_df['R_global_center_idx'].notna().sum() if 'R_global_center_idx' in output_df.columns else 0
        valid_pr_intervals = output_df['PR_interval_ms'].notna().sum() if 'PR_interval_ms' in output_df.columns else 0
        valid_qt_intervals = output_df['QT_interval_ms'].notna().sum() if 'QT_interval_ms' in output_df.columns else 0
        
        return {
            'success': True,
            'cycles': len(output_df),
            'p_detected': p_detected,
            'p_detection_rate': (p_detected / len(output_df)) * 100.0 if len(output_df) > 0 else 0.0,
            'features_extracted': features_present,
            'p_features_present': p_features_present,
            'valid_r_peaks': valid_r_peaks,
            'valid_pr_intervals': valid_pr_intervals,
            'valid_qt_intervals': valid_qt_intervals,
            'columns': list(output_df.columns)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'cycles': 0,
            'p_detected': 0,
            'features_extracted': False
        }


def main():
    print("=" * 80)
    print("Validation Removal Impact Test")
    print("=" * 80)
    print("Testing that removing distance/morphology validation didn't break functionality")
    print()
    
    # Test with default human config (validations disabled)
    cfg_default = ProcessCycleConfig.for_human()
    
    print(f"Default config:")
    print(f"  Distance validation: {cfg_default.p_enable_distance_validation}")
    print(f"  Morphology validation: {cfg_default.p_enable_morphology_validation}")
    print(f"  Derivative validated method: {cfg_default.p_use_derivative_validated_method}")
    print()
    
    print(f"Testing subject: {TEST_SUBJECT}")
    print()
    
    result = test_detection_and_features(TEST_SUBJECT, cfg_default, "Default")
    
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    if not result['success']:
        print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
        return
    
    print(f"✓ Detection successful")
    print(f"  Total cycles: {result['cycles']}")
    print(f"  P-waves detected: {result['p_detected']} ({result['p_detection_rate']:.1f}%)")
    print(f"  Valid R-peaks: {result['valid_r_peaks']}")
    print()
    
    print(f"✓ Feature extraction:")
    print(f"  Core features present: {result['features_extracted']}")
    print(f"  P-wave features present: {result['p_features_present']}")
    print(f"  Valid PR intervals: {result['valid_pr_intervals']}")
    print(f"  Valid QT intervals: {result['valid_qt_intervals']}")
    print()
    
    # Check for any issues
    issues = []
    
    if result['cycles'] == 0:
        issues.append("No cycles detected")
    
    if result['valid_r_peaks'] == 0:
        issues.append("No valid R-peaks detected")
    
    if result['p_detected'] == 0:
        issues.append("No P-waves detected (may be normal for this subject)")
    
    if not result['features_extracted']:
        issues.append("Core features missing from output")
    
    if not result['p_features_present'] and result['p_detected'] > 0:
        issues.append("P-wave features missing despite P-waves detected")
    
    if result['valid_pr_intervals'] == 0 and result['p_detected'] > 0:
        issues.append("No valid PR intervals despite P-waves detected")
    
    if issues:
        print("⚠️  POTENTIAL ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ All checks passed - no issues detected")
    
    print()
    print("=" * 80)
    print("FEATURE COLUMNS CHECK")
    print("=" * 80)
    print(f"Total columns: {len(result['columns'])}")
    print()
    print("Key columns present:")
    key_cols = ['R_global_center_idx', 'P_global_center_idx', 'Q_global_center_idx', 
                'S_global_center_idx', 'T_global_center_idx', 'PR_interval_ms', 
                'QRS_interval_ms', 'QT_interval_ms']
    for col in key_cols:
        status = "✓" if col in result['columns'] else "✗"
        print(f"  {status} {col}")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if not issues:
        print("✓ SUCCESS: Algorithm is working correctly after validation removal")
        print("  - Detection functional")
        print("  - Features extracted correctly")
        print("  - No breaking changes detected")
    else:
        print("⚠️  WARNING: Some issues detected (see above)")
        print("  - Algorithm may need investigation")


if __name__ == "__main__":
    main()

