#!/usr/bin/env python3
"""
Quick test of P-wave validation impact - single short subject.
"""

import os
import sys
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from dataclasses import replace
import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent.absolute()))
from pyhearts import PyHEARTS
from pyhearts.config import ProcessCycleConfig

SCRIPT_DIR = Path(__file__).parent.absolute()
QTDB_DATA_DIR = SCRIPT_DIR / "data" / "qtdb" / "1.0.0"

# Use just one short subject
TEST_SUBJECT = "sel100"


def analyze_p_detections(output_df):
    """Analyze P-wave detection characteristics."""
    if output_df is None or len(output_df) == 0:
        return {'detection_rate': 0.0, 'p_detected': 0, 'total_cycles': 0}
    
    total_cycles = len(output_df)
    p_col = 'P_global_center_idx' if 'P_global_center_idx' in output_df.columns else None
    
    if p_col is None:
        return {'detection_rate': 0.0, 'p_detected': 0, 'total_cycles': total_cycles}
    
    p_detected = output_df[p_col].notna().sum()
    detection_rate = (p_detected / total_cycles) * 100.0 if total_cycles > 0 else 0.0
    
    return {
        'detection_rate': detection_rate,
        'p_detected': p_detected,
        'total_cycles': total_cycles
    }


def test_configuration(subject, cfg, config_name):
    """Test P-wave detection with a specific configuration."""
    try:
        old_dir = os.getcwd()
        os.chdir(QTDB_DATA_DIR)
        record = wfdb.rdrecord(subject)
        os.chdir(old_dir)
        
        ecg_signal = record.p_signal[:, 0]
        sampling_rate = record.fs
        
        # Use only first 30 seconds for speed
        max_samples = int(30 * sampling_rate)
        ecg_signal = ecg_signal[:max_samples]
        
        analyzer = PyHEARTS(sampling_rate=sampling_rate, cfg=cfg)
        output_df, epochs_df = analyzer.analyze_ecg(ecg_signal)
        
        stats = analyze_p_detections(output_df)
        return stats
        
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def main():
    print("=" * 80)
    print("P-Wave Validation Impact Test (Quick)")
    print("=" * 80)
    print(f"Subject: {TEST_SUBJECT} (first 30 seconds only)")
    print()
    
    base_cfg = ProcessCycleConfig.for_human()
    configs = [
        ("Both enabled", replace(
            base_cfg,
            p_use_derivative_validated_method=True,
            p_enable_distance_validation=True,
            p_enable_morphology_validation=True
        )),
        ("Distance disabled", replace(
            base_cfg,
            p_use_derivative_validated_method=True,
            p_enable_distance_validation=False,
            p_enable_morphology_validation=True
        )),
        ("Morphology disabled", replace(
            base_cfg,
            p_use_derivative_validated_method=True,
            p_enable_distance_validation=True,
            p_enable_morphology_validation=False
        )),
        ("Both disabled", replace(
            base_cfg,
            p_use_derivative_validated_method=True,
            p_enable_distance_validation=False,
            p_enable_morphology_validation=False
        ))
    ]
    
    results = {}
    for config_name, cfg in configs:
        print(f"Testing {config_name}...", end=" ", flush=True)
        stats = test_configuration(TEST_SUBJECT, cfg, config_name)
        if stats:
            results[config_name] = stats
            print(f"✓ {stats['p_detected']}/{stats['total_cycles']} detected ({stats['detection_rate']:.1f}%)")
        else:
            print("✗ Failed")
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    if len(results) == 0:
        print("No results collected.")
        return
    
    for config_name in ["Both enabled", "Distance disabled", "Morphology disabled", "Both disabled"]:
        if config_name in results:
            r = results[config_name]
            print(f"\n{config_name}:")
            print(f"  Detection rate: {r['detection_rate']:.1f}%")
            print(f"  P-waves detected: {r['p_detected']}/{r['total_cycles']}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    if "Both enabled" in results and "Both disabled" in results:
        both_enabled = results["Both enabled"]
        both_disabled = results["Both disabled"]
        
        detection_diff = both_enabled['detection_rate'] - both_disabled['detection_rate']
        
        print(f"\nDetection rate difference: {detection_diff:+.1f}%")
        
        if abs(detection_diff) < 2.0:
            print("\n✓ RECOMMENDATION: Validations have minimal impact - can be kept optional")
        elif detection_diff < -5.0:
            print("\n✓ RECOMMENDATION: Consider disabling validations (significantly increases detection)")
        elif detection_diff > 5.0:
            print("\n✓ RECOMMENDATION: Keep validations enabled (significantly improves precision)")
        else:
            print("\n✓ RECOMMENDATION: Keep validations optional based on use case")


if __name__ == "__main__":
    main()

