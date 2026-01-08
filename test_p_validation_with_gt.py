#!/usr/bin/env python3
"""
Test P-wave validation impact using QTDB ground truth annotations.
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

# Use just one subject for quick test
TEST_SUBJECT = "sel100"
TOLERANCE_MS = 40.0  # 40ms tolerance for matching


def load_ground_truth_p_peaks(subject):
    """Load P-wave ground truth from QTDB .pu annotations."""
    old_dir = os.getcwd()
    os.chdir(QTDB_DATA_DIR)
    
    try:
        # Try .pu file first (contains P-wave annotations)
        annotation = wfdb.rdann(subject, 'pu')
        p_peaks = []
        
        for i in range(len(annotation.sample)):
            symbol = annotation.symbol[i]
            sample_idx = annotation.sample[i]
            
            if symbol.lower() == 'p':  # lowercase 'p' in .pu files
                p_peaks.append(sample_idx)
        
        return np.array(sorted(p_peaks))
        
    except Exception as e:
        print(f"  ERROR loading .pu annotations: {e}")
        return np.array([])
    finally:
        os.chdir(old_dir)


def match_peaks(detected_peaks, ground_truth_peaks, tolerance_samples):
    """Match detected peaks to ground truth within tolerance (one-to-one matching)."""
    if len(detected_peaks) == 0 or len(ground_truth_peaks) == 0:
        return np.array([]), np.array([])
    
    matched_detected = []
    matched_gt = []
    used_detected = set()
    used_gt = set()
    
    # Sort by distance to find best matches first
    pairs = []
    for i, gt_peak in enumerate(ground_truth_peaks):
        for j, det_peak in enumerate(detected_peaks):
            dist = abs(det_peak - gt_peak)
            if dist <= tolerance_samples:
                pairs.append((dist, i, j, gt_peak, det_peak))
    
    # Sort by distance and match greedily (one-to-one)
    pairs.sort(key=lambda x: x[0])
    
    for dist, i, j, gt_peak, det_peak in pairs:
        if i not in used_gt and j not in used_detected:
            matched_detected.append(det_peak)
            matched_gt.append(gt_peak)
            used_gt.add(i)
            used_detected.add(j)
    
    return np.array(matched_detected), np.array(matched_gt)


def calculate_metrics(detected_peaks, ground_truth_peaks, tolerance_samples, sampling_rate):
    """Calculate precision, recall, F1, and timing accuracy."""
    if len(ground_truth_peaks) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'true_positives': 0,
            'false_positives': len(detected_peaks),
            'false_negatives': len(ground_truth_peaks),
            'mae_ms': np.nan,
            'mad_ms': np.nan
        }
    
    matched_detected, matched_gt = match_peaks(detected_peaks, ground_truth_peaks, tolerance_samples)
    matched_count = len(matched_detected)
    
    recall = (matched_count / len(ground_truth_peaks)) * 100.0 if len(ground_truth_peaks) > 0 else 0.0
    precision = (matched_count / len(detected_peaks)) * 100.0 if len(detected_peaks) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    false_positives = len(detected_peaks) - matched_count
    false_negatives = len(ground_truth_peaks) - matched_count
    
    # Timing accuracy
    if matched_count > 0:
        errors_ms = ((matched_detected - matched_gt) / sampling_rate) * 1000.0
        mae_ms = np.mean(np.abs(errors_ms))
        mad_ms = np.median(np.abs(errors_ms))
    else:
        mae_ms = np.nan
        mad_ms = np.nan
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': matched_count,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'mae_ms': mae_ms,
        'mad_ms': mad_ms
    }


def test_configuration(subject, cfg, config_name, gt_p_peaks, sampling_rate, tolerance_samples):
    """Test P-wave detection with a specific configuration."""
    try:
        old_dir = os.getcwd()
        os.chdir(QTDB_DATA_DIR)
        record = wfdb.rdrecord(subject)
        os.chdir(old_dir)
        
        ecg_signal = record.p_signal[:, 0]
        
        # Use only first 30 seconds for speed
        max_samples = int(30 * sampling_rate)
        ecg_signal = ecg_signal[:max_samples]
        
        # Filter ground truth to same window
        gt_p_peaks_filtered = gt_p_peaks[gt_p_peaks < max_samples]
        
        analyzer = PyHEARTS(sampling_rate=sampling_rate, cfg=cfg)
        output_df, epochs_df = analyzer.analyze_ecg(ecg_signal)
        
        # Extract detected P peaks
        if output_df is not None and len(output_df) > 0 and 'P_global_center_idx' in output_df.columns:
            p_peaks = output_df['P_global_center_idx'].dropna()
            p_peaks = p_peaks[np.isfinite(p_peaks) & (p_peaks > 0) & (p_peaks < max_samples)].values.astype(int)
        else:
            p_peaks = np.array([])
        
        # Calculate metrics
        metrics = calculate_metrics(p_peaks, gt_p_peaks_filtered, tolerance_samples, sampling_rate)
        
        return {
            'detected_count': len(p_peaks),
            'gt_count': len(gt_p_peaks_filtered),
            **metrics
        }
        
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def main():
    print("=" * 80)
    print("P-Wave Validation Impact Test with Ground Truth")
    print("=" * 80)
    print(f"Subject: {TEST_SUBJECT} (first 30 seconds)")
    print()
    
    # Load ground truth
    gt_p_peaks = load_ground_truth_p_peaks(TEST_SUBJECT)
    if len(gt_p_peaks) == 0:
        print("ERROR: No P-wave ground truth found in annotations")
        print("Checking available symbols...")
        old_dir = os.getcwd()
        os.chdir(QTDB_DATA_DIR)
        try:
            ann = wfdb.rdann(TEST_SUBJECT, 'man')
            print(f"Available symbols: {set(ann.symbol)}")
            print(f"Total annotations: {len(ann.symbol)}")
        except:
            pass
        finally:
            os.chdir(old_dir)
        return
    
    # Get sampling rate
    old_dir = os.getcwd()
    os.chdir(QTDB_DATA_DIR)
    record = wfdb.rdrecord(TEST_SUBJECT)
    os.chdir(old_dir)
    sampling_rate = record.fs
    tolerance_samples = int(round(TOLERANCE_MS * sampling_rate / 1000.0))
    
    print(f"Ground truth P peaks: {len(gt_p_peaks)}")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Tolerance: {TOLERANCE_MS} ms ({tolerance_samples} samples)")
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
        metrics = test_configuration(TEST_SUBJECT, cfg, config_name, gt_p_peaks, sampling_rate, tolerance_samples)
        if metrics:
            results[config_name] = metrics
            print(f"✓ Precision: {metrics['precision']:.1f}%, Recall: {metrics['recall']:.1f}%, F1: {metrics['f1']:.1f}%")
        else:
            print("✗ Failed")
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    for config_name in ["Both enabled", "Distance disabled", "Morphology disabled", "Both disabled"]:
        if config_name in results:
            r = results[config_name]
            print(f"\n{config_name}:")
            print(f"  Precision: {r['precision']:.1f}%")
            print(f"  Recall:    {r['recall']:.1f}%")
            print(f"  F1:        {r['f1']:.1f}%")
            print(f"  TP:        {r['true_positives']}")
            print(f"  FP:        {r['false_positives']}")
            print(f"  FN:        {r['false_negatives']}")
            if not np.isnan(r['mae_ms']):
                print(f"  MAE:       {r['mae_ms']:.2f} ms")
                print(f"  MAD:       {r['mad_ms']:.2f} ms")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    if "Both enabled" in results and "Both disabled" in results:
        both_enabled = results["Both enabled"]
        both_disabled = results["Both disabled"]
        
        f1_diff = both_enabled['f1'] - both_disabled['f1']
        precision_diff = both_enabled['precision'] - both_disabled['precision']
        recall_diff = both_enabled['recall'] - both_disabled['recall']
        
        print(f"\nBoth enabled vs Both disabled:")
        print(f"  F1 difference:      {f1_diff:+.1f}%")
        print(f"  Precision difference: {precision_diff:+.1f}%")
        print(f"  Recall difference:    {recall_diff:+.1f}%")
        
        if f1_diff > 2.0:
            print("\n✓ RECOMMENDATION: Keep validations enabled (improves F1)")
        elif f1_diff < -2.0:
            print("\n✓ RECOMMENDATION: Disable validations (improves F1)")
        elif precision_diff > 5.0:
            print("\n✓ RECOMMENDATION: Keep validations enabled (significantly improves precision)")
        elif recall_diff < -5.0:
            print("\n✓ RECOMMENDATION: Disable validations (significantly improves recall)")
        else:
            print("\n✓ RECOMMENDATION: Validations have minimal impact - can be kept optional")


if __name__ == "__main__":
    main()

