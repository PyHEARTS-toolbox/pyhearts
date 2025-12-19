#!/usr/bin/env python3
"""
Debug script to check if inverted signal detection is working on QTDB signals.
"""

import os
import sys
import numpy as np
import wfdb
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pyhearts.processing import r_peak_detection
from pyhearts.processing.rpeak import _detect_signal_polarity
from pyhearts.config import ProcessCycleConfig

# Subjects to test
subjects = ["sele0121", "sele0122", "sele0211", "sele0409"]
qtdb_dir = project_root / "qtdb_inverted-sig_subjects_analysis" / "qtdb_raw_files"

cfg = ProcessCycleConfig()

def get_preferred_lead_index(signal_names):
    """Select the best lead from available signals."""
    preferred_order = ["ECG2", "ECG1", "MLII", "V5", "V2", "CM5", "CM4", "ML5", "V1"]
    for preferred in preferred_order:
        for idx, name in enumerate(signal_names):
            if name.upper() == preferred.upper():
                return idx
    return 0

print("="*80)
print("DEBUGGING INVERTED SIGNAL DETECTION ON QTDB PARTICIPANTS")
print("="*80)

for subject in subjects:
    print(f"\n{'='*80}")
    print(f"Subject: {subject}")
    print(f"{'='*80}")
    
    record_path = qtdb_dir / subject
    
    try:
        # Load signal
        signals, fields = wfdb.rdsamp(str(record_path))
        signal_names = fields.get("sig_name", [])
        sampling_rate = fields.get("fs", None)
        
        lead_idx = get_preferred_lead_index(signal_names)
        lead_name = signal_names[lead_idx] if lead_idx < len(signal_names) else "Unknown"
        ecg_signal = signals[:, lead_idx]
        
        print(f"  Lead: {lead_name} (index {lead_idx})")
        print(f"  Sampling rate: {sampling_rate} Hz")
        print(f"  Signal range: [{ecg_signal.min():.3f}, {ecg_signal.max():.3f}]")
        print(f"  Signal length: {len(ecg_signal)} samples ({len(ecg_signal)/sampling_rate:.1f} seconds)")
        
        # Preprocess (same as in run_inverted_qtdb.py)
        from pyhearts import PyHEARTS
        processor = PyHEARTS(sampling_rate=sampling_rate, species="human", verbose=False, plot=False)
        ecg_filtered = processor.preprocess_signal(
            ecg_signal=ecg_signal,
            highpass_cutoff=0.5,
            filter_order=4,
            lowpass_cutoff=50,
            notch_frequency=60,
            quality_factor=30,
            poly_degree=5
        )
        
        print(f"  Filtered signal range: [{ecg_filtered.min():.3f}, {ecg_filtered.max():.3f}]")
        
        # Check polarity detection
        print(f"\n  [1] POLARITY DETECTION")
        is_inverted = _detect_signal_polarity(
            ecg_filtered, 
            sampling_rate, 
            min_refrac_ms=cfg.rpeak_min_refrac_ms
        )
        print(f"      Detected as inverted: {is_inverted}")
        
        # Analyze signal characteristics
        print(f"\n  [2] SIGNAL CHARACTERISTICS")
        # Find positive and negative peaks
        from scipy.signal import find_peaks
        mad = np.median(np.abs(ecg_filtered - np.median(ecg_filtered)))
        robust_std = 1.4826 * mad if mad > 1e-9 else float(np.std(ecg_filtered))
        prominence_threshold = 2.0 * robust_std
        
        distance_samples = max(1, int(round(cfg.rpeak_min_refrac_ms * sampling_rate / 1000.0)))
        
        pos_peaks, pos_props = find_peaks(
            ecg_filtered, 
            distance=distance_samples, 
            prominence=prominence_threshold
        )
        
        neg_peaks, neg_props = find_peaks(
            -ecg_filtered, 
            distance=distance_samples, 
            prominence=prominence_threshold
        )
        
        print(f"      Positive peaks found: {len(pos_peaks)}")
        print(f"      Negative peaks found: {len(neg_peaks)}")
        
        if len(pos_peaks) > 0 and len(neg_peaks) > 0:
            pos_prominences = pos_props.get('prominences', None)
            neg_prominences = neg_props.get('prominences', None)
            
            if pos_prominences is None:
                pos_prominences = ecg_filtered[pos_peaks] - np.median(ecg_filtered)
                pos_prominences = pos_prominences[pos_prominences > 0]
            if neg_prominences is None:
                neg_prominences = np.median(ecg_filtered) - ecg_filtered[neg_peaks]
                neg_prominences = neg_prominences[neg_prominences > 0]
            
            median_pos_prom = np.median(pos_prominences) if len(pos_prominences) > 0 else 0
            median_neg_prom = np.median(neg_prominences) if len(neg_prominences) > 0 else 0
            
            median_pos_amp = np.median(np.abs(ecg_filtered[pos_peaks])) if len(pos_peaks) > 0 else 0
            median_neg_amp = np.median(np.abs(ecg_filtered[neg_peaks])) if len(neg_peaks) > 0 else 0
            
            print(f"      Median positive prominence: {median_pos_prom:.4f}")
            print(f"      Median negative prominence: {median_neg_prom:.4f}")
            print(f"      Median positive amplitude: {median_pos_amp:.4f}")
            print(f"      Median negative amplitude: {median_neg_amp:.4f}")
            print(f"      Negative/Positive prominence ratio: {median_neg_prom/median_pos_prom:.2f}" if median_pos_prom > 0 else "      (no positive peaks)")
            print(f"      Negative/Positive amplitude ratio: {median_neg_amp/median_pos_amp:.2f}" if median_pos_amp > 0 else "      (no positive peaks)")
            print(f"      Should be inverted: {median_neg_prom > 1.2 * median_pos_prom and median_neg_amp > 1.2 * median_pos_amp}")
        
        # Run R-peak detection
        print(f"\n  [3] R-PEAK DETECTION")
        r_peaks = r_peak_detection(ecg_filtered, sampling_rate, cfg=cfg, sensitivity="standard")
        print(f"      Detected R-peaks: {len(r_peaks)}")
        
        if len(r_peaks) > 0:
            # Check if peaks are at positive or negative values
            peak_values = ecg_filtered[r_peaks]
            pos_peak_count = np.sum(peak_values > 0)
            neg_peak_count = np.sum(peak_values < 0)
            
            print(f"      Peaks at positive values: {pos_peak_count} ({100*pos_peak_count/len(r_peaks):.1f}%)")
            print(f"      Peaks at negative values: {neg_peak_count} ({100*neg_peak_count/len(r_peaks):.1f}%)")
            print(f"      Peak value range: [{peak_values.min():.3f}, {peak_values.max():.3f}]")
            print(f"      Mean peak value: {peak_values.mean():.3f}")
            
            # Compare with QTDB annotations
            try:
                ann = wfdb.rdann(str(record_path), 'atr')
                qtdb_r_peaks = np.array(ann.sample)
                print(f"\n  [4] COMPARISON WITH QTDB")
                print(f"      QTDB R-peaks: {len(qtdb_r_peaks)}")
                
                # Sample a few QTDB peaks to check their values
                if len(qtdb_r_peaks) > 0:
                    sample_indices = qtdb_r_peaks[:min(10, len(qtdb_r_peaks))]
                    qtdb_values = ecg_filtered[sample_indices]
                    print(f"      QTDB peak values (first 10): {qtdb_values}")
                    print(f"      QTDB peak value range: [{qtdb_values.min():.3f}, {qtdb_values.max():.3f}]")
                    print(f"      QTDB mean peak value: {qtdb_values.mean():.3f}")
                    
                    # Check timing offset
                    if len(r_peaks) > 0 and len(qtdb_r_peaks) > 0:
                        # Find closest matches
                        offsets = []
                        for qtdb_idx in qtdb_r_peaks[:100]:  # Sample first 100
                            distances = np.abs(r_peaks - qtdb_idx)
                            min_dist_idx = np.argmin(distances)
                            min_dist = distances[min_dist_idx]
                            if min_dist < sampling_rate * 0.2:  # Within 200ms
                                offset_ms = (r_peaks[min_dist_idx] - qtdb_idx) / sampling_rate * 1000
                                offsets.append(offset_ms)
                        
                        if len(offsets) > 0:
                            print(f"      Mean timing offset: {np.mean(offsets):.1f} ms (std: {np.std(offsets):.1f} ms)")
                            print(f"      Median timing offset: {np.median(offsets):.1f} ms")
            except Exception as e:
                print(f"      Could not load QTDB annotations: {e}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print("DEBUG COMPLETE")
print(f"{'='*80}")

