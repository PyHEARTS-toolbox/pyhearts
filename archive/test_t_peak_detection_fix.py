#!/usr/bin/env python3
"""
Test T Peak Detection After Gaussian Fitting Fix

This script tests T peak detection on a subset of QTDB subjects
after fixing the Gaussian fitting "x0 is infeasible" bug.
"""

import os
import sys
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add pyhearts to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyhearts import PyHEARTS

# Paths
QTDB_DATA_DIR = Path(__file__).parent.parent / "data" / "qtdb" / "1.0.0"
OUTPUT_DIR = Path(__file__).parent / "t_detection_test_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Test subjects - mix of subjects that previously had issues
TEST_SUBJECTS = [
    'sel100',  # Previously: 0 T peaks
    'sel102',  # Previously: 0 T peaks
    'sel103',  # Previously: 0 T peaks
    'sel114',  # Previously: 0 T peaks
    'sel116',  # Previously: 1167 T peaks (working case)
    'sel117',  # Test case
    'sel123',  # Test case
]


def load_ecg_signal(subject: str) -> Tuple[Optional[np.ndarray], float]:
    """Load ECG signal from QT Database."""
    try:
        old_dir = os.getcwd()
        os.chdir(QTDB_DATA_DIR)
        record = wfdb.rdrecord(subject)
        os.chdir(old_dir)
        
        if record.p_signal is None or record.p_signal.shape[1] == 0:
            return None, record.fs
        
        signal = record.p_signal[:, 0]
        sampling_rate = float(record.fs)
        return signal, sampling_rate
    except Exception as e:
        try:
            os.chdir(old_dir)
        except:
            pass
        return None, 250.0


def load_manual_annotations(subject: str) -> Dict[str, np.ndarray]:
    """Load QTDB manual annotations from .q1c file."""
    q1c_file = QTDB_DATA_DIR / f"{subject}.q1c"
    if not q1c_file.exists():
        return {}
    
    annotation_dir = q1c_file.parent
    annotation_name = q1c_file.stem
    
    old_dir = os.getcwd()
    os.chdir(annotation_dir)
    try:
        ann = wfdb.rdann(annotation_name, 'q1c')
        
        annotations = {
            'R_peak': [],
            'T_peak': [],
        }
        
        for i, symbol in enumerate(ann.symbol):
            sample = ann.sample[i]
            if sample < 0:
                continue
            
            if symbol == 'N':  # Normal beat (R peak)
                annotations['R_peak'].append(sample)
            elif symbol.lower() == 't':
                annotations['T_peak'].append(sample)
        
        # Convert to numpy arrays
        for key in annotations:
            annotations[key] = np.array(annotations[key], dtype=int)
        
        return annotations
    except Exception as e:
        print(f"  Error loading manual annotations: {e}")
        return {}
    finally:
        os.chdir(old_dir)


def test_subject(subject: str) -> Dict:
    """Test T peak detection for a subject."""
    print(f"\n{'='*60}")
    print(f"Testing {subject}")
    print(f"{'='*60}")
    
    # Load signal
    signal, sampling_rate = load_ecg_signal(subject)
    if signal is None:
        print(f"  Failed to load signal")
        return {'subject': subject, 'error': 'signal_load_failed'}
    
    # Load manual annotations
    manual_ann = load_manual_annotations(subject)
    manual_r_count = len(manual_ann.get('R_peak', []))
    manual_t_count = len(manual_ann.get('T_peak', []))
    
    print(f"  Manual annotations: {manual_r_count} R peaks, {manual_t_count} T peaks")
    
    # Run PyHEARTS
    try:
        analyzer = PyHEARTS(
            sampling_rate=sampling_rate,
            verbose=False,
            plot=False,
        )
        
        output_df, epochs_df = analyzer.analyze_ecg(signal)
        
        if output_df is None or len(output_df) == 0:
            print(f"  PyHEARTS returned no results")
            return {'subject': subject, 'error': 'no_results'}
        
        # Statistics
        total_cycles = len(output_df)
        r_detected = output_df['R_global_center_idx'].notna().sum()
        s_detected = output_df['S_global_center_idx'].notna().sum()
        t_detected = output_df['T_global_center_idx'].notna().sum()
        
        print(f"  PyHEARTS results:")
        print(f"    Total cycles: {total_cycles}")
        print(f"    R peaks detected: {r_detected} ({100*r_detected/total_cycles:.1f}%)")
        print(f"    S peaks detected: {s_detected} ({100*s_detected/total_cycles:.1f}%)")
        print(f"    T peaks detected: {t_detected} ({100*t_detected/total_cycles:.1f}%)")
        
        # Compare to manual annotations
        if manual_r_count > 0:
            r_detection_rate = r_detected / manual_r_count if manual_r_count > 0 else 0
            print(f"    R detection rate vs manual: {100*r_detection_rate:.1f}%")
        
        if manual_t_count > 0:
            t_detection_rate = t_detected / manual_t_count if manual_t_count > 0 else 0
            print(f"    T detection rate vs manual: {100*t_detection_rate:.1f}%")
        
        return {
            'subject': subject,
            'total_cycles': total_cycles,
            'r_detected': r_detected,
            's_detected': s_detected,
            't_detected': t_detected,
            'r_rate': r_detected / total_cycles if total_cycles > 0 else 0,
            's_rate': s_detected / total_cycles if total_cycles > 0 else 0,
            't_rate': t_detected / total_cycles if total_cycles > 0 else 0,
            'manual_r_count': manual_r_count,
            'manual_t_count': manual_t_count,
            'r_detection_rate_vs_manual': r_detected / manual_r_count if manual_r_count > 0 else 0,
            't_detection_rate_vs_manual': t_detected / manual_t_count if manual_t_count > 0 else 0,
        }
        
    except Exception as e:
        print(f"  Error running PyHEARTS: {e}")
        import traceback
        traceback.print_exc()
        return {'subject': subject, 'error': str(e)}


def main():
    """Main function."""
    print("Testing T Peak Detection After Gaussian Fitting Fix")
    print("=" * 60)
    print(f"Testing {len(TEST_SUBJECTS)} subjects")
    
    results = []
    for subject in TEST_SUBJECTS:
        result = test_subject(subject)
        if result:
            results.append(result)
    
    # Summary
    if results:
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        
        summary_df = pd.DataFrame(results)
        print(summary_df.to_string(index=False))
        
        # Calculate aggregate statistics
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            total_t_detected = sum(r['t_detected'] for r in valid_results)
            total_cycles = sum(r['total_cycles'] for r in valid_results)
            total_manual_t = sum(r.get('manual_t_count', 0) for r in valid_results)
            
            print(f"\nAggregate Statistics:")
            print(f"  Total cycles: {total_cycles}")
            print(f"  Total T peaks detected: {total_t_detected} ({100*total_t_detected/total_cycles:.1f}% of cycles)")
            if total_manual_t > 0:
                print(f"  T detection rate vs manual: {100*total_t_detected/total_manual_t:.1f}%")
            
            # Compare to previous results
            print(f"\nComparison to Previous Results:")
            print(f"  Previously, sel100 had 0 T peaks detected")
            print(f"  Previously, sel116 had 1167 T peaks detected (100%)")
            
            sel100_result = next((r for r in valid_results if r['subject'] == 'sel100'), None)
            sel116_result = next((r for r in valid_results if r['subject'] == 'sel116'), None)
            
            if sel100_result:
                print(f"  After fix, sel100: {sel100_result['t_detected']} T peaks detected")
            if sel116_result:
                print(f"  After fix, sel116: {sel116_result['t_detected']} T peaks detected")
        
        # Save results
        summary_df.to_csv(OUTPUT_DIR / "t_detection_test_results.csv", index=False)
        print(f"\nResults saved to: {OUTPUT_DIR / 't_detection_test_results.csv'}")


if __name__ == "__main__":
    main()

