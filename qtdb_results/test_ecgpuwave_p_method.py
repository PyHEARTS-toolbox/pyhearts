#!/usr/bin/env python3
"""
Test fixed-window P wave method on subset of QTDB subjects.

This script:
1. Runs PyHEARTS on a subset of QTDB subjects with p_use_fixed_window_method=True
2. Compares P wave detection results to ground truth
3. Generates a comparison report showing improvement
"""

import os
import sys
import numpy as np
import pandas as pd
import wfdb
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import pyhearts
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyhearts import PyHEARTS
from pyhearts.config import ProcessCycleConfig

# Paths
QTDB_RESULTS_DIR = Path(__file__).parent
QTDB_DATA_DIR = Path(__file__).parent.parent / "data" / "qtdb" / "1.0.0"
OUTPUT_DIR = QTDB_RESULTS_DIR / "fixed_window_p_method_test"
OUTPUT_DIR.mkdir(exist_ok=True)

# Test on a diverse subset of subjects
TEST_SUBJECTS = [
    "sel30",   # Standard subject
    "sel100",  # Different subject
    "sel117",  # Known to have inverted T waves
    "sel213",  # Another subject
    "sel301",  # Another subject
]

# Tolerance for matching peaks (in milliseconds)
PEAK_MATCH_TOLERANCE_MS = 50.0


def load_qtdb_annotations(subject: str) -> Dict[str, np.ndarray]:
    """Load QTDB manual annotations from .q1c file (ground truth)."""
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
            'P_peak': [],
            'R_peak': [],
            'T_peak': [],
            'Q_peak': [],
            'S_peak': [],
        }
        
        for i, symbol in enumerate(ann.symbol):
            sample = ann.sample[i]
            if sample < 0:
                continue
            
            symbol_lower = symbol.lower()
            if symbol_lower == 'p':
                annotations['P_peak'].append(sample)
            elif symbol == 'N':  # Normal beat (R peak)
                annotations['R_peak'].append(sample)
            elif symbol_lower == 't':
                annotations['T_peak'].append(sample)
            elif symbol_lower == 'q':
                annotations['Q_peak'].append(sample)
            elif symbol_lower == 's':
                annotations['S_peak'].append(sample)
        
        # Convert to numpy arrays
        for key in annotations:
            annotations[key] = np.array(annotations[key], dtype=int)
        
        return annotations
    except Exception as e:
        print(f"  Error loading QTDB annotations: {e}")
        return {}
    finally:
        os.chdir(old_dir)


def load_ecg_signal(subject: str) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Load ECG signal from QTDB."""
    dat_file = QTDB_DATA_DIR / f"{subject}.dat"
    if not dat_file.exists():
        return None, None
    
    annotation_dir = dat_file.parent
    annotation_name = dat_file.stem
    
    old_dir = os.getcwd()
    os.chdir(annotation_dir)
    try:
        record = wfdb.rdrecord(annotation_name)
        signal = record.p_signal[:, 0]  # Use first channel
        sampling_rate = record.fs
        
        return signal, sampling_rate
    except Exception as e:
        print(f"  Error loading ECG signal: {e}")
        return None, None
    finally:
        os.chdir(old_dir)


def extract_pyhearts_peaks(df: pd.DataFrame, sampling_rate: float) -> Dict[str, np.ndarray]:
    """Extract peak indices from PyHEARTS output DataFrame."""
    peaks = {
        'P_peak': [],
        'R_peak': [],
        'T_peak': [],
        'Q_peak': [],
        'S_peak': [],
    }
    
    if df is None or len(df) == 0:
        return peaks
    
    # PyHEARTS stores peaks in columns like 'P_center_idx', 'R_center_idx', etc.
    # These are cycle-relative indices, need to map to global indices
    # For now, we'll use the cycle indices and map them
    
    for cycle_idx, row in df.iterrows():
        # Get global indices from cycle data if available
        # This is simplified - actual implementation may vary
        if 'P_center_idx' in row and pd.notna(row['P_center_idx']):
            # This is cycle-relative, would need cycle start index to convert
            # For now, we'll skip this and use a different approach
            pass
    
    # Alternative: Extract from epochs_df if available
    # For this test, we'll process the signal and extract peaks directly
    return peaks


def match_beats(
    qtdb_r: np.ndarray,
    method_r: np.ndarray,
    sampling_rate: float,
    max_distance_ms: float = 50.0
) -> List[Tuple[int, int]]:
    """Match R peaks between QTDB and PyHEARTS."""
    if len(qtdb_r) == 0 or len(method_r) == 0:
        return []
    
    max_distance_samples = int(round(max_distance_ms * sampling_rate / 1000.0))
    
    matched_pairs = []
    method_matched = set()
    
    for qtdb_idx, qtdb_r_peak in enumerate(qtdb_r):
        best_method_idx = None
        best_distance = float('inf')
        
        for method_idx, method_r_peak in enumerate(method_r):
            if method_idx in method_matched:
                continue
            
            distance_samples = abs(method_r_peak - qtdb_r_peak)
            if distance_samples <= max_distance_samples and distance_samples < best_distance:
                best_distance = distance_samples
                best_method_idx = method_idx
        
        if best_method_idx is not None:
            matched_pairs.append((qtdb_idx, best_method_idx))
            method_matched.add(best_method_idx)
    
    return matched_pairs


def compare_p_peaks(
    qtdb_p: np.ndarray,
    pyhearts_p: np.ndarray,
    matched_r_pairs: List[Tuple[int, int]],
    qtdb_r: np.ndarray,
    pyhearts_r: np.ndarray,
    sampling_rate: float,
    max_distance_ms: float = 150.0
) -> Dict[str, float]:
    """Compare P peak detections."""
    if len(matched_r_pairs) == 0:
        return {
            'detection_rate': 0.0,
            'mae_ms': float('nan'),
            'median_error_ms': float('nan'),
            'matched_count': 0,
            'total_beats': 0,
        }
    
    max_distance_samples = int(round(max_distance_ms * sampling_rate / 1000.0))
    
    matched_p_count = 0
    errors_ms = []
    
    for qtdb_r_idx, pyhearts_r_idx in matched_r_pairs:
        if qtdb_r_idx >= len(qtdb_r) or pyhearts_r_idx >= len(pyhearts_r):
            continue
        
        qtdb_r_peak = qtdb_r[qtdb_r_idx]
        pyhearts_r_peak = pyhearts_r[pyhearts_r_idx]
        
        # Find P peaks before this R peak
        qtdb_p_candidates = qtdb_p[(qtdb_p < qtdb_r_peak) & 
                                   (qtdb_r_peak - qtdb_p <= max_distance_samples)]
        pyhearts_p_candidates = pyhearts_p[(pyhearts_p < pyhearts_r_peak) & 
                                            (pyhearts_r_peak - pyhearts_p <= max_distance_samples)]
        
        if len(qtdb_p_candidates) > 0:
            # QTDB has a P peak for this beat
            qtdb_p_peak = qtdb_p_candidates[-1]  # Closest to R
            
            if len(pyhearts_p_candidates) > 0:
                # PyHEARTS also detected a P peak
                pyhearts_p_peak = pyhearts_p_candidates[-1]  # Closest to R
                
                # Calculate error in ms
                error_samples = abs(pyhearts_p_peak - qtdb_p_peak)
                error_ms = (error_samples / sampling_rate) * 1000.0
                
                if error_ms <= PEAK_MATCH_TOLERANCE_MS:
                    matched_p_count += 1
                    errors_ms.append(error_ms)
            # else: PyHEARTS missed this P peak
    
    total_beats_with_p = len([pair for pair in matched_r_pairs 
                              if pair[0] < len(qtdb_r) and 
                              len(qtdb_p[(qtdb_p < qtdb_r[pair[0]]) & 
                                        (qtdb_r[pair[0]] - qtdb_p <= max_distance_samples)]) > 0])
    
    detection_rate = (matched_p_count / total_beats_with_p * 100.0) if total_beats_with_p > 0 else 0.0
    mae_ms = np.mean(errors_ms) if len(errors_ms) > 0 else float('nan')
    median_error_ms = np.median(errors_ms) if len(errors_ms) > 0 else float('nan')
    
    return {
        'detection_rate': detection_rate,
        'mae_ms': mae_ms,
        'median_error_ms': median_error_ms,
        'matched_count': matched_p_count,
        'total_beats': total_beats_with_p,
    }


def process_subject(subject: str, use_fixed_window_method: bool = True) -> Dict:
    """Process a single subject with PyHEARTS."""
    print(f"\n{'='*60}")
    print(f"Processing subject: {subject}")
    print(f"Fixed-window P method: {use_fixed_window_method}")
    print(f"{'='*60}")
    
    # Load ground truth
    qtdb_ann = load_qtdb_annotations(subject)
    if not qtdb_ann or len(qtdb_ann.get('R_peak', [])) == 0:
        print(f"  Skipping {subject}: No ground truth annotations")
        return None
    
    # Load ECG signal
    signal, sampling_rate = load_ecg_signal(subject)
    if signal is None:
        print(f"  Skipping {subject}: Could not load ECG signal")
        return None
    
    print(f"  Signal length: {len(signal)} samples")
    print(f"  Sampling rate: {sampling_rate} Hz")
    print(f"  QTDB R peaks: {len(qtdb_ann['R_peak'])}")
    print(f"  QTDB P peaks: {len(qtdb_ann['P_peak'])}")
    
    # Create PyHEARTS analyzer with fixed-window method enabled
    cfg = ProcessCycleConfig.for_human()
    if use_fixed_window_method:
        # Ensure fixed-window method is enabled
        cfg = ProcessCycleConfig.for_human()  # Already has p_use_fixed_window_method=True
    else:
        # Disable it for comparison
        from dataclasses import replace
        cfg = replace(ProcessCycleConfig.for_human(), p_use_fixed_window_method=False)
    
    analyzer = PyHEARTS(
        sampling_rate=sampling_rate,
        species="human",
        verbose=False,
        cfg=cfg,
    )
    
    # Process ECG
    try:
        output_df, epochs_df = analyzer.analyze_ecg(signal)
        
        # Extract peaks from PyHEARTS output
        # PyHEARTS stores global indices in columns like 'P_global_center_idx', 'R_global_center_idx'
        pyhearts_r_peaks = []
        pyhearts_p_peaks = []
        
        # Extract R peaks
        r_col = 'R_global_center_idx'
        if r_col in output_df.columns:
            pyhearts_r_peaks = output_df[r_col].dropna().values.astype(int).tolist()
        else:
            print(f"  Warning: {r_col} not found in output columns: {list(output_df.columns)}")
        
        # Extract P peaks
        p_col = 'P_global_center_idx'
        if p_col in output_df.columns:
            pyhearts_p_peaks = output_df[p_col].dropna().values.astype(int).tolist()
        else:
            print(f"  Warning: {p_col} not found in output columns: {list(output_df.columns)}")
        
        print(f"  PyHEARTS R peaks: {len(pyhearts_r_peaks)}")
        print(f"  PyHEARTS P peaks: {len(pyhearts_p_peaks)}")
        
        # Match R peaks
        matched_r_pairs = match_beats(
            qtdb_ann['R_peak'],
            np.array(pyhearts_r_peaks),
            sampling_rate
        )
        print(f"  Matched R peaks: {len(matched_r_pairs)}")
        
        # Compare P peaks
        p_comparison = compare_p_peaks(
            qtdb_ann['P_peak'],
            np.array(pyhearts_p_peaks),
            matched_r_pairs,
            qtdb_ann['R_peak'],
            np.array(pyhearts_r_peaks),
            sampling_rate
        )
        
        print(f"  P Detection Rate: {p_comparison['detection_rate']:.1f}%")
        print(f"  P MAE: {p_comparison['mae_ms']:.2f} ms")
        print(f"  P Median Error: {p_comparison['median_error_ms']:.2f} ms")
        
        return {
            'subject': subject,
            'sampling_rate': sampling_rate,
            'qtdb_r_count': len(qtdb_ann['R_peak']),
            'qtdb_p_count': len(qtdb_ann['P_peak']),
            'pyhearts_r_count': len(pyhearts_r_peaks),
            'pyhearts_p_count': len(pyhearts_p_peaks),
            'matched_r_count': len(matched_r_pairs),
            'p_detection_rate': p_comparison['detection_rate'],
            'p_mae_ms': p_comparison['mae_ms'],
            'p_median_error_ms': p_comparison['median_error_ms'],
            'p_matched_count': p_comparison['matched_count'],
            'p_total_beats': p_comparison['total_beats'],
            'use_fixed_window_method': use_fixed_window_method,
        }
        
    except Exception as e:
        print(f"  Error processing {subject}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run test on subset of subjects."""
    print("="*60)
    print("Testing Fixed-Window P Wave Detection Method")
    print("="*60)
    
    results = []
    
    for subject in TEST_SUBJECTS:
        result = process_subject(subject, use_fixed_window_method=True)
        if result:
            results.append(result)
    
    if len(results) == 0:
        print("\nNo results to report!")
        return
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_csv = OUTPUT_DIR / "test_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Subjects processed: {len(results)}")
    print(f"\nP Wave Detection:")
    print(f"  Average Detection Rate: {df['p_detection_rate'].mean():.1f}%")
    print(f"  Average MAE: {df['p_mae_ms'].mean():.2f} ms")
    print(f"  Average Median Error: {df['p_median_error_ms'].mean():.2f} ms")
    print(f"\nPer-subject results:")
    for _, row in df.iterrows():
        print(f"  {row['subject']}: {row['p_detection_rate']:.1f}% detection, "
              f"{row['p_mae_ms']:.2f} ms MAE")
    
    # Save summary report
    summary_file = OUTPUT_DIR / "summary_report.txt"
    with open(summary_file, 'w') as f:
        f.write("Fixed-Window P Wave Method Test Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Subjects processed: {len(results)}\n")
        f.write(f"\nP Wave Detection:\n")
        f.write(f"  Average Detection Rate: {df['p_detection_rate'].mean():.1f}%\n")
        f.write(f"  Average MAE: {df['p_mae_ms'].mean():.2f} ms\n")
        f.write(f"  Average Median Error: {df['p_median_error_ms'].mean():.2f} ms\n")
        f.write(f"\nPer-subject results:\n")
        for _, row in df.iterrows():
            f.write(f"  {row['subject']}: {row['p_detection_rate']:.1f}% detection, "
                   f"{row['p_mae_ms']:.2f} ms MAE, "
                   f"{row['p_matched_count']}/{row['p_total_beats']} matched\n")
    
    print(f"\nSummary report saved to: {summary_file}")


if __name__ == "__main__":
    main()

