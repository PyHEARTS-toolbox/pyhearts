#!/usr/bin/env python3
"""
Re-assess T, P, S, and Q Peak Detection: PyHEARTS vs ECGpuwave
With Major Outlier Removal

This script runs a fresh comparison between PyHEARTS and ECGpuwave,
matches peaks beat-by-beat, removes outliers, and recalculates statistics.
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

# Add pyhearts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyhearts import PyHEARTS

# Paths
QTDB_DIR = "/Users/morganfitzgerald/Documents/pyhearts/data/qtdb/1.0.0"
ECGPUWAVE_RESULTS_DIR = "/Users/morganfitzgerald/Documents/pyhearts/results/ecgpuwave_results_20251223_1319"
OUTPUT_DIR = "/Users/morganfitzgerald/Documents/pyhearts/reviewer_response/gold_standard_comparison/peak_detection_reassessment_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test subjects (subset for testing, can be expanded)
TEST_SUBJECTS = [
    'sel853', 'sel233', 'sel45', 'sel46', 'sel811', 'sel883',
    'sel116', 'sel213', 'sel100', 'sel16539'
]

def load_ecgpuwave_annotations(subject: str) -> Dict[str, np.ndarray]:
    """Load ECGpuwave annotations from .pu file."""
    pu_file = os.path.join(ECGPUWAVE_RESULTS_DIR, f'{subject}.pu')
    if not os.path.exists(pu_file):
        return {}
    
    annotation_dir = os.path.dirname(pu_file)
    annotation_name = os.path.basename(pu_file).replace('.pu', '')
    
    old_dir = os.getcwd()
    os.chdir(annotation_dir)
    try:
        ann = wfdb.rdann(annotation_name, 'pu')
        
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
            elif symbol == 'N':
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
        print(f"  Error loading ECGpuwave annotations: {e}")
        return {}
    finally:
        os.chdir(old_dir)

def load_pyhearts_results(subject: str) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    """Load PyHEARTS results and run analysis if needed."""
    # For now, run PyHEARTS on the fly
    # In production, could load from saved results
    
    signal_file = os.path.join(QTDB_DIR, f'{subject}.dat')
    if not os.path.exists(signal_file):
        return None, None
    
    old_dir = os.getcwd()
    os.chdir(QTDB_DIR)
    try:
        record = wfdb.rdrecord(subject)
        os.chdir(old_dir)
        
        if record.p_signal is None or record.p_signal.shape[1] == 0:
            return None, record.fs
        
        signal = record.p_signal[:, 0]
        sampling_rate = float(record.fs)
        
        # Run PyHEARTS
        analyzer = PyHEARTS(sampling_rate=sampling_rate, verbose=False)
        output_df, epochs_df = analyzer.analyze_ecg(signal)
        
        return output_df, sampling_rate
    except Exception as e:
        try:
            os.chdir(old_dir)
        except:
            pass
        import traceback
        print(f"  Error running PyHEARTS: {e}")
        traceback.print_exc()
        return None, None

def match_peaks(
    pyhearts_peaks: np.ndarray,
    ecgpuwave_peaks: np.ndarray,
    sampling_rate: float,
    max_distance_ms: float = 50.0
) -> List[Tuple[int, int, float]]:
    """
    Match peaks between PyHEARTS and ECGpuwave.
    
    Returns list of (pyhearts_idx, ecgpuwave_idx, distance_ms) tuples.
    """
    if len(pyhearts_peaks) == 0 or len(ecgpuwave_peaks) == 0:
        return []
    
    max_distance_samples = int(round(max_distance_ms * sampling_rate / 1000.0))
    
    matched_pairs = []
    ecgpuwave_matched = set()
    
    for ph_idx, ph_peak in enumerate(pyhearts_peaks):
        best_ecg_idx = None
        best_distance = float('inf')
        
        for ecg_idx, ecg_peak in enumerate(ecgpuwave_peaks):
            if ecg_idx in ecgpuwave_matched:
                continue
            
            distance_samples = abs(ecg_peak - ph_peak)
            if distance_samples <= max_distance_samples and distance_samples < best_distance:
                best_distance = distance_samples
                best_ecg_idx = ecg_idx
        
        if best_ecg_idx is not None:
            distance_ms = best_distance * 1000.0 / sampling_rate
            matched_pairs.append((ph_idx, best_ecg_idx, distance_ms))
            ecgpuwave_matched.add(best_ecg_idx)
    
    return matched_pairs

def analyze_subject(subject: str) -> Optional[pd.DataFrame]:
    """Analyze a single subject and create comparison dataframe."""
    print(f"\nAnalyzing {subject}...")
    
    # Load PyHEARTS results
    pyhearts_df, sampling_rate = load_pyhearts_results(subject)
    if pyhearts_df is None or sampling_rate is None:
        print(f"  Failed to get PyHEARTS results")
        return None
    
    # Load ECGpuwave annotations
    ecgpuwave_ann = load_ecgpuwave_annotations(subject)
    if not ecgpuwave_ann or len(ecgpuwave_ann.get('R_peak', [])) == 0:
        print(f"  Failed to get ECGpuwave annotations")
        return None
    
    # Extract PyHEARTS peaks (using global indices)
    pyhearts_r = pyhearts_df['R_global_center_idx'].dropna().values.astype(int)
    pyhearts_p = pyhearts_df['P_global_center_idx'].dropna().values.astype(int) if 'P_global_center_idx' in pyhearts_df.columns else np.array([])
    pyhearts_t = pyhearts_df['T_global_center_idx'].dropna().values.astype(int) if 'T_global_center_idx' in pyhearts_df.columns else np.array([])
    pyhearts_q = pyhearts_df['Q_global_center_idx'].dropna().values.astype(int) if 'Q_global_center_idx' in pyhearts_df.columns else np.array([])
    pyhearts_s = pyhearts_df['S_global_center_idx'].dropna().values.astype(int) if 'S_global_center_idx' in pyhearts_df.columns else np.array([])
    
    ecgpuwave_r = ecgpuwave_ann.get('R_peak', np.array([]))
    ecgpuwave_p = ecgpuwave_ann.get('P_peak', np.array([]))
    ecgpuwave_t = ecgpuwave_ann.get('T_peak', np.array([]))
    ecgpuwave_q = ecgpuwave_ann.get('Q_peak', np.array([]))
    ecgpuwave_s = ecgpuwave_ann.get('S_peak', np.array([]))
    
    # Match R peaks first (used as reference)
    r_matches = match_peaks(pyhearts_r, ecgpuwave_r, sampling_rate, max_distance_ms=50.0)
    
    if len(r_matches) == 0:
        print(f"  No R peaks matched")
        return None
    
    # Match other peaks within the same beats (using R peaks as boundaries)
    results = []
    
    # For each matched R peak, match other peaks in that beat
    for r_match_idx, (ph_r_idx, ecg_r_idx, r_distance_ms) in enumerate(r_matches):
        ph_r_peak = pyhearts_r[ph_r_idx]
        ecg_r_peak = ecgpuwave_r[ecg_r_idx]
        
        # Define beat boundaries (from this R to next R, or use fixed window)
        next_ph_r_peak = pyhearts_r[ph_r_idx + 1] if ph_r_idx + 1 < len(pyhearts_r) else ph_r_peak + int(round(2000 * sampling_rate / 1000.0))
        next_ecg_r_peak = ecgpuwave_r[ecg_r_idx + 1] if ecg_r_idx + 1 < len(ecgpuwave_r) else ecg_r_peak + int(round(2000 * sampling_rate / 1000.0))
        
        result_row = {
            'subject': subject,
            'beat_idx': r_match_idx,
            'pyhearts_r': ph_r_peak,
            'ecgpuwave_r': ecg_r_peak,
            'r_error_ms': r_distance_ms,
        }
        
        # Match P peaks (before R)
        ph_p_in_beat = pyhearts_p[(pyhearts_p < ph_r_peak) & (pyhearts_p > ph_r_peak - int(round(500 * sampling_rate / 1000.0)))]
        ecg_p_in_beat = ecgpuwave_p[(ecgpuwave_p < ecg_r_peak) & (ecgpuwave_p > ecg_r_peak - int(round(500 * sampling_rate / 1000.0)))]
        
        if len(ph_p_in_beat) > 0 and len(ecg_p_in_beat) > 0:
            p_matches = match_peaks(ph_p_in_beat, ecg_p_in_beat, sampling_rate, max_distance_ms=100.0)
            if len(p_matches) > 0:
                _, _, p_distance_ms = p_matches[0]
                result_row['pyhearts_p'] = ph_p_in_beat[p_matches[0][0]]
                result_row['ecgpuwave_p'] = ecg_p_in_beat[p_matches[0][1]]
                result_row['p_error_ms'] = p_distance_ms
                result_row['p_detected'] = True
            else:
                result_row['p_detected'] = False
        else:
            result_row['p_detected'] = len(ph_p_in_beat) > 0
        
        # Match T peaks (after R)
        ph_t_in_beat = pyhearts_t[(pyhearts_t > ph_r_peak) & (pyhearts_t < next_ph_r_peak)]
        ecg_t_in_beat = ecgpuwave_t[(ecgpuwave_t > ecg_r_peak) & (ecgpuwave_t < next_ecg_r_peak)]
        
        if len(ph_t_in_beat) > 0 and len(ecg_t_in_beat) > 0:
            t_matches = match_peaks(ph_t_in_beat, ecg_t_in_beat, sampling_rate, max_distance_ms=100.0)
            if len(t_matches) > 0:
                _, _, t_distance_ms = t_matches[0]
                result_row['pyhearts_t'] = ph_t_in_beat[t_matches[0][0]]
                result_row['ecgpuwave_t'] = ecg_t_in_beat[t_matches[0][1]]
                result_row['t_error_ms'] = t_distance_ms
                result_row['t_detected'] = True
            else:
                result_row['t_detected'] = False
        else:
            result_row['t_detected'] = len(ph_t_in_beat) > 0
        
        # Match Q peaks (before R, close to R)
        ph_q_in_beat = pyhearts_q[(pyhearts_q < ph_r_peak) & (pyhearts_q > ph_r_peak - int(round(150 * sampling_rate / 1000.0)))]
        ecg_q_in_beat = ecgpuwave_q[(ecgpuwave_q < ecg_r_peak) & (ecgpuwave_q > ecg_r_peak - int(round(150 * sampling_rate / 1000.0)))]
        
        if len(ph_q_in_beat) > 0 and len(ecg_q_in_beat) > 0:
            q_matches = match_peaks(ph_q_in_beat, ecg_q_in_beat, sampling_rate, max_distance_ms=50.0)
            if len(q_matches) > 0:
                _, _, q_distance_ms = q_matches[0]
                result_row['pyhearts_q'] = ph_q_in_beat[q_matches[0][0]]
                result_row['ecgpuwave_q'] = ecg_q_in_beat[q_matches[0][1]]
                result_row['q_error_ms'] = q_distance_ms
                result_row['q_detected'] = True
            else:
                result_row['q_detected'] = False
        else:
            result_row['q_detected'] = len(ph_q_in_beat) > 0
        
        # Match S peaks (after R, close to R)
        ph_s_in_beat = pyhearts_s[(pyhearts_s > ph_r_peak) & (pyhearts_s < ph_r_peak + int(round(150 * sampling_rate / 1000.0)))]
        ecg_s_in_beat = ecgpuwave_s[(ecgpuwave_s > ecg_r_peak) & (ecgpuwave_s < ecg_r_peak + int(round(150 * sampling_rate / 1000.0)))]
        
        if len(ph_s_in_beat) > 0 and len(ecg_s_in_beat) > 0:
            s_matches = match_peaks(ph_s_in_beat, ecg_s_in_beat, sampling_rate, max_distance_ms=50.0)
            if len(s_matches) > 0:
                _, _, s_distance_ms = s_matches[0]
                result_row['pyhearts_s'] = ph_s_in_beat[s_matches[0][0]]
                result_row['ecgpuwave_s'] = ecg_s_in_beat[s_matches[0][1]]
                result_row['s_error_ms'] = s_distance_ms
                result_row['s_detected'] = True
            else:
                result_row['s_detected'] = False
        else:
            result_row['s_detected'] = len(ph_s_in_beat) > 0
        
        results.append(result_row)
    
    if len(results) == 0:
        return None
    
    df = pd.DataFrame(results)
    print(f"  Matched {len(results)} beats")
    
    return df

def calculate_statistics_with_outlier_removal(
    errors: np.ndarray,
    detected: np.ndarray,
    outlier_thresholds: List[float] = [200.0, 150.0, 100.0]
) -> Dict[str, Dict]:
    """Calculate statistics with different outlier removal thresholds."""
    stats_by_threshold = {}
    
    # All data (no outlier removal)
    detected_errors_all = errors[detected]
    if len(detected_errors_all) > 0:
        stats_by_threshold['all'] = {
            'n': len(detected_errors_all),
            'mean_error': np.mean(np.abs(detected_errors_all)),
            'median_error': np.median(np.abs(detected_errors_all)),
            'std_error': np.std(detected_errors_all),
            'within_20ms': (np.abs(detected_errors_all) <= 20).sum() / len(detected_errors_all) * 100,
            'within_50ms': (np.abs(detected_errors_all) <= 50).sum() / len(detected_errors_all) * 100,
        }
    
    # With outlier removal
    for threshold in outlier_thresholds:
        mask = np.abs(errors) <= threshold
        filtered_detected = detected & mask
        filtered_errors = errors[filtered_detected]
        
        if len(filtered_errors) > 0:
            outliers_removed = detected.sum() - filtered_detected.sum()
            stats_by_threshold[f'threshold_{threshold}ms'] = {
                'n': len(filtered_errors),
                'outliers_removed': outliers_removed,
                'outliers_removed_pct': outliers_removed / detected.sum() * 100 if detected.sum() > 0 else 0.0,
                'mean_error': np.mean(np.abs(filtered_errors)),
                'median_error': np.median(np.abs(filtered_errors)),
                'std_error': np.std(filtered_errors),
                'within_20ms': (np.abs(filtered_errors) <= 20).sum() / len(filtered_errors) * 100,
                'within_50ms': (np.abs(filtered_errors) <= 50).sum() / len(filtered_errors) * 100,
            }
    
    return stats_by_threshold

def main():
    print("="*80)
    print("Re-assess Peak Detection: PyHEARTS vs ECGpuwave")
    print("With Outlier Removal")
    print("="*80)
    
    all_results = []
    
    for subject in TEST_SUBJECTS:
        result_df = analyze_subject(subject)
        if result_df is not None:
            all_results.append(result_df)
    
    if len(all_results) == 0:
        print("\nNo results obtained.")
        return 1
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save full comparison data
    full_data_file = os.path.join(OUTPUT_DIR, 'full_comparison_data.csv')
    combined_df.to_csv(full_data_file, index=False)
    print(f"\n\nFull comparison data saved to: {full_data_file}")
    print(f"Total beats analyzed: {len(combined_df)}")
    
    # Analyze each peak type with outlier removal
    peak_types = ['P', 'T', 'Q', 'S']
    outlier_thresholds = [200.0, 150.0, 100.0]
    
    print("\n" + "="*80)
    print("STATISTICS WITH OUTLIER REMOVAL")
    print("="*80)
    
    summary_rows = []
    
    for peak_type in peak_types:
        error_col = f'{peak_type.lower()}_error_ms'
        detected_col = f'{peak_type.lower()}_detected'
        
        if error_col not in combined_df.columns:
            continue
        
        print(f"\n{peak_type} Peak Detection:")
        print("-" * 60)
        
        errors = combined_df[error_col].fillna(0).values
        detected = combined_df[detected_col].fillna(False).values
        
        # Calculate statistics for all data and with outlier removal
        stats_dict = calculate_statistics_with_outlier_removal(
            errors, detected, outlier_thresholds
        )
        
        for threshold_name, stats in stats_dict.items():
            if threshold_name == 'all':
                print(f"\n  All data (no outlier removal):")
            else:
                threshold_ms = threshold_name.replace('threshold_', '').replace('ms', '')
                print(f"\n  Outlier removal: > {threshold_ms} ms")
                print(f"    Outliers removed: {stats['outliers_removed']} ({stats['outliers_removed_pct']:.1f}%)")
            
            print(f"    N: {stats['n']}")
            print(f"    Mean error: {stats['mean_error']:.2f} ms")
            print(f"    Median error: {stats['median_error']:.2f} ms")
            print(f"    % within ±20ms: {stats['within_20ms']:.1f}%")
            print(f"    % within ±50ms: {stats['within_50ms']:.1f}%")
            
            summary_rows.append({
                'peak': peak_type,
                'outlier_method': threshold_name,
                'n': stats['n'],
                'mean_error_ms': stats['mean_error'],
                'median_error_ms': stats['median_error'],
                'std_error_ms': stats['std_error'],
                'within_20ms_pct': stats['within_20ms'],
                'within_50ms_pct': stats['within_50ms'],
                'outliers_removed': stats.get('outliers_removed', 0),
                'outliers_removed_pct': stats.get('outliers_removed_pct', 0.0),
            })
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary_rows)
    summary_file = os.path.join(OUTPUT_DIR, 'summary_with_outlier_removal.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\n\nSummary saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

