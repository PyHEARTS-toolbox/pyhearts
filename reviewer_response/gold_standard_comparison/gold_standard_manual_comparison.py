#!/usr/bin/env python3
"""
Gold Standard Comparison Analysis: PyHEARTS vs Manual Annotations (QT Database)

This script compares PyHEARTS detection results to manual annotations from the QT Database,
providing:
1. Mean absolute deviation of fiducial points vs. manual annotations
2. Bland-Altman plots for key intervals (PR, QRS, QT, RT, TT)
3. Demonstration that PyHEARTS hits clinically acceptable error bounds

Uses QT Database manual annotations (.q1c files from expert annotator 1) as gold standard.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Paths
PYHEARTS_RESULTS_DIR = "/Users/morganfitzgerald/Documents/qtdb/qtdb_20251231_140913"
MANUAL_ANNOTATIONS_DIR = "/Users/morganfitzgerald/Documents/pyhearts/data/qtdb/1.0.0"
OUTPUT_DIR = "/Users/morganfitzgerald/Documents/pyhearts/reviewer_response/gold_standard_comparison/pyhearts_vs_manual"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clinical error bounds (in milliseconds)
CLINICAL_BOUNDS = {
    'PR': 40.0,  # ±40ms for PR interval
    'QRS': 20.0,  # ±20ms for QRS duration
    'QT': 50.0,  # ±50ms for QT interval
    'RT': 50.0,  # ±50ms for RT interval (R peak to T peak)
    'TT': 100.0,  # ±100ms for TT interval (T peak to next T peak)
    'fiducial': 20.0,  # ±20ms for fiducial point detection
}

def remove_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Remove outliers using IQR method."""
    if len(series) < 4:
        return series
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return series[(series >= lower_bound) & (series <= upper_bound)]

def load_manual_annotations(record_name: str, annotation_file: str = 'q1c', sampling_rate: float = 250.0) -> List[Dict]:
    """
    Load manual annotations from QT Database .q1c file flexibly.
    
    Handles variable annotation formats per beat:
    - R peak ('N') is required for a valid beat
    - P onset/peak, T peak/end are optional and checked per beat
    - Different formats: (p)(N)t), (N)t), (p)(N)(t), etc.
    
    Returns list of beat dictionaries with annotations in samples.
    Only R peak is guaranteed to be present; other fields may be None.
    """
    annotation_path = os.path.join(MANUAL_ANNOTATIONS_DIR, f"{record_name}.{annotation_file}")
    
    if not os.path.exists(annotation_path):
        return []
    
    # Read from local file - need to change to directory first
    try:
        old_dir = os.getcwd()
        os.chdir(MANUAL_ANNOTATIONS_DIR)
        ann = wfdb.rdann(record_name, annotation_file)
        os.chdir(old_dir)
    except Exception as e:
        try:
            os.chdir(old_dir)
        except:
            pass
        return []
    
    beats = []
    i = 0
    
    # Find all R peaks first (required for valid beats)
    r_peak_indices = []
    for j in range(len(ann.symbol)):
        if ann.symbol[j] == 'N':  # Normal beat
            r_peak_indices.append(j)
    
    # For each R peak, look for associated annotations
    for r_idx in r_peak_indices:
        beat = {
            'r_peak': ann.sample[r_idx],
            'p_onset': None,
            'p_peak': None,
            't_peak': None,
            't_end': None
        }
        
        # Look backwards from R peak for P wave annotations
        # P wave annotations come before R peak
        # Pattern: (p) where '(' is P onset and 'p' is P peak
        # Search backwards from R peak, but not too far (within same beat)
        for j in range(r_idx - 1, max(-1, r_idx - 20), -1):
            if ann.symbol[j] == 'p':
                beat['p_peak'] = ann.sample[j]
                # Look for P onset '(' before P peak (immediately before or close)
                for k in range(max(0, j - 5), j):
                    if ann.symbol[k] == '(':
                        # Check this '(' is not closing something else
                        # If there's a 'p' right after, it's P onset
                        if k + 1 == j:  # '(' immediately before 'p'
                            beat['p_onset'] = ann.sample[k]
                        else:
                            # Check if there's a 'p' between k and j
                            has_p_between = any(ann.symbol[m] == 'p' for m in range(k + 1, j))
                            if not has_p_between:
                                beat['p_onset'] = ann.sample[k]
                        if beat['p_onset'] is not None:
                            break
                break
        
        # Look forward from R peak for T wave annotations
        # T wave annotations come after R peak
        # Pattern: t) where 't' is T peak and ')' is T end
        for j in range(r_idx + 1, min(len(ann.symbol), r_idx + 20)):
            if ann.symbol[j] in ['t', 'T']:
                beat['t_peak'] = ann.sample[j]
                # Look for T end ')' after T peak (immediately after or close)
                for k in range(j + 1, min(len(ann.symbol), j + 10)):
                    if ann.symbol[k] == ')':
                        # Check this ')' closes the T wave (should be right after 't')
                        # There might be other symbols in between, so just take first ')'
                        beat['t_end'] = ann.sample[k]
                        break
                break
        
        # Add beat (only R peak is required)
        beats.append(beat)
    
    return beats

def load_pyhearts_fiducials(csv_file: str) -> Dict[str, np.ndarray]:
    """Load PyHEARTS fiducial points from CSV."""
    df = pd.read_csv(csv_file)
    
    fiducials = {}
    
    # P wave: le_idx = onset, center_idx = peak
    for suffix, key_suffix in [('le', 'onset'), ('center', 'peak')]:
        col = f'P_global_{suffix}_idx'
        key = f'P_{key_suffix}'
        if col in df.columns:
            values = df[col].dropna().values
            fiducials[key] = np.array([int(v) for v in values if not np.isnan(v)], dtype=int)
        else:
            fiducials[key] = np.array([], dtype=int)
    
    # R peak
    col = 'R_global_center_idx'
    if col in df.columns:
        values = df[col].dropna().values
        fiducials['R_peak'] = np.array([int(v) for v in values if not np.isnan(v)], dtype=int)
    else:
        fiducials['R_peak'] = np.array([], dtype=int)
    
    # T wave: le_idx = onset, center_idx = peak, ri_idx = offset
    for suffix, key_suffix in [('le', 'onset'), ('center', 'peak'), ('ri', 'offset')]:
        col = f'T_global_{suffix}_idx'
        key = f'T_{key_suffix}'
        if col in df.columns:
            values = df[col].dropna().values
            fiducials[key] = np.array([int(v) for v in values if not np.isnan(v)], dtype=int)
        else:
            fiducials[key] = np.array([], dtype=int)
    
    return fiducials

def get_sampling_rate(meta_file: str) -> float:
    """Get sampling rate from meta.json file."""
    try:
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            return float(meta.get('sampling_rate', 250.0))
    except:
        return 250.0

def match_beats(
    manual_beats: List[Dict],
    pyhearts_r_peaks: np.ndarray,
    sampling_rate: float,
    max_match_distance_ms: float = 100.0
) -> List[Tuple[int, int]]:
    """
    Match manual annotation beats to PyHEARTS beats using R peaks.
    
    Returns list of (manual_beat_idx, pyhearts_r_idx) tuples.
    """
    if len(manual_beats) == 0 or len(pyhearts_r_peaks) == 0:
        return []
    
    max_match_distance_samples = int(round(max_match_distance_ms * sampling_rate / 1000.0))
    
    matched_pairs = []
    manual_matched = set()
    pyhearts_matched = set()
    
    # Greedy matching: sort by distance
    potential_matches = []
    for man_idx, beat in enumerate(manual_beats):
        man_r = beat['r_peak']
        for ph_idx, ph_r in enumerate(pyhearts_r_peaks):
            distance_samples = abs(ph_r - man_r)
            if distance_samples <= max_match_distance_samples:
                distance_ms = distance_samples * 1000.0 / sampling_rate
                potential_matches.append((distance_ms, man_idx, ph_idx))
    
    potential_matches.sort(key=lambda x: x[0])
    
    for distance_ms, man_idx, ph_idx in potential_matches:
        if man_idx not in manual_matched and ph_idx not in pyhearts_matched:
            matched_pairs.append((man_idx, ph_idx))
            manual_matched.add(man_idx)
            pyhearts_matched.add(ph_idx)
    
    return matched_pairs

def analyze_subject(subject: str) -> Optional[Dict]:
    """Analyze a single subject."""
    # Load PyHEARTS results
    csv_file = os.path.join(PYHEARTS_RESULTS_DIR, f'{subject}_pyhearts.csv')
    meta_file = os.path.join(PYHEARTS_RESULTS_DIR, f'{subject}_meta.json')
    
    if not os.path.exists(csv_file) or not os.path.exists(meta_file):
        return None
    
    sampling_rate = get_sampling_rate(meta_file)
    
    # Load manual annotations
    manual_beats = load_manual_annotations(subject, 'q1c', sampling_rate)
    if len(manual_beats) == 0:
        return None
    
    # Load PyHEARTS data
    df = pd.read_csv(csv_file)
    pyhearts_fiducials = load_pyhearts_fiducials(csv_file)
    pyhearts_r_peaks = pyhearts_fiducials.get('R_peak', np.array([], dtype=int))
    
    if len(pyhearts_r_peaks) == 0:
        return None
    
    # Match beats
    matched_beats = match_beats(manual_beats, pyhearts_r_peaks, sampling_rate)
    if len(matched_beats) == 0:
        return None
    
    # Calculate intervals and fiducial differences
    pr_intervals_manual = []
    pr_intervals_pyhearts = []
    qt_intervals_manual = []  # Approximate: R peak to T end
    qt_intervals_pyhearts = []
    qrs_intervals_pyhearts = []
    
    fiducial_errors = defaultdict(list)
    matched_peak_pairs = defaultdict(list)  # Store matched pairs for Bland-Altman plots
    
    for man_idx, ph_r_idx in matched_beats:
        manual_beat = manual_beats[man_idx]
        man_r = manual_beat['r_peak']
        
        # Match PyHEARTS fiducials to this beat
        ph_row_idx = None
        for idx, row in df.iterrows():
            if pd.notna(row.get('R_global_center_idx')) and int(row['R_global_center_idx']) == pyhearts_r_peaks[ph_r_idx]:
                ph_row_idx = idx
                break
        
        if ph_row_idx is None:
            continue
        
        ph_row = df.iloc[ph_row_idx]
        
        # Calculate intervals and fiducial errors based on what's available in THIS beat
        # Only compare what's present in both manual and PyHEARTS annotations
        
        # PR interval: P onset to R peak (only if both are present in this beat)
        if manual_beat['p_onset'] is not None and manual_beat['r_peak'] is not None:
            # Check if PyHEARTS has P onset for this beat
            if pd.notna(ph_row.get('P_global_le_idx')):
                pr_manual = (manual_beat['r_peak'] - manual_beat['p_onset']) * 1000.0 / sampling_rate
                
                if pd.notna(ph_row.get('PR_interval_ms')):
                    pr_intervals_pyhearts.append(ph_row['PR_interval_ms'])
                    pr_intervals_manual.append(pr_manual)
                else:
                    # Calculate from fiducials
                    pr_ph = (ph_row['R_global_center_idx'] - ph_row['P_global_le_idx']) * 1000.0 / sampling_rate
                    pr_intervals_pyhearts.append(pr_ph)
                    pr_intervals_manual.append(pr_manual)
        
        # QT interval: R peak to T end (only if both are present in this beat)
        if manual_beat['r_peak'] is not None and manual_beat['t_end'] is not None:
            # Check if PyHEARTS has T end for this beat
            if pd.notna(ph_row.get('T_global_ri_idx')):
                qt_manual = (manual_beat['t_end'] - manual_beat['r_peak']) * 1000.0 / sampling_rate
                
                if pd.notna(ph_row.get('QT_interval_ms')):
                    qt_intervals_pyhearts.append(ph_row['QT_interval_ms'])
                    qt_intervals_manual.append(qt_manual)
                else:
                    # Calculate approximate QT from R peak to T end
                    qt_ph = (ph_row['T_global_ri_idx'] - ph_row['R_global_center_idx']) * 1000.0 / sampling_rate
                    qt_intervals_pyhearts.append(qt_ph)
                    qt_intervals_manual.append(qt_manual)
        
        # QRS interval from PyHEARTS only (manual annotations don't have QRS boundaries)
        if pd.notna(ph_row.get('QRS_interval_ms')):
            qrs_intervals_pyhearts.append(ph_row['QRS_interval_ms'])
        
        # Fiducial point errors - only compare what's available in THIS beat
        fiducial_types = [
            ('p_onset', 'P_global_le_idx'),
            ('p_peak', 'P_global_center_idx'),
            ('r_peak', 'R_global_center_idx'),
            ('t_peak', 'T_global_center_idx'),
            ('t_end', 'T_global_ri_idx'),
        ]
        
        for man_key, ph_col in fiducial_types:
            # Only compare if manual annotation has this fiducial AND PyHEARTS has it too
            if manual_beat[man_key] is not None and pd.notna(ph_row.get(ph_col)):
                error_samples = int(ph_row[ph_col]) - manual_beat[man_key]
                error_ms = error_samples * 1000.0 / sampling_rate
                fiducial_errors[man_key].append(error_ms)
                
                # Store matched pairs for peak points (p_peak, r_peak, t_peak) for Bland-Altman plots
                if man_key in ['p_peak', 'r_peak', 't_peak']:
                    ph_ms = int(ph_row[ph_col]) * 1000.0 / sampling_rate
                    man_ms = manual_beat[man_key] * 1000.0 / sampling_rate
                    matched_peak_pairs[man_key].append((ph_ms, man_ms))
    
    # Ensure matched intervals
    min_len = min(len(pr_intervals_manual), len(pr_intervals_pyhearts))
    pr_intervals_manual = pr_intervals_manual[:min_len]
    pr_intervals_pyhearts = pr_intervals_pyhearts[:min_len]
    
    min_len = min(len(qt_intervals_manual), len(qt_intervals_pyhearts))
    qt_intervals_manual = qt_intervals_manual[:min_len]
    qt_intervals_pyhearts = qt_intervals_pyhearts[:min_len]
    
    return {
        'subject': subject,
        'sampling_rate': sampling_rate,
        'n_matched_beats': len(matched_beats),
        'pr_manual': pr_intervals_manual,
        'pr_pyhearts': pr_intervals_pyhearts,
        'qt_manual': qt_intervals_manual,
        'qt_pyhearts': qt_intervals_pyhearts,
        'qrs_pyhearts': qrs_intervals_pyhearts,
        'fiducial_errors': dict(fiducial_errors),
        'matched_peak_pairs': dict(matched_peak_pairs),  # Store for Bland-Altman plots
    }

def create_bland_altman_plot(
    pyhearts_values: np.ndarray,
    manual_values: np.ndarray,
    interval_name: str,
    output_file: str,
    clinical_bound: float,
    is_peak: bool = False
):
    """Create Bland-Altman plot for interval or peak comparison."""
    if len(pyhearts_values) == 0 or len(manual_values) == 0:
        return None
    
    # Calculate mean and difference
    mean_values = (pyhearts_values + manual_values) / 2.0
    diff_values = pyhearts_values - manual_values
    
    # Calculate statistics
    mean_diff = np.mean(diff_values)
    std_diff = np.std(diff_values)
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff
    
    # Calculate percentage within clinical bounds
    within_bounds = np.sum(np.abs(diff_values) <= clinical_bound) / len(diff_values) * 100
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(mean_values, diff_values, alpha=0.6, s=50)
    
    # Mean difference line
    ax.axhline(mean_diff, color='red', linestyle='--', linewidth=2, label=f'Mean difference: {mean_diff:.2f} ms')
    
    # Limits of agreement
    ax.axhline(upper_limit, color='gray', linestyle='--', linewidth=1, label=f'Upper LoA: {upper_limit:.2f} ms')
    ax.axhline(lower_limit, color='gray', linestyle='--', linewidth=1, label=f'Lower LoA: {lower_limit:.2f} ms')
    
    # Clinical bounds
    ax.axhline(clinical_bound, color='orange', linestyle=':', linewidth=2, label=f'Clinical bound: ±{clinical_bound} ms')
    ax.axhline(-clinical_bound, color='orange', linestyle=':', linewidth=2)
    
    # Formatting
    if is_peak:
        xlabel = f'Mean of PyHEARTS and Manual {interval_name} (ms)'
        ylabel = f'Difference (PyHEARTS - Manual) (ms)'
        title = f'Bland-Altman Plot: {interval_name} Peak\n'
    else:
        xlabel = f'Mean of PyHEARTS and Manual {interval_name} (ms)'
        ylabel = f'Difference (PyHEARTS - Manual) (ms)'
        title = f'Bland-Altman Plot: {interval_name} Interval\n'
    
    title += f'Within clinical bounds (±{clinical_bound} ms): {within_bounds:.1f}%'
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'upper_limit': upper_limit,
        'lower_limit': lower_limit,
        'within_bounds_pct': within_bounds,
        'n': len(diff_values)
    }

def main():
    print("="*80)
    print("Gold Standard Comparison: PyHEARTS vs Manual Annotations")
    print("="*80)
    print()
    
    # Find subjects with both PyHEARTS results and manual annotations
    pyhearts_files = list(Path(PYHEARTS_RESULTS_DIR).glob("*_pyhearts.csv"))
    pyhearts_subjects = set([f.stem.replace('_pyhearts', '') for f in pyhearts_files])
    
    # Find manual annotation files
    manual_files = list(Path(MANUAL_ANNOTATIONS_DIR).glob("*.q1c"))
    manual_subjects = set([f.stem.replace('.q1c', '') for f in manual_files])
    
    common_subjects = sorted(list(pyhearts_subjects & manual_subjects))
    
    print(f"Found {len(common_subjects)} subjects with both PyHEARTS results and manual annotations")
    print("Analyzing subjects...")
    print()
    
    results = []
    for i, subject in enumerate(common_subjects, 1):
        if i % 10 == 0:
            print(f"  Processed {i}/{len(common_subjects)} subjects...")
        result = analyze_subject(subject)
        if result:
            results.append(result)
    
    if len(results) == 0:
        print("No results!")
        return
    
    print(f"\nProcessed {len(results)} subjects")
    print("Generating analysis...")
    print()
    
    # 1. Mean Absolute Deviation of Fiducial Points
    print("="*80)
    print("1. MEAN ABSOLUTE DEVIATION OF FIDUCIAL POINTS")
    print("="*80)
    print()
    
    fiducial_types = ['p_onset', 'p_peak', 'r_peak', 't_peak', 't_end']
    
    # Collect all fiducial errors
    all_fiducial_errors = defaultdict(list)
    for r in results:
        for fid_type, errors in r.get('fiducial_errors', {}).items():
            all_fiducial_errors[fid_type].extend([abs(e) for e in errors])
    
    mad_summary = []
    for fid_type in fiducial_types:
        errors = all_fiducial_errors.get(fid_type, [])
        if len(errors) > 0:
            # Remove outliers
            error_series = pd.Series(errors)
            error_clean = remove_outliers_iqr(error_series)
            
            mean_mad = error_clean.mean()
            median_mad = error_clean.median()
            std_mad = error_clean.std()
            n_outliers = len(errors) - len(error_clean)
            
            mad_summary.append({
                'Fiducial': fid_type,
                'Mean MAD (ms)': mean_mad,
                'Median MAD (ms)': median_mad,
                'Std MAD (ms)': std_mad,
                'Total Matched': len(errors),
                'N Subjects': len([r for r in results if fid_type in r.get('fiducial_errors', {})]),
                'Outliers Removed': n_outliers
            })
            
            print(f"{fid_type:15s}: Mean MAD = {mean_mad:6.2f} ms (median: {median_mad:6.2f} ms, std: {std_mad:6.2f} ms)")
            print(f"                Total matched: {len(errors):,}, Subjects: {len([r for r in results if fid_type in r.get('fiducial_errors', {})])} (removed {n_outliers} outliers)")
            print()
    
    # Save MAD summary
    if len(mad_summary) > 0:
        df_mad = pd.DataFrame(mad_summary)
        df_mad.to_csv(os.path.join(OUTPUT_DIR, 'fiducial_mad_summary.csv'), index=False)
    
    # 2. Bland-Altman Plots for Peak Points
    print("="*80)
    print("2. BLAND-ALTMAN PLOTS FOR PEAK POINTS")
    print("="*80)
    print()
    
    peak_types = ['p_peak', 'r_peak', 't_peak']  # Q and S peaks not reliably available in manual annotations
    peak_bland_altman_stats = []
    
    for peak_type in peak_types:
        # Collect matched peak pairs across all subjects
        ph_peak_ms = []
        man_peak_ms = []
        
        for r in results:
            if 'matched_peak_pairs' in r and peak_type in r['matched_peak_pairs']:
                pairs = r['matched_peak_pairs'][peak_type]
                for ph_ms, man_ms in pairs:
                    ph_peak_ms.append(ph_ms)
                    man_peak_ms.append(man_ms)
        
        if len(ph_peak_ms) > 0 and len(man_peak_ms) > 0:
            ph_array = np.array(ph_peak_ms)
            man_array = np.array(man_peak_ms)
            
            # Remove outliers
            diff_values = ph_array - man_array
            diff_series = pd.Series(diff_values)
            diff_clean = remove_outliers_iqr(diff_series)
            
            if len(diff_clean) > 0:
                outlier_mask = diff_series.isin(diff_clean)
                ph_clean = ph_array[outlier_mask.values]
                man_clean = man_array[outlier_mask.values]
                
                # Create Bland-Altman plot
                output_file = os.path.join(OUTPUT_DIR, f'bland_altman_{peak_type}.png')
                clinical_bound = CLINICAL_BOUNDS.get('fiducial', 20.0)
                stats_dict = create_bland_altman_plot(
                    ph_clean, man_clean, peak_type, output_file,
                    clinical_bound, is_peak=False  # Already in ms
                )
                if stats_dict:
                    stats_dict['peak'] = peak_type
                    stats_dict['outliers_removed'] = len(ph_peak_ms) - len(ph_clean)
                    peak_bland_altman_stats.append(stats_dict)
                    
                    print(f"{peak_type} Peak:")
                    print(f"  N comparisons: {stats_dict['n']} (removed {stats_dict['outliers_removed']} outliers)")
                    print(f"  Mean difference: {stats_dict['mean_diff']:.2f} ms")
                    print(f"  Limits of agreement: [{stats_dict['lower_limit']:.2f}, {stats_dict['upper_limit']:.2f}] ms")
                    print(f"  Within clinical bounds (±{clinical_bound} ms): {stats_dict['within_bounds_pct']:.1f}%")
                    print()
    
    # Save peak Bland-Altman statistics
    if len(peak_bland_altman_stats) > 0:
        df_peak_ba = pd.DataFrame(peak_bland_altman_stats)
        df_peak_ba.to_csv(os.path.join(OUTPUT_DIR, 'bland_altman_peak_statistics.csv'), index=False)
    
    # 3. Bland-Altman Plots for Intervals
    print("="*80)
    print("2. BLAND-ALTMAN PLOTS FOR INTERVALS")
    print("="*80)
    print()
    
    # Collect interval values
    all_pr_manual = []
    all_pr_pyhearts = []
    all_qt_manual = []
    all_qt_pyhearts = []
    all_qrs_pyhearts = []
    
    for r in results:
        all_pr_manual.extend(r.get('pr_manual', []))
        all_pr_pyhearts.extend(r.get('pr_pyhearts', []))
        all_qt_manual.extend(r.get('qt_manual', []))
        all_qt_pyhearts.extend(r.get('qt_pyhearts', []))
        all_qrs_pyhearts.extend(r.get('qrs_pyhearts', []))
    
    bland_altman_stats = []
    
    # PR Interval
    if len(all_pr_manual) > 0 and len(all_pr_pyhearts) > 0 and len(all_pr_manual) == len(all_pr_pyhearts):
        pr_manual_array = np.array(all_pr_manual)
        pr_pyhearts_array = np.array(all_pr_pyhearts)
        
        # Remove outliers
        diff_pr = pr_pyhearts_array - pr_manual_array
        diff_series = pd.Series(diff_pr)
        diff_clean = remove_outliers_iqr(diff_series)
        
        if len(diff_clean) > 0:
            outlier_mask = diff_series.isin(diff_clean)
            pr_manual_clean = pr_manual_array[outlier_mask.values]
            pr_pyhearts_clean = pr_pyhearts_array[outlier_mask.values]
            
            output_file = os.path.join(OUTPUT_DIR, 'bland_altman_PR.png')
            stats_dict = create_bland_altman_plot(
                pr_pyhearts_clean, pr_manual_clean, 'PR', output_file,
                CLINICAL_BOUNDS['PR']
            )
            if stats_dict:
                stats_dict['interval'] = 'PR'
                stats_dict['outliers_removed'] = len(all_pr_manual) - len(pr_manual_clean)
                bland_altman_stats.append(stats_dict)
                
                print(f"PR Interval:")
                print(f"  N comparisons: {stats_dict['n']} (removed {stats_dict['outliers_removed']} outliers)")
                print(f"  Mean difference: {stats_dict['mean_diff']:.2f} ms")
                print(f"  Limits of agreement: [{stats_dict['lower_limit']:.2f}, {stats_dict['upper_limit']:.2f}] ms")
                print(f"  Within clinical bounds (±{CLINICAL_BOUNDS['PR']} ms): {stats_dict['within_bounds_pct']:.1f}%")
                print()
    
    # QT Interval (approximate: R peak to T end, not true QT)
    if len(all_qt_manual) > 0 and len(all_qt_pyhearts) > 0 and len(all_qt_manual) == len(all_qt_pyhearts):
        qt_manual_array = np.array(all_qt_manual)
        qt_pyhearts_array = np.array(all_qt_pyhearts)
        
        # Remove outliers
        diff_qt = qt_pyhearts_array - qt_manual_array
        diff_series = pd.Series(diff_qt)
        diff_clean = remove_outliers_iqr(diff_series)
        
        if len(diff_clean) > 0:
            outlier_mask = diff_series.isin(diff_clean)
            qt_manual_clean = qt_manual_array[outlier_mask.values]
            qt_pyhearts_clean = qt_pyhearts_array[outlier_mask.values]
            
            output_file = os.path.join(OUTPUT_DIR, 'bland_altman_QT.png')
            stats_dict = create_bland_altman_plot(
                qt_pyhearts_clean, qt_manual_clean, 'QT (approx)', output_file,
                CLINICAL_BOUNDS['QT']
            )
            if stats_dict:
                stats_dict['interval'] = 'QT'
                stats_dict['outliers_removed'] = len(all_qt_manual) - len(qt_manual_clean)
                bland_altman_stats.append(stats_dict)
                
                print(f"QT Interval (approximate: R peak to T end):")
                print(f"  N comparisons: {stats_dict['n']} (removed {stats_dict['outliers_removed']} outliers)")
                print(f"  Mean difference: {stats_dict['mean_diff']:.2f} ms")
                print(f"  Limits of agreement: [{stats_dict['lower_limit']:.2f}, {stats_dict['upper_limit']:.2f}] ms")
                print(f"  Within clinical bounds (±{CLINICAL_BOUNDS['QT']} ms): {stats_dict['within_bounds_pct']:.1f}%")
                print()
    
    # QRS Interval - Note: Manual annotations don't have QRS boundaries
    print("QRS Interval:")
    print("  NOTE: Manual annotations do not include QRS onset/offset boundaries,")
    print("        so QRS duration cannot be calculated from manual annotations.")
    print(f"        PyHEARTS detected {len(all_qrs_pyhearts)} QRS intervals.")
    print()
    
    # Save Bland-Altman statistics
    if len(bland_altman_stats) > 0:
        df_ba = pd.DataFrame(bland_altman_stats)
        df_ba.to_csv(os.path.join(OUTPUT_DIR, 'bland_altman_statistics.csv'), index=False)
    
    # 4. Clinical Error Bounds Summary
    print("="*80)
    print("4. CLINICAL ERROR BOUNDS ASSESSMENT")
    print("="*80)
    print()
    
    clinical_summary = []
    
    # Fiducial points
    for fid_type in fiducial_types:
        errors = all_fiducial_errors.get(fid_type, [])
        if len(errors) > 0:
            error_series = pd.Series(errors)
            error_clean = remove_outliers_iqr(error_series)
            
            if len(error_clean) > 0:
                within_bounds = np.sum(error_clean.values <= CLINICAL_BOUNDS['fiducial']) / len(error_clean) * 100
                clinical_summary.append({
                    'Type': 'Fiducial',
                    'Name': fid_type,
                    'Bound (ms)': CLINICAL_BOUNDS['fiducial'],
                    'Mean Error (ms)': error_clean.mean(),
                    'Within Bound (%)': within_bounds,
                    'N': len(error_clean),
                    'Outliers Removed': len(errors) - len(error_clean)
                })
                print(f"{fid_type:15s}: Mean error = {error_clean.mean():6.2f} ms, "
                      f"Within ±{CLINICAL_BOUNDS['fiducial']} ms: {within_bounds:.1f}% "
                      f"(removed {len(errors) - len(error_clean)} outliers)")
    
    print()
    
    # Intervals
    for ba_stat in bland_altman_stats:
        interval_name = ba_stat['interval']
        clinical_summary.append({
            'Type': 'Interval',
            'Name': interval_name,
            'Bound (ms)': CLINICAL_BOUNDS[interval_name],
            'Mean Error (ms)': abs(ba_stat['mean_diff']),
            'Within Bound (%)': ba_stat['within_bounds_pct'],
            'N': ba_stat['n'],
            'Outliers Removed': ba_stat.get('outliers_removed', 0)
        })
        print(f"{interval_name:15s}: Mean difference = {abs(ba_stat['mean_diff']):6.2f} ms, "
              f"Within ±{CLINICAL_BOUNDS[interval_name]} ms: {ba_stat['within_bounds_pct']:.1f}%")
    
    # Save clinical summary
    if len(clinical_summary) > 0:
        df_clinical = pd.DataFrame(clinical_summary)
        df_clinical.to_csv(os.path.join(OUTPUT_DIR, 'clinical_error_bounds_summary.csv'), index=False)
    
    print()
    print("="*80)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)

if __name__ == '__main__':
    main()

