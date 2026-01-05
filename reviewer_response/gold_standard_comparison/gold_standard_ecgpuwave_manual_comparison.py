#!/usr/bin/env python3
"""
Gold Standard Comparison Analysis: ECGPUWave vs Manual Annotations (QT Database)

This script compares ECGPUWave detection results to manual annotations from the QT Database,
providing:
1. Mean absolute deviation of fiducial points vs. manual annotations
2. Bland-Altman plots for peak points (P, R, T)
3. Comparison statistics to assess ECGPUWave's agreement with manual annotations

Uses QT Database manual annotations (.q1c files from expert annotator 1) as reference.
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
ECGPUWAVE_RESULTS_DIR = "/Users/morganfitzgerald/Documents/qtdb/qtdb_full_ecgpuwave_results"
FULL_SIGNAL_ECGPUWAVE_DIR = "/Users/morganfitzgerald/Documents/ecgpuwave/results/qtdb_full_ecgpuwave_results/ecgpuwave_annotations_pw"
MANUAL_ANNOTATIONS_DIR = "/Users/morganfitzgerald/Documents/pyhearts/data/qtdb/1.0.0"
OUTPUT_DIR = "/Users/morganfitzgerald/Documents/pyhearts/reviewer_response/gold_standard_comparison/ecgpuwave_vs_manual"

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

def load_ecgpuwave_annotations(pu_file: str) -> Dict[str, np.ndarray]:
    """
    Load ECGPUWave annotations with all fiducial points.
    
    Supports both .pu and .pw file formats.
    
    ECGPUWave symbols:
    - 'p', 'P': P wave peak
    - 'N': R peak
    - 't', 'T': T wave peak
    - '(', ')': Wave boundaries (onset/offset)
    - '[': P wave onset
    - ']': P wave offset
    - '{': QRS onset
    - '}': QRS offset
    
    Returns dictionary with arrays of sample indices for each annotation type.
    """
    annotation_dir = os.path.dirname(pu_file)
    record_name = os.path.basename(pu_file)
    
    # Determine extension (.pu or .pw)
    if record_name.endswith('.pu'):
        extension = 'pu'
        record_name = record_name.replace('.pu', '')
    elif record_name.endswith('.pw'):
        extension = 'pw'
        record_name = record_name.replace('.pw', '')
    else:
        return {}
    
    try:
        old_dir = os.getcwd()
        os.chdir(annotation_dir)
        ann = wfdb.rdann(record_name, extension)
        os.chdir(old_dir)
    except Exception as e:
        try:
            os.chdir(old_dir)
        except:
            pass
        return {}
    
    annotations = {
        'P_peak': [],
        'P_onset': [],
        'P_offset': [],
        'R_peak': [],
        'T_peak': [],
        'T_onset': [],
        'T_offset': [],
        'QRS_onset': [],
        'QRS_offset': [],
    }
    
    for i in range(len(ann.symbol)):
        symbol = ann.symbol[i]
        sample = ann.sample[i]
        
        if symbol in ['p', 'P']:
            annotations['P_peak'].append(sample)
        elif symbol == '[':
            annotations['P_onset'].append(sample)
        elif symbol == ']':
            annotations['P_offset'].append(sample)
        elif symbol == 'N':
            annotations['R_peak'].append(sample)
        elif symbol in ['t', 'T']:
            annotations['T_peak'].append(sample)
        elif symbol == '(':
            annotations['T_onset'].append(sample)
        elif symbol == ')':
            annotations['T_offset'].append(sample)
        elif symbol == '{':
            annotations['QRS_onset'].append(sample)
        elif symbol == '}':
            annotations['QRS_offset'].append(sample)
    
    # Convert to numpy arrays
    for key in annotations:
        annotations[key] = np.array(annotations[key], dtype=int)
    
    return annotations

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
        for j in range(r_idx - 1, max(-1, r_idx - 20), -1):
            if ann.symbol[j] == 'p':
                beat['p_peak'] = ann.sample[j]
                # Look for P onset '(' before P peak
                for k in range(max(0, j - 5), j):
                    if ann.symbol[k] == '(':
                        if k + 1 == j:
                            beat['p_onset'] = ann.sample[k]
                        else:
                            has_p_between = any(ann.symbol[m] == 'p' for m in range(k + 1, j))
                            if not has_p_between:
                                beat['p_onset'] = ann.sample[k]
                        if beat['p_onset'] is not None:
                            break
                break
        
        # Look forward from R peak for T wave annotations
        for j in range(r_idx + 1, min(len(ann.symbol), r_idx + 20)):
            if ann.symbol[j] in ['t', 'T']:
                beat['t_peak'] = ann.sample[j]
                # Look for T end ')' after T peak
                for k in range(j + 1, min(len(ann.symbol), j + 10)):
                    if ann.symbol[k] == ')':
                        beat['t_end'] = ann.sample[k]
                        break
                break
        
        beats.append(beat)
    
    return beats

def get_sampling_rate(meta_file: str) -> float:
    """Get sampling rate from meta.json file."""
    try:
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            return float(meta.get('sampling_rate', 250.0))
    except:
        return 250.0

def match_beats(manual_beats: List[Dict], ecgpuwave_r_peaks: np.ndarray, sampling_rate: float, max_distance_ms: float = 100.0) -> List[Tuple[int, int]]:
    """
    Match manual beats to ECGPUWave R peaks.
    
    Note: Manual annotations may be in a different time window than ECGPUWave.
    If there's no direct overlap, we check if there's a consistent offset pattern.
    
    Returns list of (manual_beat_idx, ecgpuwave_r_idx) tuples.
    """
    if len(manual_beats) == 0 or len(ecgpuwave_r_peaks) == 0:
        return []
    
    max_distance_samples = int(round(max_distance_ms * sampling_rate / 1000.0))
    
    # Extract manual R peaks
    manual_r_peaks = np.array([b['r_peak'] for b in manual_beats])
    
    # Check if there's any temporal overlap
    man_min, man_max = manual_r_peaks.min(), manual_r_peaks.max()
    ecg_min, ecg_max = ecgpuwave_r_peaks.min(), ecgpuwave_r_peaks.max()
    
    # If no overlap, manual annotations are likely in a different time segment
    # ECGPUWave typically processes the first ~600s, while manual annotations
    # often cover later segments. In this case, we cannot directly compare them.
    if man_max < ecg_min or man_min > ecg_max:
        # No temporal overlap - these are from different signal segments
        # This is expected for many QT Database subjects
        return []
    
    matched_pairs = []
    manual_matched = set()
    ecgpuwave_matched = set()
    
    # Create potential matches (only if there's temporal overlap)
    potential_matches = []
    for man_idx, man_r in enumerate(manual_r_peaks):
        for ecg_idx, ecg_r in enumerate(ecgpuwave_r_peaks):
            distance_samples = abs(ecg_r - man_r)
            if distance_samples <= max_distance_samples:
                potential_matches.append((distance_samples, man_idx, ecg_idx))
    
    # Sort by distance and match greedily
    potential_matches.sort(key=lambda x: x[0])
    
    for distance_samples, man_idx, ecg_idx in potential_matches:
        if man_idx not in manual_matched and ecg_idx not in ecgpuwave_matched:
            matched_pairs.append((man_idx, ecg_idx))
            manual_matched.add(man_idx)
            ecgpuwave_matched.add(ecg_idx)
    
    return matched_pairs

def create_bland_altman_plot(
    ecgpuwave_values: np.ndarray,
    manual_values: np.ndarray,
    peak_name: str,
    output_file: str,
    clinical_bound: float,
    is_peak: bool = False
):
    """Create Bland-Altman plot for peak comparison."""
    if len(ecgpuwave_values) == 0 or len(manual_values) == 0:
        return None
    
    # Calculate mean and difference
    mean_values = (ecgpuwave_values + manual_values) / 2.0
    diff_values = ecgpuwave_values - manual_values
    
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
    xlabel = f'Mean of ECGPUWave and Manual {peak_name} (ms)'
    ylabel = f'Difference (ECGPUWave - Manual) (ms)'
    title = f'Bland-Altman Plot: {peak_name} Peak\n'
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

def analyze_subject(subject: str, use_full_signal: bool = False) -> Optional[Dict]:
    """Analyze a single subject."""
    # Load ECGPUWave annotations
    # Try full-signal directory first if requested, otherwise use standard directory
    if use_full_signal:
        # Try .pw file first (full-signal format)
        pu_file = os.path.join(FULL_SIGNAL_ECGPUWAVE_DIR, f'{subject}.pw')
        if not os.path.exists(pu_file):
            return None
    else:
        pu_file = os.path.join(ECGPUWAVE_RESULTS_DIR, f'{subject}.pu')
        if not os.path.exists(pu_file):
            return None
    
    ecgpuwave_fiducials = load_ecgpuwave_annotations(pu_file)
    ecgpuwave_r_peaks = ecgpuwave_fiducials.get('R_peak', np.array([], dtype=int))
    
    if len(ecgpuwave_r_peaks) == 0:
        return None
    
    # Get sampling rate (try to find from PyHEARTS meta if available, or default)
    # For now, use default 250 Hz (QT Database standard)
    sampling_rate = 250.0
    
    # Load manual annotations
    manual_beats = load_manual_annotations(subject, 'q1c', sampling_rate)
    if len(manual_beats) == 0:
        return None
    
    # Match beats
    # Note: Manual annotations may be in a different time window than ECGPUWave
    # Try matching with a reasonable distance threshold
    matched_beats = match_beats(manual_beats, ecgpuwave_r_peaks, sampling_rate, max_distance_ms=100.0)
    if len(matched_beats) == 0:
        # If no matches, it's likely that manual annotations are in a different time segment
        # This is common - manual annotations often cover only a portion of the signal
        return None
    
    # Calculate fiducial differences and matched pairs for Bland-Altman plots
    fiducial_errors = defaultdict(list)
    matched_peak_pairs = defaultdict(list)  # Store matched pairs for Bland-Altman plots
    
    for man_idx, ecg_r_idx in matched_beats:
        manual_beat = manual_beats[man_idx]
        ecg_r_peak = ecgpuwave_r_peaks[ecg_r_idx]
        
        # Find ECGPUWave P and T peaks in the same beat
        # For P peak: look before R peak (find closest P peak before this R peak)
        ecg_p_peaks = ecgpuwave_fiducials.get('P_peak', np.array([], dtype=int))
        ecg_p_peak = None
        if len(ecg_p_peaks) > 0:
            p_before = ecg_p_peaks[ecg_p_peaks < ecg_r_peak]
            if len(p_before) > 0:
                # Find the closest P peak before R (within reasonable distance)
                p_distances = ecg_r_peak - p_before
                valid_p = p_before[p_distances < int(round(500 * sampling_rate / 1000.0))]  # Within 500ms
                if len(valid_p) > 0:
                    # Take the closest one
                    ecg_p_peak = valid_p[np.argmin(ecg_r_peak - valid_p)]
        
        # For T peak: look after R peak (find closest T peak after this R peak)
        ecg_t_peaks = ecgpuwave_fiducials.get('T_peak', np.array([], dtype=int))
        ecg_t_peak = None
        if len(ecg_t_peaks) > 0:
            t_after = ecg_t_peaks[ecg_t_peaks > ecg_r_peak]
            if len(t_after) > 0:
                # Find the closest T peak after R (within reasonable distance)
                t_distances = t_after - ecg_r_peak
                valid_t = t_after[t_distances < int(round(800 * sampling_rate / 1000.0))]  # Within 800ms
                if len(valid_t) > 0:
                    # Take the closest one
                    ecg_t_peak = valid_t[np.argmin(valid_t - ecg_r_peak)]
        
        # Compare fiducial points
        fiducial_types = [
            ('p_peak', ecg_p_peak),
            ('r_peak', ecg_r_peak),
            ('t_peak', ecg_t_peak),
        ]
        
        for man_key, ecg_sample in fiducial_types:
            if ecg_sample is not None and manual_beat[man_key] is not None:
                error_samples = ecg_sample - manual_beat[man_key]
                error_ms = error_samples * 1000.0 / sampling_rate
                fiducial_errors[man_key].append(error_ms)
                
                # Store matched pairs for peak points for Bland-Altman plots
                if man_key in ['p_peak', 'r_peak', 't_peak']:
                    ecg_ms = ecg_sample * 1000.0 / sampling_rate
                    man_ms = manual_beat[man_key] * 1000.0 / sampling_rate
                    matched_peak_pairs[man_key].append((ecg_ms, man_ms))
    
    return {
        'subject': subject,
        'sampling_rate': sampling_rate,
        'n_matched_beats': len(matched_beats),
        'fiducial_errors': dict(fiducial_errors),
        'matched_peak_pairs': dict(matched_peak_pairs),  # Store for Bland-Altman plots
    }

def main():
    print("="*80)
    print("Gold Standard Comparison: ECGPUWave vs Manual Annotations")
    print("="*80)
    print()
    
    # Check for full-signal ECGPUWave results first
    full_signal_files = list(Path(FULL_SIGNAL_ECGPUWAVE_DIR).glob("*.pw"))
    full_signal_subjects = set([f.stem.replace('.pw', '') for f in full_signal_files])
    
    # Find manual annotation files
    manual_files = list(Path(MANUAL_ANNOTATIONS_DIR).glob("*.q1c"))
    manual_subjects = set([f.stem.replace('.q1c', '') for f in manual_files])
    
    # Use full-signal subjects if available, otherwise use standard ECGPUWave results
    if len(full_signal_subjects) > 0:
        common_subjects = sorted(list(full_signal_subjects & manual_subjects))
        use_full_signal = True
        print(f"Using FULL-SIGNAL ECGPUWave results from: {FULL_SIGNAL_ECGPUWAVE_DIR}")
    else:
        # Fall back to standard ECGPUWave results
        ecgpuwave_files = list(Path(ECGPUWAVE_RESULTS_DIR).glob("*.pu"))
        ecgpuwave_subjects = set([f.stem.replace('.pu', '') for f in ecgpuwave_files])
        common_subjects = sorted(list(ecgpuwave_subjects & manual_subjects))
        use_full_signal = False
        print(f"Using standard ECGPUWave results from: {ECGPUWAVE_RESULTS_DIR}")
    
    print(f"Found {len(common_subjects)} subjects with both ECGPUWave results and manual annotations")
    print("Analyzing subjects...")
    print()
    
    results = []
    for i, subject in enumerate(common_subjects, 1):
        if i % 10 == 0:
            print(f"  Processed {i}/{len(common_subjects)} subjects...")
        result = analyze_subject(subject, use_full_signal=use_full_signal)
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
    
    fiducial_types = ['p_peak', 'r_peak', 't_peak']
    
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
    
    peak_types = ['p_peak', 'r_peak', 't_peak']
    peak_bland_altman_stats = []
    
    for peak_type in peak_types:
        # Collect matched peak pairs across all subjects
        ecg_peak_ms = []
        man_peak_ms = []
        
        for r in results:
            if 'matched_peak_pairs' in r and peak_type in r['matched_peak_pairs']:
                pairs = r['matched_peak_pairs'][peak_type]
                for ecg_ms, man_ms in pairs:
                    ecg_peak_ms.append(ecg_ms)
                    man_peak_ms.append(man_ms)
        
        if len(ecg_peak_ms) > 0 and len(man_peak_ms) > 0:
            ecg_array = np.array(ecg_peak_ms)
            man_array = np.array(man_peak_ms)
            
            # Remove outliers
            diff_values = ecg_array - man_array
            diff_series = pd.Series(diff_values)
            diff_clean = remove_outliers_iqr(diff_series)
            
            if len(diff_clean) > 0:
                outlier_mask = diff_series.isin(diff_clean)
                ecg_clean = ecg_array[outlier_mask.values]
                man_clean = man_array[outlier_mask.values]
                
                # Create Bland-Altman plot
                output_file = os.path.join(OUTPUT_DIR, f'bland_altman_{peak_type}.png')
                clinical_bound = CLINICAL_BOUNDS.get('fiducial', 20.0)
                stats_dict = create_bland_altman_plot(
                    ecg_clean, man_clean, peak_type, output_file,
                    clinical_bound, is_peak=False  # Already in ms
                )
                if stats_dict:
                    stats_dict['peak'] = peak_type
                    stats_dict['outliers_removed'] = len(ecg_peak_ms) - len(ecg_clean)
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
    
    print("="*80)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)

if __name__ == '__main__':
    main()

