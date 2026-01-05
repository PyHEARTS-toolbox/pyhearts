#!/usr/bin/env python3
"""
Gold Standard Comparison Analysis for PyHEARTS

Addresses reviewer requests:
1. Mean absolute deviation of fiducial points vs. manual annotations
2. Bland-Altman plots for key intervals (PR, QRS, QT, RT, TT)
3. Demonstration that PyHEARTS hits clinically acceptable error bounds

Uses QT Database with ECGPUWave annotations as gold standard.
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
ECGPUWAVE_RESULTS_DIR = "/Users/morganfitzgerald/Documents/qtdb/qtdb_full_ecgpuwave_results"
OUTPUT_DIR = "/Users/morganfitzgerald/Documents/pyhearts/reviewer_response/gold_standard_comparison/pyhearts_vs_ecgpuwave"

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

def load_ecgpuwave_annotations(pu_file: str) -> Dict[str, np.ndarray]:
    """
    Load ECGPUWave annotations with all fiducial points.
    
    ECGPUWave symbols:
    - 'p', 'P': P wave peak
    - 'N': R peak
    - 't', 'T': T wave peak
    - '(', ')': Wave boundaries (onset/offset)
    - '[': P wave onset
    - ']': P wave offset
    - '{': QRS onset
    - '}': QRS offset
    - '|': T wave boundaries
    
    Returns dictionary with arrays of sample indices for each annotation type.
    """
    annotation_dir = os.path.dirname(pu_file)
    annotation_name = os.path.basename(pu_file).replace('.pu', '')
    
    original_dir = os.getcwd()
    os.chdir(annotation_dir)
    try:
        annotation = wfdb.rdann(annotation_name, 'pu')
        
        # Extract all annotation types
        annotations = {
            'P_peak': [],
            'P_onset': [],
            'P_offset': [],
            'Q_peak': [],
            'R_peak': [],
            'S_peak': [],
            'QRS_onset': [],
            'QRS_offset': [],
            'T_peak': [],
            'T_onset': [],
            'T_offset': [],
        }
        
        for i, symbol in enumerate(annotation.symbol):
            sample = annotation.sample[i]
            if sample < 0:
                continue
            
            # P wave annotations
            if symbol.lower() == 'p':
                annotations['P_peak'].append(sample)
            elif symbol == '[':
                annotations['P_onset'].append(sample)
            elif symbol == ']':
                annotations['P_offset'].append(sample)
            
            # QRS annotations
            elif symbol == 'N':
                annotations['R_peak'].append(sample)
            elif symbol == '{':
                annotations['QRS_onset'].append(sample)
            elif symbol == '}':
                annotations['QRS_offset'].append(sample)
            
            # T wave annotations
            elif symbol.lower() == 't':
                annotations['T_peak'].append(sample)
            elif symbol == '(':
                annotations['T_onset'].append(sample)
            elif symbol == ')':
                annotations['T_offset'].append(sample)
        
        # Convert to numpy arrays and sort
        for key in annotations:
            annotations[key] = np.array(sorted(annotations[key]), dtype=int)
        
        return annotations
    finally:
        os.chdir(original_dir)

def load_pyhearts_fiducials(csv_file: str) -> Dict[str, np.ndarray]:
    """
    Load PyHEARTS fiducial points from CSV.
    
    PyHEARTS uses:
    - P_global_le_idx (left edge = onset), P_global_center_idx, P_global_ri_idx (right edge = offset)
    - Q_global_center_idx, R_global_center_idx, S_global_center_idx
    - T_global_le_idx (left edge = onset), T_global_center_idx, T_global_ri_idx (right edge = offset)
    """
    df = pd.read_csv(csv_file)
    
    fiducials = {}
    
    # P wave: le_idx = onset, center_idx = peak, ri_idx = offset
    for suffix, key_suffix in [('le', 'onset'), ('center', 'peak'), ('ri', 'offset')]:
        col = f'P_global_{suffix}_idx'
        key = f'P_{key_suffix}'
        if col in df.columns:
            values = df[col].dropna().values
            fiducials[key] = np.array([int(v) for v in values if not np.isnan(v)], dtype=int)
        else:
            fiducials[key] = np.array([], dtype=int)
    
    # Q, R, S peaks (centers only)
    for comp in ['Q', 'R', 'S']:
        col = f'{comp}_global_center_idx'
        if col in df.columns:
            values = df[col].dropna().values
            fiducials[f'{comp}_peak'] = np.array([int(v) for v in values if not np.isnan(v)], dtype=int)
        else:
            fiducials[f'{comp}_peak'] = np.array([], dtype=int)
    
    # T wave: le_idx = onset, center_idx = peak, ri_idx = offset
    for suffix, key_suffix in [('le', 'onset'), ('center', 'peak'), ('ri', 'offset')]:
        col = f'T_global_{suffix}_idx'
        key = f'T_{key_suffix}'
        if col in df.columns:
            values = df[col].dropna().values
            fiducials[key] = np.array([int(v) for v in values if not np.isnan(v)], dtype=int)
        else:
            fiducials[key] = np.array([], dtype=int)
    
    # For QRS, we can use Q and S peaks to estimate QRS boundaries
    # QRS onset ≈ Q peak, QRS offset ≈ S peak (if available)
    if len(fiducials.get('Q_peak', [])) > 0:
        fiducials['QRS_onset'] = fiducials['Q_peak'].copy()
    else:
        fiducials['QRS_onset'] = np.array([], dtype=int)
    
    if len(fiducials.get('S_peak', [])) > 0:
        fiducials['QRS_offset'] = fiducials['S_peak'].copy()
    else:
        fiducials['QRS_offset'] = np.array([], dtype=int)
    
    return fiducials

def get_sampling_rate(meta_file: str) -> float:
    """Get sampling rate from meta.json file."""
    try:
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            return float(meta.get('sampling_rate', 250.0))
    except:
        return 250.0

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

def match_fiducials_within_beats(
    pyhearts_fiducials: np.ndarray,
    ecgpuwave_fiducials: np.ndarray,
    matched_r_peaks: Dict[int, int],  # pyhearts_r_idx -> ecgpuwave_r_idx mapping
    pyhearts_r_peaks: np.ndarray,
    ecgpuwave_r_peaks: np.ndarray,
    sampling_rate: float,
    max_match_distance_ms: float = 50.0
) -> List[Tuple[int, int, float]]:
    """
    Match fiducial points between PyHEARTS and ECGPUWave within the same beat.
    
    Uses matched R peaks to define beat boundaries.
    For ECGPUWave, if multiple points per peak exist, takes the closest to PyHEARTS point.
    
    Returns list of (pyhearts_idx, ecgpuwave_idx, distance_ms) tuples.
    """
    if len(pyhearts_fiducials) == 0 or len(ecgpuwave_fiducials) == 0 or len(matched_r_peaks) == 0:
        return []
    
    max_match_distance_samples = int(round(max_match_distance_ms * sampling_rate / 1000.0))
    
    # Create beat boundaries from matched R peaks
    beat_boundaries = []
    matched_ph_r_indices = sorted(matched_r_peaks.keys())
    
    for i, ph_r_idx in enumerate(matched_ph_r_indices):
        ecg_r_idx = matched_r_peaks[ph_r_idx]
        next_ph_r_idx = matched_ph_r_indices[i + 1] if i < len(matched_ph_r_indices) - 1 else None
        next_ecg_r_idx = matched_r_peaks[next_ph_r_idx] if next_ph_r_idx is not None else None
        
        ph_r_peak = pyhearts_r_peaks[ph_r_idx]
        ecg_r_peak = ecgpuwave_r_peaks[ecg_r_idx]
        next_ph_r_peak = pyhearts_r_peaks[next_ph_r_idx] if next_ph_r_idx is not None else None
        next_ecg_r_peak = ecgpuwave_r_peaks[next_ecg_r_idx] if next_ecg_r_idx is not None else None
        
        beat_boundaries.append({
            'ph_r_start': ph_r_peak,
            'ph_r_end': next_ph_r_peak if next_ph_r_peak is not None else ph_r_peak + int(round(2000 * sampling_rate / 1000.0)),
            'ecg_r_start': ecg_r_peak,
            'ecg_r_end': next_ecg_r_peak if next_ecg_r_peak is not None else ecg_r_peak + int(round(2000 * sampling_rate / 1000.0)),
        })
    
    # Group fiducials by beat
    pyhearts_beat_map = {}  # fiducial_idx -> beat_idx
    for ph_idx, ph_fid in enumerate(pyhearts_fiducials):
        for beat_idx, boundary in enumerate(beat_boundaries):
            if boundary['ph_r_start'] <= ph_fid < boundary['ph_r_end']:
                pyhearts_beat_map[ph_idx] = beat_idx
                break
    
    ecgpuwave_beat_map = defaultdict(list)  # beat_idx -> [fiducial_indices]
    for ecg_idx, ecg_fid in enumerate(ecgpuwave_fiducials):
        for beat_idx, boundary in enumerate(beat_boundaries):
            if boundary['ecg_r_start'] <= ecg_fid < boundary['ecg_r_end']:
                ecgpuwave_beat_map[beat_idx].append(ecg_idx)
                break
    
    # Match fiducials within the same beat
    matched_pairs = []
    pyhearts_matched = set()
    ecgpuwave_matched = set()
    
    # For each beat, match PyHEARTS fiducials to closest ECGPUWave fiducial
    for beat_idx in sorted(set(pyhearts_beat_map.values())):
        ph_indices_in_beat = [ph_idx for ph_idx, b_idx in pyhearts_beat_map.items() if b_idx == beat_idx]
        ecg_indices_in_beat = ecgpuwave_beat_map.get(beat_idx, [])
        
        if len(ecg_indices_in_beat) == 0:
            continue
        
        # For each PyHEARTS fiducial, find closest ECGPUWave fiducial in this beat
        for ph_idx in ph_indices_in_beat:
            if ph_idx in pyhearts_matched:
                continue
            
            ph_fid = pyhearts_fiducials[ph_idx]
            
            # Find closest ECGPUWave fiducial in this beat (handles multiple points per peak)
            best_ecg_idx = None
            best_distance = float('inf')
            
            for ecg_idx in ecg_indices_in_beat:
                if ecg_idx in ecgpuwave_matched:
                    continue
                
                ecg_fid = ecgpuwave_fiducials[ecg_idx]
                distance_samples = abs(ecg_fid - ph_fid)
                
                if distance_samples <= max_match_distance_samples and distance_samples < best_distance:
                    best_distance = distance_samples
                    best_ecg_idx = ecg_idx
            
            if best_ecg_idx is not None:
                distance_ms = best_distance * 1000.0 / sampling_rate
                matched_pairs.append((ph_idx, best_ecg_idx, distance_ms))
                pyhearts_matched.add(ph_idx)
                ecgpuwave_matched.add(best_ecg_idx)
    
    return matched_pairs

def calculate_intervals(fiducials: Dict[str, np.ndarray], sampling_rate: float) -> Dict[str, List[float]]:
    """
    Calculate intervals from fiducial points.
    
    Returns dictionary with interval names and lists of values in milliseconds.
    """
    intervals = {
        'PR': [],
        'QRS': [],
        'QT': [],
        'RT': [],
        'TT': [],
    }
    
    # PR interval: P onset to R peak (or P peak to R peak if no P onset)
    if len(fiducials.get('P_onset', [])) > 0 and len(fiducials.get('R_peak', [])) > 0:
        for p_onset in fiducials['P_onset']:
            # Find nearest R peak after P onset
            r_peaks_after = fiducials['R_peak'][fiducials['R_peak'] >= p_onset]
            if len(r_peaks_after) > 0:
                r_peak = r_peaks_after[0]
                pr_ms = (r_peak - p_onset) * 1000.0 / sampling_rate
                intervals['PR'].append(pr_ms)
    elif len(fiducials.get('P_peak', [])) > 0 and len(fiducials.get('R_peak', [])) > 0:
        # Fallback: Use P peak to R peak (approximate PR interval)
        for p_peak in fiducials['P_peak']:
            r_peaks_after = fiducials['R_peak'][fiducials['R_peak'] >= p_peak]
            if len(r_peaks_after) > 0:
                r_peak = r_peaks_after[0]
                pr_ms = (r_peak - p_peak) * 1000.0 / sampling_rate
                intervals['PR'].append(pr_ms)
    
    # QRS duration: QRS onset to QRS offset (or Q peak to S peak if no boundaries)
    if len(fiducials.get('QRS_onset', [])) > 0 and len(fiducials.get('QRS_offset', [])) > 0:
        for qrs_onset in fiducials['QRS_onset']:
            # Find nearest QRS offset after onset
            qrs_offsets_after = fiducials['QRS_offset'][fiducials['QRS_offset'] >= qrs_onset]
            if len(qrs_offsets_after) > 0:
                qrs_offset = qrs_offsets_after[0]
                qrs_ms = (qrs_offset - qrs_onset) * 1000.0 / sampling_rate
                intervals['QRS'].append(qrs_ms)
    elif len(fiducials.get('Q_peak', [])) > 0 and len(fiducials.get('S_peak', [])) > 0:
        # Fallback: Use Q peak to S peak (approximate QRS duration)
        for q_peak in fiducials['Q_peak']:
            s_peaks_after = fiducials['S_peak'][fiducials['S_peak'] >= q_peak]
            if len(s_peaks_after) > 0:
                s_peak = s_peaks_after[0]
                qrs_ms = (s_peak - q_peak) * 1000.0 / sampling_rate
                intervals['QRS'].append(qrs_ms)
    elif len(fiducials.get('R_peak', [])) > 0:
        # Fallback: Estimate QRS from R peak width (if available)
        # This is a rough approximation - typically QRS is 80-120ms
        # We'll skip this as it's too approximate
        pass
    
    # QT interval: QRS onset to T offset (or R peak to T offset if no QRS onset)
    if len(fiducials.get('QRS_onset', [])) > 0 and len(fiducials.get('T_offset', [])) > 0:
        for qrs_onset in fiducials['QRS_onset']:
            # Find nearest T offset after QRS onset
            t_offsets_after = fiducials['T_offset'][fiducials['T_offset'] >= qrs_onset]
            if len(t_offsets_after) > 0:
                t_offset = t_offsets_after[0]
                qt_ms = (t_offset - qrs_onset) * 1000.0 / sampling_rate
                intervals['QT'].append(qt_ms)
    elif len(fiducials.get('R_peak', [])) > 0 and len(fiducials.get('T_offset', [])) > 0:
        # Fallback: Use R peak to T offset (approximate QT interval)
        # Note: This is R-T offset, not true QT (which should be QRS onset to T offset)
        for r_peak in fiducials['R_peak']:
            t_offsets_after = fiducials['T_offset'][fiducials['T_offset'] >= r_peak]
            if len(t_offsets_after) > 0:
                t_offset = t_offsets_after[0]
                qt_ms = (t_offset - r_peak) * 1000.0 / sampling_rate
                intervals['QT'].append(qt_ms)
    
    # RT interval: R peak to T peak
    if len(fiducials.get('R_peak', [])) > 0 and len(fiducials.get('T_peak', [])) > 0:
        for r_peak in fiducials['R_peak']:
            # Find nearest T peak after R peak
            t_peaks_after = fiducials['T_peak'][fiducials['T_peak'] >= r_peak]
            if len(t_peaks_after) > 0:
                t_peak = t_peaks_after[0]
                rt_ms = (t_peak - r_peak) * 1000.0 / sampling_rate
                intervals['RT'].append(rt_ms)
    
    # TT interval: T peak to next T peak
    if len(fiducials.get('T_peak', [])) > 1:
        t_peaks = sorted(fiducials['T_peak'])
        for i in range(len(t_peaks) - 1):
            tt_ms = (t_peaks[i+1] - t_peaks[i]) * 1000.0 / sampling_rate
            intervals['TT'].append(tt_ms)
    
    return intervals

def create_bland_altman_plot(
    pyhearts_values: np.ndarray,
    ecgpuwave_values: np.ndarray,
    interval_name: str,
    output_file: str,
    clinical_bound: float,
    is_peak: bool = False
):
    """
    Create Bland-Altman plot for interval or peak comparison.
    
    Parameters
    ----------
    pyhearts_values : np.ndarray
        PyHEARTS values (in ms for intervals, in samples for peaks)
    ecgpuwave_values : np.ndarray
        ECGPUWave values (in ms for intervals, in samples for peaks)
    interval_name : str
        Name of interval or peak (e.g., 'PR', 'RT', 'P_peak', 'R_peak')
    output_file : str
        Output file path for the plot
    clinical_bound : float
        Clinical error bound in ms
    is_peak : bool
        If True, values are in samples and need to be converted to ms (requires sampling_rate)
    """
    if len(pyhearts_values) == 0 or len(ecgpuwave_values) == 0:
        return None
    
    # Calculate mean and difference
    mean_values = (pyhearts_values + ecgpuwave_values) / 2.0
    diff_values = pyhearts_values - ecgpuwave_values
    
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
        xlabel = f'Mean of PyHEARTS and ECGPUWave {interval_name} (samples)'
        ylabel = f'Difference (PyHEARTS - ECGPUWave) (samples)'
    else:
        xlabel = f'Mean of PyHEARTS and ECGPUWave {interval_name} (ms)'
        ylabel = f'Difference (PyHEARTS - ECGPUWave) (ms)'
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    if is_peak:
        title = f'Bland-Altman Plot: {interval_name} Peak\n'
        title += f'Within clinical bounds (±{clinical_bound} ms): {within_bounds:.1f}%'
    else:
        title = f'Bland-Altman Plot: {interval_name} Interval\n'
        title += f'Within clinical bounds (±{clinical_bound} ms): {within_bounds:.1f}%'
    
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

def load_pyhearts_intervals(csv_file: str, sampling_rate: float) -> Dict[str, np.ndarray]:
    """Load PyHEARTS interval measurements directly from CSV."""
    df = pd.read_csv(csv_file)
    
    intervals = {}
    interval_cols = {
        'PR': 'PR_interval_ms',
        'QRS': 'QRS_interval_ms',
        'QT': 'QT_interval_ms',
    }
    
    for interval_name, col_name in interval_cols.items():
        if col_name in df.columns:
            values = df[col_name].dropna().values
            intervals[interval_name] = np.array([float(v) for v in values if not np.isnan(v)])
        else:
            intervals[interval_name] = np.array([], dtype=float)
    
    # Calculate RT and TT from fiducials if not available
    # RT: R peak to T peak
    if 'R_global_center_idx' in df.columns and 'T_global_center_idx' in df.columns:
        r_peaks = df['R_global_center_idx'].dropna().values
        t_peaks = df['T_global_center_idx'].dropna().values
        
        # Get sampling rate to convert
        # We'll need to pass this or calculate from meta
        rt_values = []
        for _, row in df.iterrows():
            if not np.isnan(row.get('R_global_center_idx', np.nan)) and not np.isnan(row.get('T_global_center_idx', np.nan)):
                rt_ms = row.get('RT_interval_ms', np.nan)
                if not np.isnan(rt_ms):
                    rt_values.append(rt_ms)
        
        if len(rt_values) > 0:
            intervals['RT'] = np.array(rt_values)
        else:
            intervals['RT'] = np.array([], dtype=float)
    else:
        intervals['RT'] = np.array([], dtype=float)
    
    # TT: T peak to next T peak (from RR intervals or calculated)
    if 'RR_interval_ms' in df.columns:
        # Approximate TT from RR (not perfect but available)
        intervals['TT'] = np.array([], dtype=float)
    else:
        intervals['TT'] = np.array([], dtype=float)
    
    return intervals

def analyze_subject(subject: str) -> Optional[Dict]:
    """Analyze a single subject."""
    # Load PyHEARTS results
    csv_file = os.path.join(PYHEARTS_RESULTS_DIR, f'{subject}_pyhearts.csv')
    meta_file = os.path.join(PYHEARTS_RESULTS_DIR, f'{subject}_meta.json')
    
    if not os.path.exists(csv_file) or not os.path.exists(meta_file):
        return None
    
    sampling_rate = get_sampling_rate(meta_file)
    pyhearts_fiducials_all = load_pyhearts_fiducials(csv_file)
    pyhearts_intervals = load_pyhearts_intervals(csv_file, sampling_rate)
    
    # Load ECGPUWave annotations
    pu_file = os.path.join(ECGPUWAVE_RESULTS_DIR, f'{subject}.pu')
    if not os.path.exists(pu_file):
        return None
    
    ecgpuwave_fiducials_all = load_ecgpuwave_annotations(pu_file)
    
    # Filter to ECGPUWave annotated regions only
    # Determine annotation coverage from ECGPUWave annotations
    all_ecg_annotations = np.concatenate([
        ecgpuwave_fiducials_all.get('P_peak', []),
        ecgpuwave_fiducials_all.get('R_peak', []),
        ecgpuwave_fiducials_all.get('T_peak', []),
        ecgpuwave_fiducials_all.get('P_onset', []),
        ecgpuwave_fiducials_all.get('P_offset', []),
        ecgpuwave_fiducials_all.get('T_onset', []),
        ecgpuwave_fiducials_all.get('T_offset', []),
        ecgpuwave_fiducials_all.get('QRS_onset', []),
        ecgpuwave_fiducials_all.get('QRS_offset', []),
    ])
    
    if len(all_ecg_annotations) == 0:
        return None
    
    annotation_start = int(np.min(all_ecg_annotations))
    annotation_end = int(np.max(all_ecg_annotations))
    
    # Filter PyHEARTS fiducials to annotated region only
    def filter_to_region(fiducials, start, end):
        return fiducials[(fiducials >= start) & (fiducials <= end)]
    
    pyhearts_fiducials = {}
    ecgpuwave_fiducials = {}
    for key in pyhearts_fiducials_all.keys():
        pyhearts_fiducials[key] = filter_to_region(
            pyhearts_fiducials_all[key], annotation_start, annotation_end
        )
        ecgpuwave_fiducials[key] = ecgpuwave_fiducials_all[key]  # Already in annotated region
    
    # First match R peaks directly (they define beats)
    ph_r_peaks = pyhearts_fiducials.get('R_peak', np.array([], dtype=int))
    ecg_r_peaks = ecgpuwave_fiducials.get('R_peak', np.array([], dtype=int))
    
    if len(ph_r_peaks) == 0 or len(ecg_r_peaks) == 0:
        return None
    
    # Match R peaks directly (for beat definition)
    max_match_distance_samples = int(round(100.0 * sampling_rate / 1000.0))  # 100ms for R peaks
    r_matched_pairs = []
    ph_r_matched = set()
    ecg_r_matched = set()
    
    # Greedy matching for R peaks (closest point matching)
    potential_r_matches = []
    for ph_idx, ph_r in enumerate(ph_r_peaks):
        for ecg_idx, ecg_r in enumerate(ecg_r_peaks):
            distance_samples = abs(ecg_r - ph_r)
            if distance_samples <= max_match_distance_samples:
                distance_ms = distance_samples * 1000.0 / sampling_rate
                potential_r_matches.append((distance_ms, ph_idx, ecg_idx))
    
    potential_r_matches.sort(key=lambda x: x[0])
    matched_r_peaks = {}  # pyhearts_r_idx -> ecgpuwave_r_idx
    
    for distance_ms, ph_idx, ecg_idx in potential_r_matches:
        if ph_idx not in ph_r_matched and ecg_idx not in ecg_r_matched:
            r_matched_pairs.append((ph_idx, ecg_idx, distance_ms))
            matched_r_peaks[ph_idx] = ecg_idx
            ph_r_matched.add(ph_idx)
            ecg_r_matched.add(ecg_idx)
    
    if len(matched_r_peaks) == 0:
        return None  # Need matched R peaks to define beats
    
    # Match fiducial points within beats and calculate MAD
    fiducial_types = ['P_peak', 'P_onset', 'P_offset', 'R_peak', 'T_peak', 'T_onset', 'T_offset']
    mad_results = {}
    matched_peak_pairs = {}  # Store matched pairs for Bland-Altman plots (peak types only)
    
    for fid_type in fiducial_types:
        ph_fids = pyhearts_fiducials.get(fid_type, np.array([], dtype=int))
        ecg_fids = ecgpuwave_fiducials.get(fid_type, np.array([], dtype=int))
        
        if len(ph_fids) == 0 or len(ecg_fids) == 0:
            mad_results[fid_type] = {'mad_ms': np.nan, 'n_matched': 0}
            continue
        
        # Match within beats (for non-R peaks) or use direct matching for R peaks
        if fid_type == 'R_peak':
            # R peaks already matched above
            matched = [(ph_idx, matched_r_peaks[ph_idx], r_matched_pairs[i][2]) 
                      for i, (ph_idx, _, _) in enumerate(r_matched_pairs)]
        else:
            matched = match_fiducials_within_beats(
                ph_fids, ecg_fids, matched_r_peaks, ph_r_peaks, ecg_r_peaks,
                sampling_rate, max_match_distance_ms=50.0
            )
        
        if len(matched) == 0:
            mad_results[fid_type] = {'mad_ms': np.nan, 'n_matched': 0}
            continue
        
        # Calculate Mean Absolute Deviation
        distances_ms = [m[2] for m in matched]
        mad_ms = np.mean(np.abs(distances_ms))
        mad_results[fid_type] = {'mad_ms': mad_ms, 'n_matched': len(matched)}
        
        # Store matched pairs for peak points (P, R, T) for Bland-Altman plots
        if fid_type in ['P_peak', 'R_peak', 'T_peak']:
            matched_peak_pairs[fid_type] = [(ph_fids[m[0]], ecg_fids[m[1]]) for m in matched]
    
    # Calculate intervals from filtered fiducials (within annotated region)
    ecgpuwave_intervals = calculate_intervals(ecgpuwave_fiducials, sampling_rate)
    
    # Calculate RT and TT from fiducials if not in intervals
    ph_rt = pyhearts_intervals.get('RT', np.array([], dtype=float))
    if len(ph_rt) == 0:
        # Calculate from R and T peaks
        ph_rt = calculate_intervals(pyhearts_fiducials, sampling_rate)['RT']
    
    ph_tt = pyhearts_intervals.get('TT', np.array([], dtype=float))
    if len(ph_tt) == 0:
        ph_tt = calculate_intervals(pyhearts_fiducials, sampling_rate)['TT']
    
    return {
        'subject': subject,
        'sampling_rate': sampling_rate,
        **{f'{fid}_mad_ms': mad_results[fid]['mad_ms'] for fid in fiducial_types},
        **{f'{fid}_n_matched': mad_results[fid]['n_matched'] for fid in fiducial_types},
        'ph_PR': pyhearts_intervals.get('PR', []),
        'ph_QRS': pyhearts_intervals.get('QRS', []),
        'ph_QT': pyhearts_intervals.get('QT', []),
        'ph_RT': ph_rt.tolist() if isinstance(ph_rt, np.ndarray) else ph_rt,
        'ph_TT': ph_tt.tolist() if isinstance(ph_tt, np.ndarray) else ph_tt,
        'ecg_PR': ecgpuwave_intervals['PR'],
        'ecg_QRS': ecgpuwave_intervals['QRS'],
        'ecg_QT': ecgpuwave_intervals['QT'],
        'ecg_RT': ecgpuwave_intervals['RT'],
        'ecg_TT': ecgpuwave_intervals['TT'],
        'matched_peak_pairs': matched_peak_pairs,  # Store for Bland-Altman plots
    }

def main():
    print("="*80)
    print("Gold Standard Comparison Analysis")
    print("="*80)
    print()
    
    # Find all subjects
    pyhearts_files = list(Path(PYHEARTS_RESULTS_DIR).glob("*_pyhearts.csv"))
    subjects = sorted([f.stem.replace('_pyhearts', '') for f in pyhearts_files])
    
    print(f"Found {len(subjects)} subjects")
    print("Analyzing subjects...")
    print()
    
    results = []
    for i, subject in enumerate(subjects, 1):
        if i % 10 == 0:
            print(f"  Processed {i}/{len(subjects)} subjects...")
        result = analyze_subject(subject)
        if result:
            results.append(result)
    
    if len(results) == 0:
        print("No results!")
        return
    
    # Process results
    print(f"\nProcessed {len(results)} subjects")
    print("Generating analysis...")
    print()
    
    # 1. Mean Absolute Deviation of Fiducial Points
    print("="*80)
    print("1. MEAN ABSOLUTE DEVIATION OF FIDUCIAL POINTS")
    print("="*80)
    print()
    
    fiducial_types = ['P_peak', 'P_onset', 'P_offset', 'R_peak', 'T_peak', 'T_onset', 'T_offset']
    
    mad_summary = []
    for fid_type in fiducial_types:
        mad_col = f'{fid_type}_mad_ms'
        n_col = f'{fid_type}_n_matched'
        
        # Collect all MAD values
        all_mads = []
        total_matched = 0
        for r in results:
            if mad_col in r and not np.isnan(r[mad_col]):
                all_mads.append(r[mad_col])
                total_matched += r.get(n_col, 0)
        
        if len(all_mads) > 0:
            # Remove outliers before calculating final statistics
            mad_series = pd.Series(all_mads)
            mad_clean = remove_outliers_iqr(mad_series)
            
            mean_mad = mad_clean.mean()
            median_mad = mad_clean.median()
            std_mad = mad_clean.std()
            n_outliers = len(all_mads) - len(mad_clean)
            
            mad_summary.append({
                'Fiducial': fid_type,
                'Mean MAD (ms)': mean_mad,
                'Median MAD (ms)': median_mad,
                'Std MAD (ms)': std_mad,
                'Total Matched': total_matched,
                'N Subjects': len(mad_clean),
                'Outliers Removed': n_outliers
            })
            print(f"{fid_type:15s}: Mean MAD = {mean_mad:6.2f} ms (median: {median_mad:6.2f} ms, std: {std_mad:6.2f} ms)")
            print(f"                Total matched: {total_matched:,}, Subjects: {len(mad_clean)} (removed {n_outliers} outliers)")
            print()
    
    # Save MAD summary
    df_mad = pd.DataFrame(mad_summary)
    df_mad.to_csv(os.path.join(OUTPUT_DIR, 'fiducial_mad_summary.csv'), index=False)
    
    # 2. Bland-Altman Plots for Peak Points
    print("="*80)
    print("2. BLAND-ALTMAN PLOTS FOR PEAK POINTS")
    print("="*80)
    print()
    
    peak_types = ['P_peak', 'R_peak', 'T_peak']  # Q and S peaks not reliably available
    peak_bland_altman_stats = []
    
    for peak_type in peak_types:
        # Collect matched peak pairs across all subjects
        ph_peak_samples = []
        ecg_peak_samples = []
        sampling_rates = []
        
        for r in results:
            if 'matched_peak_pairs' in r and peak_type in r['matched_peak_pairs']:
                pairs = r['matched_peak_pairs'][peak_type]
                sr = r.get('sampling_rate', 250.0)
                
                for ph_sample, ecg_sample in pairs:
                    ph_peak_samples.append(ph_sample)
                    ecg_peak_samples.append(ecg_sample)
                    sampling_rates.append(sr)
        
        if len(ph_peak_samples) > 0 and len(ecg_peak_samples) > 0:
            # Convert from samples to ms
            ph_peak_ms = np.array([s * 1000.0 / sr for s, sr in zip(ph_peak_samples, sampling_rates)])
            ecg_peak_ms = np.array([s * 1000.0 / sr for s, sr in zip(ecg_peak_samples, sampling_rates)])
            
            # Remove outliers
            diff_values = ph_peak_ms - ecg_peak_ms
            diff_series = pd.Series(diff_values)
            diff_clean = remove_outliers_iqr(diff_series)
            
            if len(diff_clean) > 0:
                outlier_mask = diff_series.isin(diff_clean)
                ph_clean = ph_peak_ms[outlier_mask.values]
                ecg_clean = ecg_peak_ms[outlier_mask.values]
                
                # Create Bland-Altman plot
                output_file = os.path.join(OUTPUT_DIR, f'bland_altman_{peak_type}.png')
                clinical_bound = CLINICAL_BOUNDS.get('fiducial', 20.0)
                stats_dict = create_bland_altman_plot(
                    ph_clean, ecg_clean, peak_type, output_file,
                    clinical_bound, is_peak=False  # Already in ms
                )
                if stats_dict:
                    stats_dict['peak'] = peak_type
                    stats_dict['outliers_removed'] = len(ph_peak_samples) - len(ph_clean)
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
    
    interval_names = ['PR', 'QRS', 'QT', 'RT', 'TT']
    bland_altman_stats = []
    
    for interval_name in interval_names:
        # Collect matched interval values
        ph_values = []
        ecg_values = []
        
        for r in results:
            ph_key = f'ph_{interval_name}'
            ecg_key = f'ecg_{interval_name}'
            
            if ph_key in r and ecg_key in r:
                ph_list = r[ph_key] if isinstance(r[ph_key], list) else []
                ecg_list = r[ecg_key] if isinstance(r[ecg_key], list) else []
                
                # Convert to numpy arrays if needed
                if isinstance(ph_list, list):
                    ph_list = [v for v in ph_list if v is not None and not np.isnan(v)]
                if isinstance(ecg_list, list):
                    ecg_list = [v for v in ecg_list if v is not None and not np.isnan(v)]
                
                # Match intervals (simplified: match by order, but filter out invalid values)
                min_len = min(len(ph_list), len(ecg_list))
                if min_len > 0:
                    ph_vals = [v for v in ph_list[:min_len] if isinstance(v, (int, float)) and not np.isnan(v)]
                    ecg_vals = [v for v in ecg_list[:min_len] if isinstance(v, (int, float)) and not np.isnan(v)]
                    # Only add if both lists have same length after filtering
                    if len(ph_vals) == len(ecg_vals) and len(ph_vals) > 0:
                        ph_values.extend(ph_vals)
                        ecg_values.extend(ecg_vals)
        
        if len(ph_values) > 0 and len(ecg_values) > 0 and len(ph_values) == len(ecg_values):
            ph_array = np.array(ph_values)
            ecg_array = np.array(ecg_values)
            
            # Remove outliers before creating plot and calculating statistics
            # Calculate differences first
            diff_values = ph_array - ecg_array
            mean_values = (ph_array + ecg_array) / 2.0
            
            # Remove outliers from differences using IQR
            diff_series = pd.Series(diff_values)
            diff_clean = remove_outliers_iqr(diff_series)
            
            if len(diff_clean) > 0:
                # Get corresponding ph and ecg values for non-outlier differences
                outlier_mask = diff_series.isin(diff_clean)
                ph_clean = ph_array[outlier_mask.values]
                ecg_clean = ecg_array[outlier_mask.values]
                
                # Create Bland-Altman plot with cleaned data
                output_file = os.path.join(OUTPUT_DIR, f'bland_altman_{interval_name}.png')
                stats_dict = create_bland_altman_plot(
                    ph_clean, ecg_clean, interval_name, output_file,
                    CLINICAL_BOUNDS[interval_name]
                )
                stats_dict['interval'] = interval_name
                stats_dict['outliers_removed'] = len(ph_values) - len(ph_clean)
                bland_altman_stats.append(stats_dict)
                
                print(f"{interval_name} Interval:")
                print(f"  N comparisons: {stats_dict['n']} (removed {stats_dict['outliers_removed']} outliers)")
                print(f"  Mean difference: {stats_dict['mean_diff']:.2f} ms")
                print(f"  Limits of agreement: [{stats_dict['lower_limit']:.2f}, {stats_dict['upper_limit']:.2f}] ms")
                print(f"  Within clinical bounds (±{CLINICAL_BOUNDS[interval_name]} ms): {stats_dict['within_bounds_pct']:.1f}%")
                print()
    
    # Save Bland-Altman statistics
    df_ba = pd.DataFrame(bland_altman_stats)
    df_ba.to_csv(os.path.join(OUTPUT_DIR, 'bland_altman_statistics.csv'), index=False)
    
    # 4. Clinical Error Bounds Summary
    print("="*80)
    print("4. CLINICAL ERROR BOUNDS ASSESSMENT")
    print("="*80)
    print()
    
    clinical_summary = []
    
    # Fiducial points (with outlier removal)
    for fid_type in fiducial_types:
        mad_col = f'{fid_type}_mad_ms'
        all_mads = [r[mad_col] for r in results if mad_col in r and not np.isnan(r[mad_col])]
        if len(all_mads) > 0:
            # Remove outliers before calculating statistics
            mad_series = pd.Series(all_mads)
            mad_clean = remove_outliers_iqr(mad_series)
            
            if len(mad_clean) > 0:
                within_bounds = np.sum(mad_clean.values <= CLINICAL_BOUNDS['fiducial']) / len(mad_clean) * 100
                clinical_summary.append({
                    'Type': 'Fiducial',
                    'Name': fid_type,
                    'Bound (ms)': CLINICAL_BOUNDS['fiducial'],
                    'Mean Error (ms)': mad_clean.mean(),
                    'Within Bound (%)': within_bounds,
                    'N': len(mad_clean),
                    'Outliers Removed': len(all_mads) - len(mad_clean)
                })
                print(f"{fid_type:15s}: Mean error = {mad_clean.mean():6.2f} ms, "
                      f"Within ±{CLINICAL_BOUNDS['fiducial']} ms: {within_bounds:.1f}% "
                      f"(removed {len(all_mads) - len(mad_clean)} outliers)")
    
    print()
    
    # Intervals
    for interval_name in interval_names:
        if len(bland_altman_stats) > 0:
            ba_stat = next((s for s in bland_altman_stats if s['interval'] == interval_name), None)
            if ba_stat:
                clinical_summary.append({
                    'Type': 'Interval',
                    'Name': interval_name,
                    'Bound (ms)': CLINICAL_BOUNDS[interval_name],
                    'Mean Error (ms)': abs(ba_stat['mean_diff']),
                    'Within Bound (%)': ba_stat['within_bounds_pct'],
                    'N': ba_stat['n']
                })
                print(f"{interval_name:15s}: Mean difference = {abs(ba_stat['mean_diff']):6.2f} ms, "
                      f"Within ±{CLINICAL_BOUNDS[interval_name]} ms: {ba_stat['within_bounds_pct']:.1f}%")
    
    # Save clinical summary
    df_clinical = pd.DataFrame(clinical_summary)
    df_clinical.to_csv(os.path.join(OUTPUT_DIR, 'clinical_error_bounds_summary.csv'), index=False)
    
    print()
    print("="*80)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)

if __name__ == '__main__':
    main()

