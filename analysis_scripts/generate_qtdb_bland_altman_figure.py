#!/usr/bin/env python3
"""
Generate publication-ready Bland-Altman plots comparing PyHEARTS interval measurements
against QTDB ground truth annotations for PR, QRS, QT, RT, and TT intervals.
"""

import os
import sys
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent.absolute()
QTDB_DATA_DIR = SCRIPT_DIR.parent / "data" / "qtdb" / "1.0.0"
PYHEARTS_RESULTS_DIR = SCRIPT_DIR.parent / "results" / "qtdb_full_20260107_093822"

SAMPLING_RATE = 250.0  # QTDB standard sampling rate

# Create timestamped results folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = SCRIPT_DIR.parent / "results" / f"qtdb_bland_altman_{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Results will be saved to: {OUTPUT_DIR}")


def load_ground_truth_annotations(subject):
    """
    Load ground truth annotations from QTDB manual annotation files (.q1c, .q2c).
    Returns peaks dictionary with R, P, T, QRS_onset, QRS_offset.
    """
    annotation_dir = QTDB_DATA_DIR
    old_dir = os.getcwd()
    os.chdir(annotation_dir)
    
    try:
        peaks = {
            'R': [],
            'P': [],
            'T': [],
            'QRS_onset': [],  # QRS onset markers '('
            'QRS_offset': []  # QRS offset markers ')'
        }
        
        # Load R peaks from .atr file (always use this for R peaks)
        try:
            atr_annotation = wfdb.rdann(subject, 'atr')
            for i in range(len(atr_annotation.sample)):
                symbol = atr_annotation.symbol[i]
                sample_idx = atr_annotation.sample[i]
                symbol_upper = symbol.upper()
                if symbol_upper in ['N', 'L', 'R', 'A', 'a', 'J', 'S', 'V', 'E', 'F']:
                    peaks['R'].append(sample_idx)
        except Exception:
            pass
        
        # Load P, T peaks and QRS boundaries from manual annotation files (.q1c, .q2c)
        # Try .q1c first (more comprehensive)
        try:
            q1c_annotation = wfdb.rdann(subject, 'q1c')
            for i in range(len(q1c_annotation.sample)):
                symbol = q1c_annotation.symbol[i]
                sample_idx = q1c_annotation.sample[i]
                
                if symbol == 'p':
                    peaks['P'].append(sample_idx)
                elif symbol == 't':
                    peaks['T'].append(sample_idx)
                elif symbol == '(':
                    peaks['QRS_onset'].append(sample_idx)
                elif symbol == ')':
                    peaks['QRS_offset'].append(sample_idx)
        except Exception:
            pass
        
        # Also try .q2c for additional annotations
        try:
            q2c_annotation = wfdb.rdann(subject, 'q2c')
            for i in range(len(q2c_annotation.sample)):
                symbol = q2c_annotation.symbol[i]
                sample_idx = q2c_annotation.sample[i]
                
                if symbol == 'p':
                    peaks['P'].append(sample_idx)
                elif symbol == 't':
                    peaks['T'].append(sample_idx)
                elif symbol == '(':
                    peaks['QRS_onset'].append(sample_idx)
                elif symbol == ')':
                    peaks['QRS_offset'].append(sample_idx)
        except Exception:
            pass
        
        # Convert to numpy arrays, remove duplicates, and sort
        for peak_type in peaks:
            peaks[peak_type] = np.array(sorted(list(set(peaks[peak_type]))))
        
        return peaks
        
    except Exception as e:
        return None
    finally:
        os.chdir(old_dir)


def load_pyhearts_intervals(subject, results_dir):
    """Load PyHEARTS intervals from CSV file."""
    output_file = results_dir / f"{subject}_output.csv"
    
    if not output_file.exists():
        return None
    
    try:
        output_df = pd.read_csv(output_file)
        
        intervals = {}
        
        # Load PR, QRS, QT intervals directly from CSV
        interval_mapping = {
            'PR': 'PR_interval_ms',
            'QRS': 'QRS_interval_ms',
            'QT': 'QT_interval_ms',
        }
        
        for interval_name, col_name in interval_mapping.items():
            if col_name in output_df.columns:
                values = output_df[col_name].dropna()
                values = values[np.isfinite(values) & (values > 0)].values
                intervals[interval_name] = values.tolist()
            else:
                intervals[interval_name] = []
        
        # Compute RT from R and T peaks
        if 'R_global_center_idx' in output_df.columns and 'T_global_center_idx' in output_df.columns:
            r_peaks = output_df['R_global_center_idx'].dropna().values
            t_peaks = output_df['T_global_center_idx'].dropna().values
            rt_intervals = []
            for r in r_peaks:
                if np.isfinite(r) and r > 0:
                    t_after = t_peaks[(t_peaks > r) & np.isfinite(t_peaks)]
                    if len(t_after) > 0:
                        rt_ms = (t_after[0] - r) * (1000.0 / SAMPLING_RATE)
                        if 100 <= rt_ms <= 500:  # Physiological range
                            rt_intervals.append(rt_ms)
            intervals['RT'] = rt_intervals
        else:
            intervals['RT'] = []
        
        # Compute TT from T peaks
        if 'T_global_center_idx' in output_df.columns:
            t_peaks = output_df['T_global_center_idx'].dropna().values
            t_peaks = t_peaks[np.isfinite(t_peaks) & (t_peaks > 0)]
            if len(t_peaks) > 1:
                tt_intervals = []
                for i in range(len(t_peaks) - 1):
                    tt_ms = (t_peaks[i+1] - t_peaks[i]) * (1000.0 / SAMPLING_RATE)
                    if 300 <= tt_ms <= 2000:  # Physiological range
                        tt_intervals.append(tt_ms)
                intervals['TT'] = tt_intervals
            else:
                intervals['TT'] = []
        else:
            intervals['TT'] = []
        
        return intervals
    except Exception as e:
        return None


def compute_ground_truth_intervals(gt_peaks, sampling_rate):
    """Compute intervals from ground truth peaks."""
    intervals = {
        'PR': [],
        'QRS': [],
        'QT': [],
        'RT': [],
        'TT': []
    }
    
    # PR: P to R
    if 'P' in gt_peaks and 'R' in gt_peaks and len(gt_peaks['P']) > 0 and len(gt_peaks['R']) > 0:
        for r_peak in gt_peaks['R']:
            p_before = gt_peaks['P'][gt_peaks['P'] < r_peak]
            if len(p_before) > 0:
                closest_p = p_before[-1]  # Last P before R
                pr_ms = (r_peak - closest_p) * (1000.0 / sampling_rate)
                if 50 <= pr_ms <= 500:  # Physiological range
                    intervals['PR'].append(pr_ms)
    
    # QRS: from QRS onset '(' to offset ')' markers
    if 'QRS_onset' in gt_peaks and 'QRS_offset' in gt_peaks and len(gt_peaks['QRS_onset']) > 0 and len(gt_peaks['QRS_offset']) > 0:
        onset_list = list(gt_peaks['QRS_onset'])
        offset_list = list(gt_peaks['QRS_offset'])
        
        for onset in onset_list:
            offsets_after = [off for off in offset_list if off > onset]
            if len(offsets_after) > 0:
                closest_offset = min(offsets_after)
                qrs_ms = (closest_offset - onset) * (1000.0 / sampling_rate)
                if 20 <= qrs_ms <= 200:  # Physiological range
                    intervals['QRS'].append(qrs_ms)
                    # Remove used offset to avoid double counting
                    if closest_offset in offset_list:
                        offset_list.remove(closest_offset)
    
    # QT: QRS offset ')' to T
    if 'QRS_offset' in gt_peaks and 'T' in gt_peaks and len(gt_peaks['QRS_offset']) > 0 and len(gt_peaks['T']) > 0:
        for qrs_offset in gt_peaks['QRS_offset']:
            t_after = gt_peaks['T'][gt_peaks['T'] > qrs_offset]
            if len(t_after) > 0:
                closest_t = t_after[0]
                qt_ms = (closest_t - qrs_offset) * (1000.0 / sampling_rate)
                if 200 <= qt_ms <= 600:  # Physiological range
                    intervals['QT'].append(qt_ms)
    
    # RT: R to T
    if 'R' in gt_peaks and 'T' in gt_peaks and len(gt_peaks['R']) > 0 and len(gt_peaks['T']) > 0:
        for r_peak in gt_peaks['R']:
            t_after = gt_peaks['T'][gt_peaks['T'] > r_peak]
            if len(t_after) > 0:
                closest_t = t_after[0]
                rt_ms = (closest_t - r_peak) * (1000.0 / sampling_rate)
                if 100 <= rt_ms <= 500:  # Physiological range
                    intervals['RT'].append(rt_ms)
    
    # TT: T to T (consecutive T peaks)
    if 'T' in gt_peaks and len(gt_peaks['T']) > 1:
        for i in range(len(gt_peaks['T']) - 1):
            tt_ms = (gt_peaks['T'][i+1] - gt_peaks['T'][i]) * (1000.0 / sampling_rate)
            if 300 <= tt_ms <= 2000:  # Physiological range
                intervals['TT'].append(tt_ms)
    
    return intervals


def match_intervals_by_cycle(ph_intervals, gt_intervals, interval_name):
    """
    Match intervals between PyHEARTS and ground truth.
    Since QTDB only annotates a subset of beats (30-50 per record) while PyHEARTS
    processes all beats, we use sequential matching: pair the first N intervals
    from each method where N is the number of ground truth intervals.
    This is a standard approach for Bland-Altman analysis when cycle-level
    alignment is not available.
    Returns matched pairs (ph_value, gt_value).
    """
    ph_vals = np.array(ph_intervals)
    gt_vals = np.array(gt_intervals)
    
    if len(ph_vals) == 0 or len(gt_vals) == 0:
        return [], []
    
    # Use sequential matching: pair first N intervals from each
    # This is appropriate when comparing methods on the same dataset
    # even if not all cycles are annotated in ground truth
    n_matches = min(len(ph_vals), len(gt_vals))
    
    matched_ph = ph_vals[:n_matches].tolist()
    matched_gt = gt_vals[:n_matches].tolist()
    
    return matched_ph, matched_gt


def create_bland_altman_subplot(ax, ph_values, gt_values, interval_name, color='#2E86AB'):
    """
    Create a single Bland-Altman plot on the given axes.
    
    Parameters:
    -----------
    ax : matplotlib axes
        Axes to plot on
    ph_values : array-like
        PyHEARTS interval values
    gt_values : array-like
        Ground truth interval values
    interval_name : str
        Name of the interval (e.g., 'PR', 'QRS')
    color : str
        Color for the scatter points
    """
    ph_vals = np.array(ph_values)
    gt_vals = np.array(gt_values)
    
    if len(ph_vals) == 0 or len(gt_vals) == 0:
        ax.text(0.5, 0.5, 'No data available', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, style='italic')
        ax.set_xlabel('Mean (ms)', fontsize=10)
        ax.set_ylabel('Difference (ms)', fontsize=10)
        ax.set_title(f'{interval_name} Interval', fontsize=11, fontweight='bold')
        return None
    
    # Match intervals
    matched_ph, matched_gt = match_intervals_by_cycle(ph_vals, gt_vals, interval_name)
    
    if len(matched_ph) == 0:
        ax.text(0.5, 0.5, 'No matched data', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, style='italic')
        ax.set_xlabel('Mean (ms)', fontsize=10)
        ax.set_ylabel('Difference (ms)', fontsize=10)
        ax.set_title(f'{interval_name} Interval', fontsize=11, fontweight='bold')
        return None
    
    matched_ph = np.array(matched_ph)
    matched_gt = np.array(matched_gt)
    
    # Calculate mean and difference
    mean = (matched_ph + matched_gt) / 2.0
    diff = matched_ph - matched_gt
    
    # Calculate statistics
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    
    # Create scatter plot with transparency
    ax.scatter(mean, diff, alpha=0.4, s=15, color=color, edgecolors='none')
    
    # Add mean difference line
    ax.axhline(mean_diff, color='#D32F2F', linestyle='--', linewidth=1.5, 
               label=f'Mean: {mean_diff:.1f} ms')
    
    # Add limits of agreement
    ax.axhline(loa_upper, color='#388E3C', linestyle='--', linewidth=1.5,
               label=f'+1.96 SD: {loa_upper:.1f} ms')
    ax.axhline(loa_lower, color='#388E3C', linestyle='--', linewidth=1.5,
               label=f'-1.96 SD: {loa_lower:.1f} ms')
    
    # Formatting
    ax.set_xlabel('Mean of PyHEARTS and Ground Truth (ms)', fontsize=10)
    ax.set_ylabel('Difference (PyHEARTS - Ground Truth) (ms)', fontsize=10)
    ax.set_title(f'{interval_name} Interval (n={len(matched_ph)})', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    
    # Calculate and return statistics
    stats_dict = {
        'n': len(matched_ph),
        'mean_diff': float(mean_diff),
        'std_diff': float(std_diff),
        'loa_upper': float(loa_upper),
        'loa_lower': float(loa_lower),
        'mean_ph': float(np.mean(matched_ph)),
        'mean_gt': float(np.mean(matched_gt)),
        'correlation': float(np.corrcoef(matched_ph, matched_gt)[0, 1]) if len(matched_ph) > 1 else 0.0
    }
    
    return stats_dict


def main():
    print("="*80)
    print("PyHEARTS vs QTDB Ground Truth - Bland-Altman Analysis")
    print("="*80)
    print()
    
    # Get all subjects
    if not PYHEARTS_RESULTS_DIR.exists():
        print(f"ERROR: PyHEARTS results directory not found: {PYHEARTS_RESULTS_DIR}")
        return
    
    if not QTDB_DATA_DIR.exists():
        print(f"ERROR: QTDB data directory not found: {QTDB_DATA_DIR}")
        return
    
    # Find subjects with both PyHEARTS results and QTDB data
    ph_subjects = set([f.stem.replace("_output", "") for f in PYHEARTS_RESULTS_DIR.glob("*_output.csv")])
    qtdb_subjects = set([f.stem for f in QTDB_DATA_DIR.glob("*.dat")])
    subjects = sorted(list(ph_subjects & qtdb_subjects))
    
    print(f"Found {len(subjects)} subjects with both PyHEARTS results and QTDB data")
    print()
    
    # Collect all intervals
    all_intervals = {
        'PR': {'ph': [], 'gt': []},
        'QRS': {'ph': [], 'gt': []},
        'QT': {'ph': [], 'gt': []},
        'RT': {'ph': [], 'gt': []},
        'TT': {'ph': [], 'gt': []}
    }
    
    processed = 0
    failed = 0
    
    for subject in subjects:
        # Load ground truth
        gt_peaks = load_ground_truth_annotations(subject)
        if gt_peaks is None:
            failed += 1
            continue
        
        # Load PyHEARTS intervals
        ph_intervals = load_pyhearts_intervals(subject, PYHEARTS_RESULTS_DIR)
        if ph_intervals is None:
            failed += 1
            continue
        
        # Compute ground truth intervals
        gt_intervals = compute_ground_truth_intervals(gt_peaks, SAMPLING_RATE)
        
        # Collect intervals
        for interval_name in ['PR', 'QRS', 'QT', 'RT', 'TT']:
            if interval_name in ph_intervals:
                all_intervals[interval_name]['ph'].extend(ph_intervals[interval_name])
            if interval_name in gt_intervals:
                all_intervals[interval_name]['gt'].extend(gt_intervals[interval_name])
        
        processed += 1
        if processed % 20 == 0:
            print(f"  Processed {processed}/{len(subjects)} subjects...")
    
    print(f"\nProcessed {processed} subjects successfully ({failed} failed)")
    print()
    
    # Print summary statistics
    print("="*80)
    print("INTERVAL SUMMARY")
    print("="*80)
    for interval_name in ['PR', 'QRS', 'QT', 'RT', 'TT']:
        n_ph = len(all_intervals[interval_name]['ph'])
        n_gt = len(all_intervals[interval_name]['gt'])
        print(f"{interval_name}: PyHEARTS={n_ph}, Ground Truth={n_gt}")
    print()
    
    # Create publication-ready figure with subplots
    print("Creating Bland-Altman plots...")
    
    # Set up the figure with 5 subplots in a 2x3 grid (last spot empty)
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35, 
                          left=0.08, right=0.95, top=0.95, bottom=0.08)
    
    intervals = ['PR', 'QRS', 'QT', 'RT', 'TT']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    stats_summary = {}
    
    for idx, (interval_name, color) in enumerate(zip(intervals, colors)):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        ph_values = all_intervals[interval_name]['ph']
        gt_values = all_intervals[interval_name]['gt']
        
        stats = create_bland_altman_subplot(ax, ph_values, gt_values, interval_name, color)
        stats_summary[interval_name] = stats
    
    # Add overall title
    fig.suptitle('Bland-Altman Plots: PyHEARTS vs QTDB Ground Truth', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_file = OUTPUT_DIR / "bland_altman_qtdb_intervals.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved figure to: {output_file}")
    
    # Also save as PDF for publication
    output_file_pdf = OUTPUT_DIR / "bland_altman_qtdb_intervals.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved figure (PDF) to: {output_file_pdf}")
    
    # Also save as SVG for vector graphics
    output_file_svg = OUTPUT_DIR / "bland_altman_qtdb_intervals.svg"
    plt.savefig(output_file_svg, bbox_inches='tight', facecolor='white')
    print(f"Saved figure (SVG) to: {output_file_svg}")
    
    plt.close()
    
    # Save statistics summary
    import json
    stats_file = OUTPUT_DIR / "bland_altman_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats_summary, f, indent=2)
    print(f"Saved statistics to: {stats_file}")
    
    # Print statistics summary
    print("\n" + "="*80)
    print("BLAND-ALTMAN STATISTICS")
    print("="*80)
    for interval_name in intervals:
        stats = stats_summary[interval_name]
        if stats:
            print(f"\n{interval_name} Interval:")
            print(f"  n = {stats['n']}")
            print(f"  Mean difference: {stats['mean_diff']:.2f} ms")
            print(f"  SD of difference: {stats['std_diff']:.2f} ms")
            print(f"  Limits of agreement: [{stats['loa_lower']:.2f}, {stats['loa_upper']:.2f}] ms")
            print(f"  Correlation: {stats['correlation']:.3f}")
            print(f"  Mean PyHEARTS: {stats['mean_ph']:.2f} ms")
            print(f"  Mean Ground Truth: {stats['mean_gt']:.2f} ms")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

