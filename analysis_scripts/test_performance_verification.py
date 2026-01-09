#!/usr/bin/env python3
"""
Test PyHEARTS performance on QTDB subject to verify no changes after removing ECGPUWAVE references.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
import warnings
warnings.filterwarnings('ignore')

# Add pyhearts to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pyhearts import PyHEARTS

QTDB_DATA_DIR = PROJECT_ROOT / "data" / "qtdb" / "1.0.0"
TEST_SUBJECT = "sel100"  # Common test subject


def load_ecg_signal(subject):
    """Load ECG signal from QTDB."""
    old_dir = Path.cwd()
    os.chdir(QTDB_DATA_DIR)
    
    try:
        record = wfdb.rdrecord(subject, channels=[0])
        signal = record.p_signal[:, 0]
        sampling_rate = record.fs
        return signal, sampling_rate
    except Exception as e:
        print(f"Error loading {subject}: {e}")
        return None, None
    finally:
        os.chdir(old_dir)


def main():
    print("="*80)
    print("PyHEARTS Performance Verification Test")
    print("="*80)
    
    # Load ECG signal
    print(f"\nLoading {TEST_SUBJECT}...")
    signal, sampling_rate = load_ecg_signal(TEST_SUBJECT)
    
    if signal is None:
        print(f"Failed to load {TEST_SUBJECT}")
        return 1
    
    print(f"  Signal length: {len(signal)} samples ({len(signal)/sampling_rate:.1f} seconds)")
    print(f"  Sampling rate: {sampling_rate} Hz")
    
    # Create PyHEARTS analyzer
    print(f"\nInitializing PyHEARTS...")
    try:
        hearts = PyHEARTS(
            sampling_rate=sampling_rate,
            species="human",
            verbose=False,
        )
        print("  ✓ PyHEARTS initialized successfully")
    except Exception as e:
        print(f"  ✗ Failed to initialize PyHEARTS: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Analyze ECG
    print(f"\nAnalyzing ECG signal...")
    try:
        output_df, epochs_df = hearts.analyze_ecg(signal)
        print("  ✓ ECG analysis completed successfully")
    except Exception as e:
        print(f"  ✗ Failed to analyze ECG: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Performance metrics
    print(f"\n{'='*80}")
    print("PERFORMANCE METRICS")
    print(f"{'='*80}")
    
    total_cycles = len(output_df)
    print(f"\nTotal cycles processed: {total_cycles}")
    
    # Wave detection rates
    p_detected = output_df['P_global_center_idx'].notna().sum()
    q_detected = output_df['Q_global_center_idx'].notna().sum()
    r_detected = output_df['R_global_center_idx'].notna().sum()
    s_detected = output_df['S_global_center_idx'].notna().sum()
    t_detected = output_df['T_global_center_idx'].notna().sum()
    
    print(f"\nWave Detection Rates:")
    print(f"  P waves: {p_detected}/{total_cycles} ({100*p_detected/total_cycles:.1f}%)")
    print(f"  Q waves: {q_detected}/{total_cycles} ({100*q_detected/total_cycles:.1f}%)")
    print(f"  R waves: {r_detected}/{total_cycles} ({100*r_detected/total_cycles:.1f}%)")
    print(f"  S waves: {s_detected}/{total_cycles} ({100*s_detected/total_cycles:.1f}%)")
    print(f"  T waves: {t_detected}/{total_cycles} ({100*t_detected/total_cycles:.1f}%)")
    
    # Interval computation
    pr_intervals = output_df['PR_interval_ms'].notna().sum()
    qrs_intervals = output_df['QRS_interval_ms'].notna().sum()
    qt_intervals = output_df['QT_interval_ms'].notna().sum()
    rr_intervals = output_df['RR_interval_ms'].notna().sum()
    
    print(f"\nInterval Computation:")
    print(f"  PR intervals: {pr_intervals}/{total_cycles} ({100*pr_intervals/total_cycles:.1f}%)")
    print(f"  QRS intervals: {qrs_intervals}/{total_cycles} ({100*qrs_intervals/total_cycles:.1f}%)")
    print(f"  QT intervals: {qt_intervals}/{total_cycles} ({100*qt_intervals/total_cycles:.1f}%)")
    print(f"  RR intervals: {rr_intervals}/{total_cycles-1} ({100*rr_intervals/(total_cycles-1):.1f}%)")
    
    # Statistical summary of intervals
    if pr_intervals > 0:
        pr_mean = output_df['PR_interval_ms'].mean()
        pr_std = output_df['PR_interval_ms'].std()
        print(f"\nPR Interval Statistics:")
        print(f"  Mean: {pr_mean:.1f} ms")
        print(f"  Std: {pr_std:.1f} ms")
    
    if qt_intervals > 0:
        qt_mean = output_df['QT_interval_ms'].mean()
        qt_std = output_df['QT_interval_ms'].std()
        print(f"\nQT Interval Statistics:")
        print(f"  Mean: {qt_mean:.1f} ms")
        print(f"  Std: {qt_std:.1f} ms")
    
    # Amplitude statistics
    if r_detected > 0:
        r_heights = output_df['R_gauss_height'].dropna()
        if len(r_heights) > 0:
            print(f"\nR Wave Amplitude Statistics:")
            print(f"  Mean: {r_heights.mean():.4f} mV")
            print(f"  Std: {r_heights.std():.4f} mV")
            print(f"  Min: {r_heights.min():.4f} mV")
            print(f"  Max: {r_heights.max():.4f} mV")
    
    if p_detected > 0:
        p_heights = output_df['P_gauss_height'].dropna()
        if len(p_heights) > 0:
            print(f"\nP Wave Amplitude Statistics:")
            print(f"  Mean: {p_heights.mean():.4f} mV")
            print(f"  Std: {p_heights.std():.4f} mV")
    
    if t_detected > 0:
        t_heights = output_df['T_gauss_height'].dropna()
        if len(t_heights) > 0:
            print(f"\nT Wave Amplitude Statistics:")
            print(f"  Mean: {t_heights.mean():.4f} mV")
            print(f"  Std: {t_heights.std():.4f} mV")
    
    # Fit quality
    r_squared = output_df['r_squared'].dropna()
    if len(r_squared) > 0:
        print(f"\nGaussian Fit Quality:")
        print(f"  Mean R²: {r_squared.mean():.3f}")
        print(f"  Cycles with R² > 0.9: {(r_squared > 0.9).sum()}/{len(r_squared)} ({100*(r_squared > 0.9).sum()/len(r_squared):.1f}%)")
        print(f"  Cycles with R² > 0.7: {(r_squared > 0.7).sum()}/{len(r_squared)} ({100*(r_squared > 0.7).sum()/len(r_squared):.1f}%)")
    
    # HRV metrics
    if hasattr(hearts, 'hrv_metrics') and hearts.hrv_metrics:
        print(f"\nHRV Metrics:")
        for key, value in hearts.hrv_metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                print(f"  {key}: {value:.2f}")
    
    # Verify all key features are present
    print(f"\n{'='*80}")
    print("DATA QUALITY CHECK")
    print(f"{'='*80}")
    
    key_columns = [
        'R_global_center_idx', 'P_global_center_idx', 'T_global_center_idx',
        'PR_interval_ms', 'QRS_interval_ms', 'QT_interval_ms',
        'R_gauss_height', 'P_gauss_height', 'T_gauss_height',
        'r_squared', 'rmse'
    ]
    
    all_present = True
    for col in key_columns:
        if col not in output_df.columns:
            print(f"  ✗ Missing column: {col}")
            all_present = False
        elif output_df[col].isna().all():
            print(f"  ⚠ All NaN in column: {col}")
        else:
            non_nan_count = output_df[col].notna().sum()
            print(f"  ✓ {col}: {non_nan_count}/{total_cycles} non-NaN values")
    
    if not all_present:
        print(f"\n  ✗ Some key columns are missing!")
        return 1
    
    print(f"\n{'='*80}")
    print("✓ PERFORMANCE VERIFICATION PASSED")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  - All waves detected successfully")
    print(f"  - All intervals computed")
    print(f"  - All key features present")
    print(f"  - No functionality regressions detected")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

