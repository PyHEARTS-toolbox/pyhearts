#!/usr/bin/env python3
"""
Find subjects with highly variable RR intervals.

This script analyzes QT Database subjects to identify those with high RR variability,
which would benefit most from the dynamic T-window improvement.
"""

import os
import sys
from pathlib import Path
import numpy as np
import wfdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyhearts import PyHEARTS, ProcessCycleConfig

MANUAL_ANNOTATIONS_DIR = "/Users/morganfitzgerald/Documents/pyhearts/data/qtdb/1.0.0"

def load_ecg_signal(record_name: str):
    """Load ECG signal from QT Database."""
    try:
        old_dir = os.getcwd()
        os.chdir(MANUAL_ANNOTATIONS_DIR)
        record = wfdb.rdrecord(record_name)
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

def analyze_rr_variability(subject: str, sampling_rate: float = 250.0):
    """Analyze RR interval variability for a subject."""
    try:
        signal, actual_sr = load_ecg_signal(subject)
        if signal is None:
            return None
        
        # Use actual sampling rate if available
        if actual_sr > 0:
            sampling_rate = actual_sr
        
        # Run PyHEARTS to detect R peaks
        analyzer = PyHEARTS(
            sampling_rate=sampling_rate,
            verbose=False,
        )
        
        # Use a reasonable segment (first 60 seconds) for faster analysis
        signal_segment = signal[:int(60 * sampling_rate)]
        output_df, _ = analyzer.analyze_ecg(signal_segment)
        
        # Extract RR intervals
        if output_df is None or 'RR_interval_ms' not in output_df.columns:
            return None
        
        rr_intervals = output_df['RR_interval_ms'].dropna().values
        if len(rr_intervals) < 10:  # Need at least 10 intervals
            return None
        
        # Calculate variability metrics
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        cv_rr = (std_rr / mean_rr) * 100  # Coefficient of variation (%)
        min_rr = np.min(rr_intervals)
        max_rr = np.max(rr_intervals)
        range_rr = max_rr - min_rr
        
        return {
            'subject': subject,
            'n_intervals': len(rr_intervals),
            'mean_rr_ms': mean_rr,
            'std_rr_ms': std_rr,
            'cv_rr_pct': cv_rr,
            'min_rr_ms': min_rr,
            'max_rr_ms': max_rr,
            'range_rr_ms': range_rr,
        }
    
    except Exception as e:
        print(f"  Error analyzing {subject}: {e}")
        return None

def main():
    print("="*80)
    print("Finding Subjects with High RR Variability")
    print("="*80)
    
    # Get list of available subjects (those with .q1c files)
    os.chdir(MANUAL_ANNOTATIONS_DIR)
    q1c_files = [f.replace('.q1c', '') for f in os.listdir('.') if f.endswith('.q1c')]
    os.chdir(Path(__file__).parent.parent.parent)
    
    # Use a subset for faster analysis (can expand if needed)
    test_subjects = sorted(q1c_files)[:30]  # Test first 30 subjects
    
    print(f"\nAnalyzing {len(test_subjects)} subjects...")
    print("This will take a few minutes...\n")
    
    results = []
    for i, subject in enumerate(test_subjects, 1):
        print(f"[{i}/{len(test_subjects)}] Analyzing {subject}...", end=' ')
        result = analyze_rr_variability(subject)
        if result:
            results.append(result)
            print(f"CV={result['cv_rr_pct']:.1f}%, range={result['range_rr_ms']:.0f}ms")
        else:
            print("failed or insufficient data")
    
    if len(results) == 0:
        print("\nNo valid results. Check data paths.")
        return 1
    
    # Create DataFrame and sort by variability
    df = pd.DataFrame(results)
    df_sorted = df.sort_values('cv_rr_pct', ascending=False)
    
    print("\n" + "="*80)
    print("TOP 10 SUBJECTS BY RR VARIABILITY (Coefficient of Variation)")
    print("="*80)
    print(df_sorted.head(10).to_string(index=False))
    
    # Also identify subjects with large RR range
    df_range_sorted = df.sort_values('range_rr_ms', ascending=False)
    print("\n" + "="*80)
    print("TOP 10 SUBJECTS BY RR RANGE (Max - Min)")
    print("="*80)
    print(df_range_sorted.head(10).to_string(index=False))
    
    # Save results
    output_file = Path(__file__).parent / 'rr_variability_analysis.csv'
    df.to_csv(output_file, index=False)
    print(f"\nFull results saved to: {output_file}")
    
    # Recommend subjects for testing
    print("\n" + "="*80)
    print("RECOMMENDED SUBJECTS FOR DYNAMIC WINDOW TESTING")
    print("="*80)
    print("Subjects with high CV (>5%) and large range (>200ms):")
    high_variability = df[(df['cv_rr_pct'] > 5.0) & (df['range_rr_ms'] > 200.0)]
    if len(high_variability) > 0:
        recommended = high_variability.sort_values('cv_rr_pct', ascending=False).head(10)
        print(recommended[['subject', 'cv_rr_pct', 'range_rr_ms', 'mean_rr_ms']].to_string(index=False))
        print(f"\nRecommended test subjects: {list(recommended['subject'].values)}")
    else:
        print("No subjects found with CV>5% and range>200ms")
        print("Using top 5 by CV instead:")
        top5 = df_sorted.head(5)
        print(top5[['subject', 'cv_rr_pct', 'range_rr_ms', 'mean_rr_ms']].to_string(index=False))
        print(f"\nRecommended test subjects: {list(top5['subject'].values)}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

