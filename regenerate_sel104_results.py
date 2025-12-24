#!/usr/bin/env python3
"""
Regenerate pyhearts results for sel104 with updated T peak detection.
"""

import os
import sys
import numpy as np
import pandas as pd
import wfdb
from pyhearts import PyHEARTS

# QTDB path
QTDB_PATH = "/Users/morganfitzgerald/Documents/pyhearts/data/qtdb/1.0.0"
RECORD_NAME = "sel104"
DURATION_SECONDS = 15
OUTPUT_CSV = "sel104_output_15s.csv"
EPOCHS_CSV = "sel104_epochs_15s.csv"

def main():
    """Main function."""
    print(f"Regenerating pyhearts results for {RECORD_NAME}...")
    
    # Load signal
    original_dir = os.getcwd()
    os.chdir(QTDB_PATH)
    try:
        record = wfdb.rdrecord(RECORD_NAME)
        sampling_rate = record.fs
        signal = record.p_signal[:, 0]
    finally:
        os.chdir(original_dir)
    
    # Crop to first 15 seconds
    max_samples = int(DURATION_SECONDS * sampling_rate)
    signal = signal[:max_samples]
    
    print(f"Signal: {len(signal)} samples at {sampling_rate} Hz")
    
    # Run pyhearts analysis
    print("\nRunning pyhearts analysis...")
    analyzer = PyHEARTS(
        sampling_rate=sampling_rate,
        species="human",
        sensitivity="high",
        verbose=True,
        plot=False
    )
    
    output_df, epochs_df = analyzer.analyze_ecg(signal)
    
    # Count T peaks
    num_t_peaks = output_df['T_global_center_idx'].notna().sum()
    print(f"\nDetected {num_t_peaks} T peaks")
    
    # Save results
    print(f"\nSaving results to {OUTPUT_CSV} and {EPOCHS_CSV}...")
    output_df.to_csv(OUTPUT_CSV)
    epochs_df.to_csv(EPOCHS_CSV)
    
    print(f"Results saved successfully!")
    print(f"  Output: {OUTPUT_CSV}")
    print(f"  Epochs: {EPOCHS_CSV}")

if __name__ == "__main__":
    main()


