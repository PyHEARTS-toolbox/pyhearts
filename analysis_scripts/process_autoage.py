#!/usr/bin/env python3
"""
Process all autoage subjects with PyHEARTS and save results to a timestamped folder.
Results are saved in results/autoage_YYYYMMDD_HHMMSS/
"""

import os
import sys
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
import json
import hashlib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
AUTOAGE_DATA_DIR = PROJECT_ROOT / "data" / "autoage"

# Create timestamped results folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = PROJECT_ROOT / "results" / f"autoage_{timestamp}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Results will be saved to: {RESULTS_DIR}")

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))
from pyhearts import PyHEARTS


def get_code_hash():
    """Get hash of current code for reproducibility."""
    code_files = [
        PROJECT_ROOT / "pyhearts" / "processing" / "processcycle.py",
        PROJECT_ROOT / "pyhearts" / "processing" / "rpeak.py",
        PROJECT_ROOT / "pyhearts" / "core" / "fit.py",
    ]
    
    hasher = hashlib.sha256()
    for file_path in code_files:
        if file_path.exists():
            with open(file_path, 'rb') as f:
                hasher.update(f.read())
    
    return hasher.hexdigest()[:16]  # Short hash


def process_subject(subject, verbose=False):
    """Process a single autoage subject."""
    print(f"\n{'='*80}")
    print(f"Processing: {subject}")
    print(f"{'='*80}")
    
    # Load signal
    dat_file = AUTOAGE_DATA_DIR / f"{subject}.dat"
    if not dat_file.exists():
        print(f"  ERROR: Signal file not found")
        return None
    
    annotation_dir = dat_file.parent
    annotation_name = dat_file.stem
    old_dir = os.getcwd()
    os.chdir(annotation_dir)
    
    try:
        record = wfdb.rdrecord(annotation_name)
        signal = record.p_signal[:, 0]
        sampling_rate = record.fs
    except Exception as e:
        print(f"  ERROR: Failed to load signal: {e}")
        return None
    finally:
        os.chdir(old_dir)
    
    print(f"  Signal: {len(signal)} samples ({len(signal)/sampling_rate:.1f} seconds) at {sampling_rate} Hz")
    
    # Run PyHEARTS with human preset (now uses ecgpuwave-style high sensitivity by default)
    try:
        hearts = PyHEARTS(sampling_rate=sampling_rate, verbose=verbose, plot=False, species='human')
        output_df, epochs_df = hearts.analyze_ecg(signal)
        
        if output_df is None or len(output_df) == 0:
            print(f"  WARNING: No cycles detected")
            return None
        
        # Extract peak counts from output_df (per-cycle data)
        ph_r = output_df['R_global_center_idx'].dropna()
        ph_r = ph_r[np.isfinite(ph_r) & (ph_r > 0)].values.astype(int) if len(ph_r) > 0 else np.array([])
        
        ph_p = output_df['P_global_center_idx'].dropna() if 'P_global_center_idx' in output_df.columns else pd.Series()
        ph_p = ph_p[np.isfinite(ph_p) & (ph_p > 0)].values.astype(int) if len(ph_p) > 0 else np.array([])
        
        ph_t = output_df['T_global_center_idx'].dropna() if 'T_global_center_idx' in output_df.columns else pd.Series()
        ph_t = ph_t[np.isfinite(ph_t) & (ph_t > 0)].values.astype(int) if len(ph_t) > 0 else np.array([])
        
        ph_q = output_df['Q_global_center_idx'].dropna() if 'Q_global_center_idx' in output_df.columns else pd.Series()
        ph_q = ph_q[np.isfinite(ph_q) & (ph_q > 0)].values.astype(int) if len(ph_q) > 0 else np.array([])
        
        ph_s = output_df['S_global_center_idx'].dropna() if 'S_global_center_idx' in output_df.columns else pd.Series()
        ph_s = ph_s[np.isfinite(ph_s) & (ph_s > 0)].values.astype(int) if len(ph_s) > 0 else np.array([])
        
        print(f"  Detected: {len(ph_r)} R peaks, {len(ph_p)} P peaks, {len(ph_t)} T peaks, {len(ph_q)} Q peaks, {len(ph_s)} S peaks")
        print(f"  Cycles: {len(output_df)}")
        
        # Save results
        # Save output_df (per-cycle data)
        output_file = RESULTS_DIR / f"{subject}_output.csv"
        output_df.to_csv(output_file, index=False)
        
        # Save epochs_df (per-sample data) if available
        if epochs_df is not None and len(epochs_df) > 0:
            epochs_file = RESULTS_DIR / f"{subject}_epochs.csv"
            epochs_df.to_csv(epochs_file, index=False)
        
        # Save metadata
        meta = {
            "subject": subject,
            "sampling_rate_hz": float(sampling_rate),
            "signal_length_samples": int(len(signal)),
            "signal_duration_sec": float(len(signal) / sampling_rate),
            "total_cycles": int(len(output_df)),
            "r_peaks_detected": int(len(ph_r)),
            "p_peaks_detected": int(len(ph_p)),
            "t_peaks_detected": int(len(ph_t)),
            "q_peaks_detected": int(len(ph_q)),
            "s_peaks_detected": int(len(ph_s)),
            "code_hash": get_code_hash(),
            "timestamp": datetime.now().isoformat(),
        }
        
        meta_file = RESULTS_DIR / f"{subject}_meta.json"
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"  Saved: {output_file.name}, {meta_file.name}")
        if epochs_df is not None:
            print(f"         {epochs_file.name}")
        
        return meta
        
    except Exception as e:
        print(f"  ERROR: Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("="*80)
    print("Processing All Autoage Subjects with PyHEARTS (ecgpuwave-style default)")
    print("="*80)
    print(f"Results will be saved to: {RESULTS_DIR}")
    print(f"Config: Using default human preset (v2.0-human-ecgpuwave-style)")
    print()
    
    # Get all available subjects
    if not AUTOAGE_DATA_DIR.exists():
        print(f"ERROR: Autoage data directory not found: {AUTOAGE_DATA_DIR}")
        return
    
    subjects = []
    for dat_file in AUTOAGE_DATA_DIR.glob("*.dat"):
        subjects.append(dat_file.stem)
    
    subjects = sorted(subjects)
    print(f"Found {len(subjects)} subjects")
    print(f"First 10 subjects: {', '.join(subjects[:10])}")
    print()
    
    # Process each subject
    results = []
    failed_subjects = []
    
    start_time = datetime.now()
    
    for i, subject in enumerate(subjects, 1):
        print(f"[{i}/{len(subjects)}] ", end="")
        result = process_subject(subject, verbose=False)
        
        if result is not None:
            results.append(result)
        else:
            failed_subjects.append(subject)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Processed: {len(results)}/{len(subjects)} subjects")
    print(f"Failed: {len(failed_subjects)}")
    print(f"Total time: {duration/60:.1f} minutes ({duration:.1f} seconds)")
    
    if failed_subjects:
        print(f"\nFailed subjects: {', '.join(failed_subjects)}")
        failed_file = RESULTS_DIR / "failed_subjects.csv"
        pd.DataFrame({"subject": failed_subjects}).to_csv(failed_file, index=False)
        print(f"Saved failed subjects list to: {failed_file.name}")
    
    # Overall statistics
    if results:
        total_r = sum(r.get("r_peaks_detected", 0) for r in results)
        total_p = sum(r.get("p_peaks_detected", 0) for r in results)
        total_t = sum(r.get("t_peaks_detected", 0) for r in results)
        total_q = sum(r.get("q_peaks_detected", 0) for r in results)
        total_s = sum(r.get("s_peaks_detected", 0) for r in results)
        total_cycles = sum(r.get("total_cycles", 0) for r in results)
        
        print(f"\nOverall Statistics:")
        print(f"  Total cycles processed: {total_cycles}")
        print(f"  Total R peaks detected: {total_r}")
        print(f"  Total P peaks detected: {total_p}")
        print(f"  Total T peaks detected: {total_t}")
        print(f"  Total Q peaks detected: {total_q}")
        print(f"  Total S peaks detected: {total_s}")
        
        # Save summary
        summary_file = RESULTS_DIR / "summary.json"
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_subjects": len(subjects),
            "processed_subjects": len(results),
            "failed_subjects": len(failed_subjects),
            "failed_subject_list": failed_subjects,
            "total_cycles": total_cycles,
            "total_r_peaks": total_r,
            "total_p_peaks": total_p,
            "total_t_peaks": total_t,
            "total_q_peaks": total_q,
            "total_s_peaks": total_s,
            "processing_time_seconds": duration,
            "code_hash": get_code_hash(),
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_file.name}")
    
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

