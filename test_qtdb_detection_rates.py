#!/usr/bin/env python3
"""
Process all QTDB subjects with PyHEARTS and check R-peak detection rates.
Verifies that the indentation fix works across the entire dataset.
"""

import os
import sys
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.absolute()))
from pyhearts import PyHEARTS

SCRIPT_DIR = Path(__file__).parent.absolute()
QTDB_DATA_DIR = SCRIPT_DIR / "data" / "qtdb" / "1.0.0"

# Create timestamped results folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = SCRIPT_DIR / "results" / f"qtdb_detection_rates_{timestamp}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Results will be saved to: {RESULTS_DIR}")


def load_ground_truth(subject):
    """Load ground truth annotations from QTDB."""
    annotation_dir = QTDB_DATA_DIR
    old_dir = os.getcwd()
    os.chdir(annotation_dir)
    
    try:
        # Load manual annotations
        annotation = wfdb.rdann(subject, 'atr')
        return annotation
    except Exception as e:
        return None
    finally:
        os.chdir(old_dir)


def process_subject(subject, verbose=False):
    """Process a QTDB subject and calculate detection rate."""
    print(f"\n{'='*80}")
    print(f"Processing: {subject}")
    print(f"{'='*80}")
    
    # Load signal
    dat_file = QTDB_DATA_DIR / f"{subject}.dat"
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
    
    # Load ground truth
    gt_annotation = load_ground_truth(subject)
    if gt_annotation is None:
        print(f"  WARNING: Could not load ground truth annotations")
        return None
    
    # Extract R peaks from ground truth (symbol 'N' = normal beat, plus other valid beat types)
    gt_r_peaks = np.array([gt_annotation.sample[i] for i in range(len(gt_annotation.sample)) 
                           if gt_annotation.symbol[i] in ['N', 'L', 'R', 'A', 'a', 'J', 'S', 'V', 'E', 'F']])
    print(f"  Ground truth R peaks: {len(gt_r_peaks)}")
    
    # Run PyHEARTS
    try:
        hearts = PyHEARTS(sampling_rate=sampling_rate, verbose=verbose, plot=False)
        output_df, epochs_df = hearts.analyze_ecg(signal)
        
        if output_df is None or len(output_df) == 0:
            print(f"  WARNING: No cycles detected")
            return None
        
        # Extract detected R peaks
        ph_r = output_df['R_global_center_idx'].dropna()
        ph_r = ph_r[np.isfinite(ph_r) & (ph_r > 0)].values.astype(int) if len(ph_r) > 0 else np.array([])
        
        ph_p = output_df['P_global_center_idx'].dropna() if 'P_global_center_idx' in output_df.columns else pd.Series()
        ph_p = ph_p[np.isfinite(ph_p) & (ph_p > 0)].values.astype(int) if len(ph_p) > 0 else np.array([])
        
        ph_t = output_df['T_global_center_idx'].dropna() if 'T_global_center_idx' in output_df.columns else pd.Series()
        ph_t = ph_t[np.isfinite(ph_t) & (ph_t > 0)].values.astype(int) if len(ph_t) > 0 else np.array([])
        
        print(f"  Detected: {len(ph_r)} R peaks, {len(ph_p)} P peaks, {len(ph_t)} T peaks")
        print(f"  Cycles: {len(output_df)}")
        
        # Calculate detection rate for R peaks
        detection_rate = None
        precision = None
        matched = 0
        
        if len(gt_r_peaks) > 0 and len(ph_r) > 0:
            # Match detected peaks to ground truth (within 50ms tolerance)
            tolerance_ms = 50
            tolerance_samples = int(tolerance_ms * sampling_rate / 1000.0)
            
            for gt_r in gt_r_peaks:
                distances = np.abs(ph_r - gt_r)
                if np.min(distances) <= tolerance_samples:
                    matched += 1
            
            detection_rate = (matched / len(gt_r_peaks)) * 100.0
            precision = (matched / len(ph_r)) * 100.0 if len(ph_r) > 0 else 0.0
            
            print(f"  Detection Rate (Recall): {detection_rate:.2f}%")
            print(f"  Precision: {precision:.2f}%")
            print(f"  Matched: {matched}/{len(gt_r_peaks)}")
            
            if detection_rate >= 80.0:
                print(f"  ✓ PASS: Detection rate is above 80%")
            else:
                print(f"  ✗ FAIL: Detection rate is below 80%")
        else:
            print(f"  WARNING: Cannot calculate detection rate (GT: {len(gt_r_peaks)}, Detected: {len(ph_r)})")
        
        return {
            "subject": subject,
            "gt_r_peaks": len(gt_r_peaks),
            "detected_r_peaks": len(ph_r),
            "matched_r_peaks": matched,
            "detection_rate": detection_rate,
            "precision": precision,
            "detected_p_peaks": len(ph_p),
            "detected_t_peaks": len(ph_t),
            "total_cycles": len(output_df),
            "sampling_rate_hz": float(sampling_rate),
            "signal_length_samples": int(len(signal)),
        }
        
    except Exception as e:
        print(f"  ERROR: Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("="*80)
    print("Processing All QTDB Subjects - Detection Rate Verification")
    print("="*80)
    print(f"Results will be saved to: {RESULTS_DIR}")
    print()
    
    # Get all available subjects
    if not QTDB_DATA_DIR.exists():
        print(f"ERROR: QTDB data directory not found: {QTDB_DATA_DIR}")
        return
    
    subjects = []
    for dat_file in QTDB_DATA_DIR.glob("*.dat"):
        subjects.append(dat_file.stem)
    
    subjects = sorted(subjects)
    print(f"Found {len(subjects)} subjects")
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
    
    # Detection rate statistics
    if results:
        valid_results = [r for r in results if r.get("detection_rate") is not None]
        
        if valid_results:
            detection_rates = [r["detection_rate"] for r in valid_results]
            precisions = [r["precision"] for r in valid_results if r.get("precision") is not None]
            
            print(f"\nDetection Rate Statistics ({len(valid_results)} subjects with valid rates):")
            print(f"  Mean Detection Rate: {np.mean(detection_rates):.2f}%")
            print(f"  Median Detection Rate: {np.median(detection_rates):.2f}%")
            print(f"  Min Detection Rate: {np.min(detection_rates):.2f}%")
            print(f"  Max Detection Rate: {np.max(detection_rates):.2f}%")
            print(f"  Std Dev: {np.std(detection_rates):.2f}%")
            
            if precisions:
                print(f"\nPrecision Statistics:")
                print(f"  Mean Precision: {np.mean(precisions):.2f}%")
                print(f"  Median Precision: {np.median(precisions):.2f}%")
                print(f"  Min Precision: {np.min(precisions):.2f}%")
                print(f"  Max Precision: {np.max(precisions):.2f}%")
            
            # Count subjects above 80% threshold
            above_80 = sum(1 for r in detection_rates if r >= 80.0)
            print(f"\n  Subjects above 80% detection rate: {above_80}/{len(valid_results)} ({above_80/len(valid_results)*100:.1f}%)")
            
            # Subjects below 80%
            below_80 = [r for r in valid_results if r["detection_rate"] < 80.0]
            if below_80:
                print(f"\n  Subjects below 80% detection rate ({len(below_80)}):")
                for r in below_80:
                    print(f"    {r['subject']}: {r['detection_rate']:.2f}%")
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_file = RESULTS_DIR / "detection_rates.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")
        
        # Save summary
        summary_file = RESULTS_DIR / "summary.json"
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_subjects": len(subjects),
            "processed_subjects": len(results),
            "failed_subjects": len(failed_subjects),
            "failed_subject_list": failed_subjects,
            "subjects_with_valid_rates": len(valid_results) if valid_results else 0,
            "mean_detection_rate": float(np.mean(detection_rates)) if valid_results and detection_rates else None,
            "median_detection_rate": float(np.median(detection_rates)) if valid_results and detection_rates else None,
            "min_detection_rate": float(np.min(detection_rates)) if valid_results and detection_rates else None,
            "max_detection_rate": float(np.max(detection_rates)) if valid_results and detection_rates else None,
            "subjects_above_80_percent": above_80 if valid_results else 0,
            "processing_time_seconds": duration,
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_file.name}")
    
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

