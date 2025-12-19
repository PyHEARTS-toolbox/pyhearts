#!/usr/bin/env python3
"""
Run PyHEARTS analysis on the four inverted signal QTDB participants.

This script processes:
- sele0121
- sele0122
- sele0211
- sele0409

Results are saved to qtdb_inverted-sig_subjects_analysis/pyhearts_results_new/
"""

import os
import sys
import numpy as np
import wfdb
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pyhearts import PyHEARTS

# --- Paths ---
base_dir = project_root / "qtdb_inverted-sig_subjects_analysis"
qtdb_dir = base_dir / "qtdb_raw_files"
results_dir = base_dir / "pyhearts_results_new"

# Create results directory
results_dir.mkdir(exist_ok=True)

# Subjects to process
subjects = ["sele0121", "sele0122", "sele0211", "sele0409"]

# Setup logging
log_file = results_dir / "analysis_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_preferred_lead_index(signal_names):
    """
    Select the best lead from available signals.
    Preference order:
        ECG2 > ECG1 > MLII > V5 > V2 > CM5 > CM4 > ML5 > V1 > first available
    """
    preferred_order = ["ECG2", "ECG1", "MLII", "V5", "V2", "CM5", "CM4", "ML5", "V1"]
    
    for preferred in preferred_order:
        for idx, name in enumerate(signal_names):
            if name.upper() == preferred.upper():
                return idx
    
    return 0


def load_qtdb_record(record_path):
    """Load a single QTDB record."""
    try:
        # Read signal data
        signals, fields = wfdb.rdsamp(str(record_path))
        
        # Try to load annotations
        annotations = None
        atr_path = str(record_path) + ".atr"
        if os.path.exists(atr_path):
            try:
                ann = wfdb.rdann(str(record_path), 'atr')
                annotations = {
                    "sample": np.array(ann.sample),
                    "symbol": np.array(ann.symbol),
                    "aux_note": np.array(ann.aux_note) if hasattr(ann, 'aux_note') else None
                }
            except Exception as e:
                logger.warning(f"Could not load annotations: {e}")
                annotations = None
        
        return {
            "ecg_signal": np.array(signals),
            "metadata": fields,
            "annotations": annotations
        }
    except Exception as e:
        logger.error(f"Failed to load {record_path}: {e}")
        return None


def run_pyhearts_analysis():
    """Run PyHEARTS analysis on all subjects."""
    logger.info("="*80)
    logger.info("PYHEARTS ANALYSIS ON INVERTED SIGNAL QTDB PARTICIPANTS")
    logger.info("="*80)
    logger.info(f"Results will be saved to: {results_dir}")
    logger.info(f"Subjects to process: {', '.join(subjects)}")
    logger.info("")
    
    # Track results
    results_summary = []
    lead_info_path = results_dir / "selected_leads.csv"
    
    with open(lead_info_path, "w") as f:
        f.write("subject,lead_name,lead_index\n")
    
    for subject in subjects:
        logger.info("-"*80)
        logger.info(f"Processing subject: {subject}")
        logger.info("-"*80)
        
        record_path = qtdb_dir / subject
        hea_path = qtdb_dir / f"{subject}.hea"
        
        if not hea_path.exists():
            logger.error(f"Header file does not exist: {hea_path}")
            results_summary.append({
                "subject": subject,
                "status": "failed",
                "reason": "record_not_found"
            })
            continue
        
        # Load QTDB data
        logger.info(f"Loading QTDB data from: {record_path}")
        record_data = load_qtdb_record(record_path)
        
        if record_data is None:
            results_summary.append({
                "subject": subject,
                "status": "failed",
                "reason": "load_failed"
            })
            continue
        
        try:
            signals = record_data["ecg_signal"]
            signal_names = record_data["metadata"].get("sig_name", [])
            sampling_rate = record_data["metadata"].get("fs", None)
            
            logger.info(f"  Signal shape: {signals.shape}")
            logger.info(f"  Signal names: {signal_names}")
            logger.info(f"  Sampling rate: {sampling_rate} Hz")
            
            if signals.ndim < 2 or signals.shape[1] == 0:
                logger.error("No valid signals found")
                results_summary.append({
                    "subject": subject,
                    "status": "failed",
                    "reason": "no_signals"
                })
                continue
            
            if sampling_rate is None:
                logger.error("No sampling rate found")
                results_summary.append({
                    "subject": subject,
                    "status": "failed",
                    "reason": "no_sampling_rate"
                })
                continue
            
            # Select preferred lead
            lead_idx = get_preferred_lead_index(signal_names)
            lead_name = signal_names[lead_idx] if lead_idx < len(signal_names) else "Unknown"
            logger.info(f"  Selected lead: {lead_name} (index {lead_idx})")
            
            with open(lead_info_path, "a") as f:
                f.write(f"{subject},{lead_name},{lead_idx}\n")
            
            ecg_signal = signals[:, lead_idx]
            logger.info(f"  ECG signal range: [{ecg_signal.min():.3f}, {ecg_signal.max():.3f}]")
            logger.info(f"  ECG signal length: {len(ecg_signal)} samples ({len(ecg_signal)/sampling_rate:.1f} seconds)")
            
            # Initialize PyHEARTS with human presets
            logger.info("  Initializing PyHEARTS analyzer...")
            ecg_processor = PyHEARTS(
                sampling_rate=sampling_rate,
                species="human",        # Use optimized human presets
                sensitivity="standard", # Balanced detection
                verbose=False,
                plot=False
            )
            
            # Preprocess signal
            logger.info("  Preprocessing signal...")
            ecg_filt = ecg_processor.preprocess_signal(
                ecg_signal=ecg_signal,
                highpass_cutoff=0.5,
                filter_order=4,
                lowpass_cutoff=50,
                notch_frequency=60,  # 60 Hz for US power mains
                quality_factor=30,
                poly_degree=5
            )
            
            if ecg_filt is None:
                logger.error("  Preprocessing returned None")
                results_summary.append({
                    "subject": subject,
                    "status": "failed",
                    "reason": "preprocessing_failed"
                })
                continue
            
            logger.info(f"  Filtered signal range: [{ecg_filt.min():.3f}, {ecg_filt.max():.3f}]")
            
            # Analyze ECG
            # Pass raw signal for polarity detection (preprocessing may change apparent polarity)
            logger.info("  Running PyHEARTS analysis...")
            output_df, epochs_df = ecg_processor.analyze_ecg(ecg_filt, raw_ecg=ecg_signal)
            
            if output_df is None or output_df.empty:
                logger.warning("  Analysis returned empty results")
                results_summary.append({
                    "subject": subject,
                    "status": "failed",
                    "reason": "empty_results"
                })
                continue
            
            logger.info(f"  Detected {len(output_df)} cardiac cycles")
            
            # Compute HRV metrics
            logger.info("  Computing HRV metrics...")
            ecg_processor.compute_hrv_metrics()
            
            # Save outputs
            logger.info("  Saving results...")
            ecg_processor.save_output(subject, results_dir)
            ecg_processor.save_hrv_metrics(subject, results_dir)
            
            # Get R-peak count for summary
            r_peaks = ecg_processor.r_peak_indices
            n_r_peaks = len(r_peaks) if r_peaks is not None else 0
            
            logger.info(f"  ✓ Successfully processed {subject}")
            logger.info(f"    - R-peaks detected: {n_r_peaks}")
            logger.info(f"    - Cardiac cycles: {len(output_df)}")
            
            results_summary.append({
                "subject": subject,
                "status": "success",
                "r_peaks": n_r_peaks,
                "cycles": len(output_df)
            })
            
        except Exception as e:
            logger.exception(f"  ✗ Exception during processing: {e}")
            results_summary.append({
                "subject": subject,
                "status": "failed",
                "reason": f"exception: {str(e)}"
            })
    
    # Save summary
    logger.info("")
    logger.info("="*80)
    logger.info("ANALYSIS SUMMARY")
    logger.info("="*80)
    
    summary_path = results_dir / "analysis_summary.csv"
    with open(summary_path, "w") as f:
        f.write("subject,status,r_peaks,cycles,reason\n")
        for result in results_summary:
            f.write(f"{result['subject']},{result['status']},")
            f.write(f"{result.get('r_peaks', '')},{result.get('cycles', '')},")
            f.write(f"{result.get('reason', '')}\n")
    
    # Print summary
    successful = [r for r in results_summary if r['status'] == 'success']
    failed = [r for r in results_summary if r['status'] == 'failed']
    
    logger.info(f"Successfully processed: {len(successful)}/{len(subjects)}")
    for result in successful:
        logger.info(f"  ✓ {result['subject']}: {result.get('r_peaks', 'N/A')} R-peaks, {result.get('cycles', 'N/A')} cycles")
    
    if failed:
        logger.info(f"Failed: {len(failed)}/{len(subjects)}")
        for result in failed:
            logger.info(f"  ✗ {result['subject']}: {result.get('reason', 'unknown')}")
    
    logger.info("")
    logger.info(f"Results saved to: {results_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    run_pyhearts_analysis()

