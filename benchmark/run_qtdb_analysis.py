import os
import sys
import numpy as np
import wfdb
import logging
from datetime import datetime
from pyhearts import PyHEARTS

# --- Cluster paths ---
data_dir = "/labs/bvoyteklab/ecg_param/qtdb/physionet.org/files/qtdb/1.0.0"

# Create a unique results directory per run using current date-time suffix
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"/labs/bvoyteklab/ecg_param/results/qtdb_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# --- Parse command-line arguments ---
if len(sys.argv) != 3:
    sys.exit(1)

START_INDEX = int(sys.argv[1])
END_INDEX = int(sys.argv[2])


def get_preferred_lead_index(signal_names):
    """
    Select the best lead from available signals.
    Preference order:
        ECG2 > ECG1 > MLII > V5 > V2 > CM5 > CM4 > ML5 > V1 > first available

    This prioritizes ECG2/ECG1 explicitly (as many QTDB records label lead II
    that way), while still falling back to MLII and other common leads.
    """
    preferred_order = ["ECG2", "ECG1", "MLII", "V5", "V2", "CM5", "CM4", "ML5", "V1"]
    
    # Try to find preferred leads in order
    for preferred in preferred_order:
        for idx, name in enumerate(signal_names):
            if name.upper() == preferred.upper():
                return idx
    
    # Fallback to first signal if no preferred lead found
    return 0


def load_qtdb_data(directory):
    """Load QTDB records from the specified directory."""
    data_dict = {}
    
    # Get list of record names from RECORDS file or .hea files
    records_file = os.path.join(directory, "RECORDS")
    if os.path.exists(records_file):
        with open(records_file, 'r') as f:
            record_names = [line.strip() for line in f if line.strip()]
    else:
        # Fallback: find all .hea files
        record_names = [f.replace('.hea', '') for f in os.listdir(directory) if f.endswith('.hea')]
    
    record_names = sorted(set(record_names))
    chunked_record_names = record_names[START_INDEX:END_INDEX]

    for record in chunked_record_names:
        record_path = os.path.join(directory, record)
        try:
            signals, fields = wfdb.rdsamp(record_path)
            annotations = None
            
            # Try to load .atr annotations if available
            atr_path = record_path + ".atr"
            if os.path.exists(atr_path):
                try:
                    ann = wfdb.rdann(record_path, 'atr')
                    annotations = {
                        "sample": np.array(ann.sample),
                        "symbol": np.array(ann.symbol),
                        "aux_note": np.array(ann.aux_note) if hasattr(ann, 'aux_note') else None
                    }
                except Exception:
                    annotations = None
            
            data_dict[record] = {
                "ecg_signal": np.array(signals),
                "metadata": fields,
                "annotations": annotations
            }
        except Exception as e:
            print(f"[{record}] Failed to load: {e}")

    return data_dict


def run_pyhearts_analysis(data, results_dir):
    """Run PyHEARTS analysis on all loaded records."""
    failure_reasons = {}
    lead_info_path = os.path.join(results_dir, "selected_leads.csv")

    log_dir = os.path.join(os.path.dirname(results_dir), "qtdb_logs")
    os.makedirs(log_dir, exist_ok=True)

    with open(lead_info_path, "w") as f:
        f.write("subject,lead_name,lead_index\n")

    for subject, record_data in data.items():
        # --- Logger setup per subject ---
        log_path = os.path.join(log_dir, f"{subject}_log.txt")
        logger = logging.getLogger(f"qtdb.{subject}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        while logger.handlers:
            logger.removeHandler(logger.handlers[0])
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)

        try:
            signals = record_data["ecg_signal"]
            signal_names = record_data["metadata"].get("sig_name", [])
            sampling_rate = record_data["metadata"].get("fs", None)

            if signals.ndim < 2 or signals.shape[1] == 0:
                failure_reasons[subject] = "no_signals_found"
                logger.warning("No valid signals found")
                continue

            if sampling_rate is None:
                failure_reasons[subject] = "no_sampling_rate"
                logger.warning("No sampling rate found in metadata")
                continue

            # Select preferred lead
            lead_idx = get_preferred_lead_index(signal_names)
            lead_name = signal_names[lead_idx] if lead_idx < len(signal_names) else "Unknown"
            
            logger.info(f"Selected lead: {lead_name} (index {lead_idx})")
            
            with open(lead_info_path, "a") as f:
                f.write(f"{subject},{lead_name},{lead_idx}\n")

            ecg_signal = signals[:, lead_idx]

            # Initialize PyHEARTS processor with optimized settings
            # - species="human": Uses QTDB-optimized human presets (v1.2)
            # - sensitivity="standard": Balanced detection (recommended)
            #   Options: "standard" (balanced), "high" (more R-peaks), "maximum" (aggressive gap-fill)
            # 
            # Note: "high" sensitivity previously caused T-wave collapse due to 
            # gap-filling detecting T-peaks as R-peaks. Use "standard" for balanced results.
            ecg_processor = PyHEARTS(
                sampling_rate=sampling_rate,
                species="human",        # Use optimized human presets (v1.2)
                sensitivity="standard", # Balanced - avoids T-wave regression
                verbose=False,
                plot=False
            )
            
            # Preprocess signal (QTDB uses 250 Hz, so 60 Hz notch for US power)
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
                failure_reasons[subject] = "preprocessing_failed"
                logger.warning("Preprocessing returned None")
                continue

            # Analyze ECG
            ecg_processor.analyze_ecg(ecg_filt)
            ecg_processor.compute_hrv_metrics()
            
            # Save outputs
            ecg_processor.save_output(subject, results_dir)
            ecg_processor.save_hrv_metrics(subject, results_dir)
            
            logger.info("Successfully processed")

        except Exception as e:
            failure_reasons[subject] = f"processing_failed: {str(e)}"
            logger.exception(f"Exception during processing: {e}")

        finally:
            logger.handlers.clear()
            del logger

    # Save failures
    fail_log_path = os.path.join(results_dir, "failed_subjects.csv")
    with open(fail_log_path, "w") as f:
        f.write("subject,reason\n")
        for subj, reason in failure_reasons.items():
            f.write(f"{subj},{reason}\n")


# --- Main ---
if __name__ == "__main__":
    data = load_qtdb_data(data_dir)
    run_pyhearts_analysis(data, results_dir)


