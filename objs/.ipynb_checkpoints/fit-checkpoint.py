import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Any, Literal


import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from hashlib import sha256
import subprocess

from pyhearts.config import ProcessCycleConfig   
from pyhearts.feature import calc_hrv_metrics
from pyhearts.processing import (
    epoch_ecg,
    initialize_output_dict,
    preprocess_ecg,
    process_cycle,
    r_peak_detection,
)


from typing import Optional, Tuple, Any, Literal
from dataclasses import replace

class PyHEARTS:
    """
    A class for analyzing ECG signals using PyHEARTS.

    This includes preprocessing, R-peak detection, cycle segmentation,
    waveform feature extraction, shape analysis, and HRV metric computation.
    """
    def __init__(
        self,
        sampling_rate: float,
        verbose: bool = False,
        plot: bool = False,
        cfg: Optional[ProcessCycleConfig] = None,
        *,
        species: Optional[Literal["human", "mouse"]] = None,
        **overrides: Any,
    ):
        self.sampling_rate = sampling_rate
        self.verbose = verbose
        self.plot = plot

        # 1) choose a base config
        if cfg is not None:
            base = cfg
        else:
            if species == "mouse":
                base = ProcessCycleConfig.for_mouse()
            elif species == "human":
                base = ProcessCycleConfig.for_human()
            else:
                # wide defaults (species-agnostic)
                base = ProcessCycleConfig()

        # 2) apply any field-level overrides (validated)
        for k in overrides:
            if not hasattr(base, k):
                raise TypeError(f"Unknown config key: {k}")
        self.cfg: ProcessCycleConfig = replace(base, **overrides)

        # internals
        self.output_dict: Optional[dict] = None
        self.previous_r_center_samples: Optional[np.ndarray] = None
        self.previous_p_center_samples: Optional[np.ndarray] = None
        self.previous_gauss_features: Optional[dict] = None
        self.sig_corrected_dict: dict = {}
        self.hrv_metrics: dict = {}
    ######     
    # ===== Repro/metadata helpers (private) =====
    def _git_info(self) -> dict:
        def run(cmd):
            try:
                return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
            except Exception:
                return None
        return {
            "commit": run(["git", "rev-parse", "HEAD"]),
            "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "status_clean": (run(["git", "status", "--porcelain"]) == ""),
        }

    def _code_sha256(self) -> str | None:
        """Return SHA256 of this source file for reproducibility."""
        try:
            return sha256(Path(__file__).read_bytes()).hexdigest()
        except Exception:
            return None
            
    def _resolved_config(self) -> dict:
        return asdict(self.cfg)

    def _metadata_payload(self) -> dict:
        cfg = self._resolved_config()
        cfg_hash = sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        return {
            "pyhearts_version": cfg.get("version"),
            "sampling_rate_hz": self.sampling_rate,
            "verbose": self.verbose,
            "plot": self.plot,
            "config": cfg,
            "config_sha256": cfg_hash,
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "git": self._git_info(),
            "code_sha256": self._code_sha256(),   # <— here
        }
    def _save_metadata(self, file_id: str, results_dir: str) -> None:
        path = Path(results_dir) / f"{file_id}_meta.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(
                    self._metadata_payload(),
                    f,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False
                )
            logging.info("Saved metadata to %s", path)
        except ValueError as ve:
            # triggers if NaN/Inf present due to allow_nan=False
            logging.error("Metadata contains non-JSON numbers: %s", ve)
            raise
        except Exception as e:
            logging.error("Failed to save metadata: %s", e)
            raise
    # ===== Public API below =====
    def preprocess_signal(
        self,
        ecg_signal: np.ndarray,
        highpass_cutoff: Optional[float] = None,
        filter_order: Optional[int] = None,
        lowpass_cutoff: Optional[float] = None,
        notch_frequency: Optional[float] = None,
        quality_factor: Optional[float] = None,
        poly_degree: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Preprocess an ECG signal.
    
        Applies optional high-pass, low-pass, and notch filters, as well as polynomial detrending,
        to remove baseline wander, noise, and line interference from the input ECG signal.
    
        Parameters
        ----------
        ecg_signal : np.ndarray
            Raw ECG signal array (in mV).
        highpass_cutoff : float, optional
            High-pass filter cutoff frequency in Hz. If None, no high-pass filtering is applied.
        filter_order : int, optional
            Order of the digital filter. If None, uses the default in `preprocess_ecg`.
        lowpass_cutoff : float, optional
            Low-pass filter cutoff frequency in Hz. If None, no low-pass filtering is applied.
        notch_frequency : float, optional
            Notch filter center frequency in Hz (e.g., 50 or 60 for mains noise). If None, no notch is applied.
        quality_factor : float, optional
            Quality factor for the notch filter. Ignored if `notch_frequency` is None.
        poly_degree : int, optional
            Degree of polynomial detrending to apply. If None, no polynomial detrending is applied.
    
        Returns
        -------
        np.ndarray or None
            Preprocessed ECG signal, or None if preprocessing fails.
        """
        return preprocess_ecg(
            ecg_signal,
            self.sampling_rate,
            highpass_cutoff,
            filter_order,
            lowpass_cutoff,
            notch_frequency,
            quality_factor,
            poly_degree,
        )

    def initialize_output_dict(
        self,
        cycle_inds,
        components,
        peak_features,
        intervals,
        pairwise_differences=None,
    ):
        """
        Create and initialize the output dictionary for ECG feature extraction.
    
        Sets up an empty structure to store per-cycle morphology features, timing intervals,
        and pairwise voltage differences.
    
        Parameters
        ----------
        cycle_inds : array-like
            Sequence of cycle indices to initialize.
        components : list of str
            ECG waveform labels (e.g., ["P", "Q", "R", "S", "T"]).
        peak_features : list of str
            Keys for per-waveform morphological features (e.g., height, duration).
        intervals : list of str
            Keys for inter-wave timing intervals.
        pairwise_differences : list of str, optional
            Keys for voltage difference features between waveform pairs.
    
        Returns
        -------
        dict
            Dictionary with initialized keys and NaN-filled values ready for feature population.
        """
        return initialize_output_dict(
            cycle_inds=cycle_inds,
            components=components,
            peak_features=peak_features,
            intervals=intervals,
            pairwise_differences=pairwise_differences,
        )

    def process_cycle_wrapper(self, one_cycle: pd.DataFrame, cycle_idx: int):
        """
        Process and extract features from a single ECG cycle.
    
        Wraps `process_cycle` to update internal state with previous R/P indices, Gaussian
        fit parameters, and optionally corrected signals.
    
        Parameters
        ----------
        one_cycle : pd.DataFrame
            DataFrame containing the time-series samples for one ECG cycle.
        cycle_idx : int
            Index of the cycle within the overall ECG signal.
    
        Returns
        -------
        None
            Updates internal attributes: output_dict, previous centers, Gaussian parameters,
            and corrected signal dictionary.
        """
        (
            self.output_dict,
            self.previous_r_center_samples,
            self.previous_p_center_samples,
            sig_corrected,
            self.previous_gauss_features,
        ) = process_cycle(
            one_cycle,
            self.output_dict,
            self.sampling_rate,
            cycle_idx,
            self.previous_r_center_samples,
            self.previous_p_center_samples,
            previous_gauss_features=self.previous_gauss_features,
            expected_max_energy=self.expected_max_energy,
            plot=self.plot,
            verbose=self.verbose,
            cfg=self.cfg,   
        )


        if sig_corrected is not None:
            self.sig_corrected_dict[cycle_idx] = sig_corrected

    def analyze_ecg(
        self,
        ecg_signal: np.ndarray,
        verbose: Optional[bool] = None,
        plot: Optional[bool] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the full PyHEARTS ECG analysis pipeline.
    
        Steps:
        1. Detect R-peaks.
        2. Segment the ECG into cycles.
        3. Initialize output structures.
        4. Extract morphological and interval features for each cycle.
        5. Return the final feature table and segmented cycles.
    
        Parameters
        ----------
        ecg_signal : np.ndarray
            Raw ECG signal array (in mV).
        verbose : bool, optional
            If True, print detailed progress/logging messages.
        plot : bool, optional
            If True, plot intermediate processing results.
    
        Returns
        -------
        pd.DataFrame
            Feature table for all processed cycles.
        pd.DataFrame
            DataFrame of segmented ECG cycles with labels.
        """
        
        if verbose is not None:
            self.verbose = verbose
        if plot is not None:
            self.plot = plot

        self.previous_r_center_samples = None
        self.previous_p_center_samples = None
        self.previous_gauss_features = None
        self.sig_corrected_dict = {}
        self.output_dict = None
        self.hrv_metrics = {}

        try:
            filtered_r_peaks = r_peak_detection(
                    ecg_signal, 
                self.sampling_rate, 
                cfg=self.cfg,
                plot=self.plot,
            )
            self.r_peak_indices = filtered_r_peaks
        
            # Handle no R-peaks case
            if filtered_r_peaks is None or len(filtered_r_peaks) == 0:

                logging.warning("No R-peaks detected. Analysis cannot proceed.")
                self.epochs_df = pd.DataFrame()
                self.output_df = pd.DataFrame()
                return self.output_df, self.epochs_df

            epochs_df, expected_max_energy = epoch_ecg(
                ecg_signal,
                filtered_r_peaks,
                self.sampling_rate,
                plot=self.plot,
                verbose=self.verbose,
                corr_thresh=self.cfg.epoch_corr_thresh,
                var_thresh=self.cfg.epoch_var_thresh,
                estimate_energy=True,
            )

            self.epochs_df = epochs_df
            self.expected_max_energy = expected_max_energy
            
            component_keys = ["P", "Q", "R", "S", "T"]
            peak_feature_keys = [
                "global_center_idx",
                "global_le_idx",
                "global_ri_idx",

                "center_ms",
                "le_ms",
                "ri_ms",
                
                "center_idx",
                "le_idx",
                "ri_idx",
                
                "gauss_center",
                "gauss_height",
                
                "gauss_stdev_samples",
                "gauss_stdev_ms",
                "gauss_fwhm_samples",
                "gauss_fwhm_ms",
                
                "center_voltage",
                "le_voltage",
                "ri_voltage",
                
                "duration_ms",
                "rise_ms",
                "decay_ms",
                
                "rdsm",
                "sharpness",

                "voltage_integral_uv_ms",
            ]
            interval_keys = [
                "PR_interval_ms",
                "PR_segment_ms",
                "QRS_interval_ms",
                "ST_segment_ms",
                "ST_interval_ms",
                "QT_interval_ms",
                "PP_interval_ms",
                "RR_interval_ms",
            ]
            
            pairwise_diff_keys = [
                "R_minus_S_voltage_diff_signed",
                "R_minus_P_voltage_diff_signed",
                "T_minus_R_voltage_diff_signed",
            ]

            # Use the actual cycle labels from epochs_df (sorted for determinism)
            cycles = np.sort(epochs_df["cycle"].unique())
            
            self.output_dict = self.initialize_output_dict(
                cycle_inds=np.arange(len(cycles)),
                components=component_keys,
                peak_features=peak_feature_keys,
                intervals=interval_keys,
                pairwise_differences=pairwise_diff_keys,
            )
            
            for cycle_idx, cycle_label in enumerate(cycles):
                one_cycle = epochs_df.loc[epochs_df["cycle"] == cycle_label]
                try:
                    self.process_cycle_wrapper(one_cycle, cycle_idx)
                except Exception as e:
                    logging.error(f"Error processing cycle {cycle_idx}: {e}")
                    continue

            self.output_df = pd.DataFrame.from_dict(self.output_dict, orient="columns")
            self.output_df.index.name = "cycle_index"
    
            return self.output_df, self.epochs_df #output_df, epochs_df = analyzer.analyze_ecg(signal) return both for acessible unpacking

        
        except Exception as e:
            logging.error(f"Error in analyze_ecg: {e}")
            self.output_df = pd.DataFrame()
            self.epochs_df = pd.DataFrame()
            return self.output_df, self.epochs_df

    def save_output(self, file_id: str, results_dir: str):
        """
        Save the extracted ECG features to a CSV file.
    
        Parameters
        ----------
        file_id : str
            Identifier for the ECG recording (used in filename).
        results_dir : str
            Directory where the CSV will be saved.
    
        Returns
        -------
        None
            Writes `{file_id}_pyhearts.csv` to the results directory.
        """
        output_path = f"{results_dir}/{file_id}_pyhearts.csv"
        try:
            output_df = pd.DataFrame.from_dict(self.output_dict, orient="columns")
            output_df.index.name = "cycle_index"
            output_df.to_csv(output_path, index=True, na_rep="NaN")
            self._save_metadata(file_id, results_dir)
            logging.info(f"Data saved to {output_path} successfully.")
        except Exception as e:
            logging.error(f"Error saving output for {file_id}: {e}")

    def compute_hrv_metrics(self):
        """
        Compute heart rate variability (HRV) metrics from R-R intervals.
    
        Uses `calc_hrv_metrics` to compute standard HRV measures, including:
        - Average heart rate (bpm)
        - SDNN: standard deviation of NN intervals
        - RMSSD: root mean square of successive differences
        - NN50: count of interval pairs differing by >50 ms
    
        Returns
        -------
        None
            Updates `self.hrv_metrics` with computed values, or an empty dict if computation fails.
        """
        try:
            if self.output_dict is None or "RR_interval_ms" not in self.output_dict:
                raise ValueError("RR intervals are missing in output_dict. Cannot compute HRV metrics.")

            rr_intervals = np.array(self.output_dict["RR_interval_ms"])
            clean_rr_intervals = rr_intervals[~np.isnan(rr_intervals)]
            self.rr_intervals_ms = clean_rr_intervals

            if len(clean_rr_intervals) < 2:
                raise ValueError("Insufficient valid R-R intervals for HRV computation.")

            if len(clean_rr_intervals) < 60:
                logging.info(f"Skipping HRV computation — only {len(clean_rr_intervals)} RR intervals.")
                self.hrv_metrics = {}
                return

            average_heart_rate, sdnn, rmssd, nn50 = calc_hrv_metrics(clean_rr_intervals)
            self.hrv_metrics = {
                "average_heart_rate": average_heart_rate,
                "sdnn": sdnn,
                "rmssd": rmssd,
                "nn50": nn50,
            }

        except ValueError as ve:
            logging.warning(f"Validation Error in compute_hrv_metrics: {ve}")
            self.hrv_metrics = {}
        except Exception as e:
            logging.error(f"Unexpected error in compute_hrv_metrics: {e}")
            self.hrv_metrics = {}
    
    def save_hrv_metrics(self, file_id: str, results_dir: str):
        """
        Save computed HRV metrics to a CSV file.
    
        Parameters
        ----------
        file_id : str
            Identifier for the ECG recording (used in filename).
        results_dir : str
            Directory where the CSV will be saved.
    
        Returns
        -------
        None
            Writes `{file_id}_hrv_metrics.csv` to the results directory.
        """
        try:
            if not self.hrv_metrics:
                logging.info(f"HRV metrics are empty for file {file_id}. Nothing to save.")
                return

            hrv_df = pd.DataFrame([self.hrv_metrics])
            output_path = f"{results_dir}/{file_id}_hrv_metrics.csv"
            hrv_df.to_csv(output_path, index=False)
            self._save_metadata(file_id, results_dir)
            logging.info(f"HRV metrics for {file_id} saved to {output_path}.")

        except Exception as e:
            logging.error(f"Unexpected error in save_hrv_metrics for {file_id}: {e}")

    def save_rr_intervals(self, file_id: str, results_dir: str):
        """
        Save cleaned R-R interval series to a CSV file.
    
        Parameters
        ----------
        file_id : str
            Identifier for the ECG recording (used in filename).
        results_dir : str
            Directory where the CSV will be saved.
    
        Returns
        -------
        None
            Writes `{file_id}_rr_intervals.csv` to the results directory.
        """
        try:
            if not hasattr(self, "rr_intervals_ms") or len(self.rr_intervals_ms) == 0:
                logging.warning(f"No RR intervals found for file {file_id}.")
                return

            rr_df = pd.DataFrame({"rr_interval_ms": self.rr_intervals_ms})
            output_path = f"{results_dir}/{file_id}_rr_intervals.csv"
            rr_df.to_csv(output_path, index=False)
            self._save_metadata(file_id, results_dir)
            logging.info(f"RR intervals for {file_id} saved to {output_path}.")

        except Exception as e:
            logging.error(f"Unexpected error in save_rr_intervals for {file_id}: {e}")
