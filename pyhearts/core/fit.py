from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pyhearts.config import ProcessCycleConfig
from pyhearts.feature import calc_hrv_metrics, compute_beat_to_beat_variability
from pyhearts.processing import (
    epoch_ecg,
    initialize_output_dict,
    preprocess_ecg,
    process_cycle,
    r_peak_detection,
)

class PyHEARTS:
    """
    A class for analyzing ECG signals using PyHEARTS.

    This includes preprocessing, R-peak detection, cycle segmentation,
    waveform feature extraction, shape analysis, and HRV metric computation.
    
    Key parameters for tuning detection performance:
    
    sensitivity : {"standard", "high", "maximum"}
        Controls detection sensitivity vs. precision trade-off:
        - "standard": Balanced (default, ~57% precision)
        - "high": Higher recall (+15-20%), slightly lower precision
        - "maximum": Maximum recall, may include some noise
    
    species : {"human", "mouse"}
        Uses species-specific presets optimized for physiology.
        
    Based on QTDB benchmark (Dec 2024):
    - Fiducial accuracy: <8ms average error when detected
    - Use "high" sensitivity for improved R-peak recall (>70% vs ~51%)
    - Human preset optimized for PR/QT interval accuracy
    """
    def __init__(
        self,
        sampling_rate: float,
        verbose: bool = False,
        plot: bool = False,
        cfg: Optional[ProcessCycleConfig] = None,
        *,
        species: Optional[Literal["human", "mouse"]] = None,
        sensitivity: Literal["standard", "high", "maximum"] = "standard",
        **overrides: Any,
    ):
        self.sampling_rate = sampling_rate
        self.verbose = verbose
        self.plot = plot
        self.sensitivity = sensitivity

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
        self.variability_metrics: dict = {}
        self.variability_metrics: dict = {}
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
        payload = {
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
            "code_sha256": self._code_sha256(),
        }
        # Add warning flag for Q/S detection quality at lower sampling rates
        # Q and S waves are narrow, high-frequency components; detection quality
        # may be reduced at sampling rates below 300 Hz
        if self.sampling_rate < 300.0:
            payload["quality_warnings"] = {
                "q_s_wave_detection": "Q and S wave detection may be impaired at sampling rates below 300 Hz due to reduced temporal resolution"
            }
        return payload
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

    def process_cycle_wrapper(self, one_cycle: pd.DataFrame, cycle_idx: int, precomputed_peaks: dict | None = None, full_derivative: np.ndarray | None = None, p_training_signal_peak: float | None = None, p_training_noise_peak: float | None = None):
        """
        Process and extract features from a single ECG cycle.
    
        Wraps `process_cycle` to update internal state with previous R/P indices, Gaussian
        fit parameters, and optionally corrected signals.
        
        # CRITICAL: Always log entry for cycles 50-60 to debug missing cycles 51-59
        if 50 <= cycle_idx <= 60:
            logging.info(f"[PROCESS_CYCLE_WRAPPER_ENTRY] Cycle {cycle_idx}: Entered process_cycle_wrapper, one_cycle len={len(one_cycle) if one_cycle is not None else None}")
    
        Parameters
        ----------
        one_cycle : pd.DataFrame
            DataFrame containing the time-series samples for one ECG cycle.
        cycle_idx : int
            Index of the cycle within the overall ECG signal.
        precomputed_peaks : dict, optional
            Precomputed peak annotations (not currently used).
        full_derivative : np.ndarray, optional
            Full-signal derivative for T-peak detection.
        p_training_signal_peak : float, optional
            P wave training phase signal peak threshold (adaptive signal/noise separation).
        p_training_noise_peak : float, optional
            P wave training phase noise peak threshold (adaptive signal/noise separation).
    
        Returns
        -------
        None
            Updates internal attributes: output_dict, previous centers, Gaussian parameters,
            and corrected signal dictionary.
        """
        # CRITICAL: Always log entry for cycles 50-60 to debug missing cycles 51-59
        if 50 <= cycle_idx <= 60:
            logging.info(f"[PROCESS_CYCLE_WRAPPER_ENTRY] Cycle {cycle_idx}: Entered process_cycle_wrapper, one_cycle len={len(one_cycle) if one_cycle is not None else None}")
        
        # CRITICAL: Wrap process_cycle call in try-except to catch any silent failures
        try:
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
            precomputed_peaks=precomputed_peaks,
            original_r_peaks=self.r_peak_indices if hasattr(self, 'r_peak_indices') else None,
            full_derivative=full_derivative,
            p_training_signal_peak=p_training_signal_peak,
            p_training_noise_peak=p_training_noise_peak,
        )
        except Exception as e:
            # CRITICAL: Always log exceptions in process_cycle to prevent silent failures
            logging.error(f"[PROCESS_CYCLE_WRAPPER_ERROR] Cycle {cycle_idx}: Exception in process_cycle: {e}")
            import traceback
            logging.error(f"[PROCESS_CYCLE_WRAPPER_ERROR] Cycle {cycle_idx} traceback:\n{traceback.format_exc()}")
            # Re-raise to let the caller handle it (they have their own exception handling)
            raise

        if sig_corrected is not None:
            self.sig_corrected_dict[cycle_idx] = sig_corrected

    def analyze_ecg(
        self,
        ecg_signal: np.ndarray,
        verbose: Optional[bool] = None,
        plot: Optional[bool] = None,
        raw_ecg: Optional[np.ndarray] = None,
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
            Preprocessed ECG signal array (in mV).
        verbose : bool, optional
            If True, print detailed progress/logging messages.
        plot : bool, optional
            If True, plot intermediate processing results.
        raw_ecg : np.ndarray, optional
            Raw (unfiltered) ECG signal for polarity detection. If provided, polarity
            is detected on the raw signal to avoid issues where preprocessing may
            change apparent signal polarity. If None, polarity is detected on the
            filtered signal.
    
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
        self.variability_metrics = {}
        self.variability_metrics = {}

        try:
            # Check signal quality before processing
            from pyhearts.processing.quality import assess_signal_quality
            
            is_acceptable, quality_metrics, quality_reason = assess_signal_quality(
                ecg_signal,
                self.sampling_rate,
                min_snr_db=15.0,  # Minimum 15 dB SNR
                min_amplitude_range_mv=0.3,  # Minimum 0.3 mV peak-to-peak
                max_baseline_wander_mv=0.3,  # Maximum 0.3 mV baseline wander
            )
            
            if not is_acceptable:
                if self.verbose:
                    logging.warning(
                        f"Signal quality check failed: {quality_reason}. "
                        f"Metrics: SNR={quality_metrics.get('snr_db', 'N/A'):.1f} dB, "
                        f"Amplitude={quality_metrics.get('amplitude_range_mv', 'N/A'):.3f} mV"
                    )
                # Continue anyway but log the warning
                # (Don't fail completely - let user decide)
            
            # Use requested R-peak detector
            if self.cfg.rpeak_method == "pan_tompkins":
                from pyhearts.processing.pan_tompkins import pan_tompkins_r_peak_detection
                filtered_r_peaks = pan_tompkins_r_peak_detection(
                    ecg_signal,
                    self.sampling_rate,
                    cfg=self.cfg,
                    plot=self.plot,
                    raw_ecg=raw_ecg,
                )
            elif self.cfg.rpeak_method == "bandpass_energy":
                from pyhearts.processing.bandpass_energy_rpeak import bandpass_energy_r_peak_detection
                filtered_r_peaks = bandpass_energy_r_peak_detection(
                    ecg_signal,
                    self.sampling_rate,
                    cfg=self.cfg,
                    plot=self.plot,
                    raw_ecg=raw_ecg,
                )
            else:
                filtered_r_peaks = r_peak_detection(
                    ecg_signal, 
                    self.sampling_rate, 
                    cfg=self.cfg,
                    plot=self.plot,
                    sensitivity=self.sensitivity,
                    raw_ecg=raw_ecg,
                )
            self.r_peak_indices = filtered_r_peaks
        
            # Handle no R-peaks case
            if filtered_r_peaks is None or len(filtered_r_peaks) == 0:

                logging.warning("No R-peaks detected. Analysis cannot proceed.")
                self.epochs_df = pd.DataFrame()
                self.output_df = pd.DataFrame()
                return self.output_df, self.epochs_df

            # Use peak-level validation instead of cycle-level filtering
            # This processes all detected R-peaks and validates at the peak level
            epochs_df, expected_max_energy = epoch_ecg(
                ecg_signal,
                filtered_r_peaks,
                self.sampling_rate,
                plot=self.plot,
                verbose=self.verbose,
                corr_thresh=self.cfg.epoch_corr_thresh,
                var_thresh=self.cfg.epoch_var_thresh,
                estimate_energy=True,
                skip_template_filtering=True,  # Validate at peak level
            )

            self.epochs_df = epochs_df
            self.expected_max_energy = expected_max_energy
            
            # Use the actual cycle labels from epochs_df (sorted for determinism)
            cycles = np.sort(epochs_df["cycle"].unique())
            
            # CRITICAL: Log cycles array to check if cycles 51-59 are present
            logging.info(f"[CYCLES_ARRAY] Total cycles: {len(cycles)}, cycles 50-60: {cycles[50:61] if len(cycles) > 60 else cycles[50:]}")
            if len(cycles) > 60:
                cycles_51_59 = cycles[(cycles >= 51) & (cycles <= 59)]
                logging.info(f"[CYCLES_ARRAY] Cycles 51-59 in array: {cycles_51_59}")
                if len(cycles_51_59) < 9:
                    logging.warning(f"[CYCLES_ARRAY] WARNING: Only {len(cycles_51_59)} of 9 expected cycles (51-59) found in cycles array!")
            
            # Precomputed peaks are no longer used (derivative-based detection removed)
            precomputed_peaks = None
            
            # Compute P wave training phase thresholds (adaptive signal/noise separation)
            # Analyzes first 1-3 seconds to learn P wave signal vs noise characteristics
            p_training_signal_peak = None
            p_training_noise_peak = None
            if len(cycles) > 0:
                from pyhearts.processing.p_training_phase import compute_p_training_phase_thresholds
                # Build full detrended signal for training phase (cycles are already detrended)
                max_idx = epochs_df["index"].max()
                full_signal_for_training = np.zeros(int(max_idx) + 1)
                for cycle_label in cycles[:min(10, len(cycles))]:  # Use first 10 cycles for training
                    cycle_data = epochs_df.loc[epochs_df["cycle"] == cycle_label].sort_values('index')
                    cycle_indices = cycle_data["index"].values.astype(int)
                    cycle_signal = cycle_data["signal_y"].values
                    cycle_start = int(cycle_indices[0])
                    cycle_end = int(cycle_indices[-1])
                    full_signal_for_training[cycle_start:cycle_end+1] = cycle_signal
                
                try:
                    p_training_signal_peak, p_training_noise_peak = compute_p_training_phase_thresholds(
                        full_signal_for_training,
                        self.sampling_rate,
                        training_start_sec=1.0,
                        training_end_sec=3.0,
                        bandpass_low_hz=self.cfg.pwave_bandpass_low_hz,
                        bandpass_high_hz=self.cfg.pwave_bandpass_high_hz,
                        bandpass_order=self.cfg.pwave_bandpass_order,
                    )
                    if self.verbose:
                        logging.info(f"P training phase: signal_peak={p_training_signal_peak:.4f} mV, noise_peak={p_training_noise_peak:.4f} mV")
                except Exception as e:
                    if self.verbose:
                        logging.warning(f"P training phase failed: {e}, using defaults")
                    p_training_signal_peak = None
                    p_training_noise_peak = None
            
            # Compute full-signal derivative for T-peak detection (reduces edge artifacts)
            # Build full detrended signal from cycles (cycles are already detrended in epoch.py)
            full_derivative = None
            if len(cycles) > 0:
                from pyhearts.processing.derivative_t_detection import compute_filtered_derivative
                from scipy.signal import detrend as scipy_detrend
                
                # Build full signal from cycles
                max_idx = epochs_df["index"].max()
                full_signal = np.zeros(int(max_idx) + 1)
                
                for cycle_label in cycles:
                    cycle_data = epochs_df.loc[epochs_df["cycle"] == cycle_label].sort_values('index')
                    cycle_indices = cycle_data["index"].values.astype(int)
                    cycle_signal = cycle_data["signal_y"].values
                    
                    cycle_start = int(cycle_indices[0])
                    cycle_end = int(cycle_indices[-1])
                    full_signal[cycle_start:cycle_end+1] = cycle_signal
                
                # Compute derivative on full signal (avoids edge artifacts from filtering cycle segments)
                full_derivative = compute_filtered_derivative(
                    full_signal,
                    self.sampling_rate,
                    lowpass_cutoff=40.0,
                )
            
            component_keys = ["P", "Q", "R", "S", "T"]
            peak_feature_keys = [
                # Global indices (absolute sample indices)
                "global_center_idx",
                "global_le_idx",
                "global_ri_idx",

                # Time-domain locations (ms relative to cycle)
                "center_ms",
                "le_ms",
                "ri_ms",

                # Local indices (within cycle / detrended segment)
                "center_idx",
                "le_idx",
                "ri_idx",

                # Gaussian fit parameters (morphology)
                "gauss_center",
                "gauss_height",
                "gauss_stdev_samples",
                "gauss_stdev_ms",
                "gauss_fwhm_samples",
                "gauss_fwhm_ms",

                # FWHM-based boundary indices (inner width around peak)
                "fwhm_le_idx",
                "fwhm_ri_idx",
                "fwhm_le_ms",
                "fwhm_ri_ms",
                "fwhm_global_le_idx",
                "fwhm_global_ri_idx",

                # Amplitudes at key points
                "center_voltage",
                "le_voltage",
                "ri_voltage",

                # Duration / symmetry / sharpness
                "duration_ms",
                "rise_ms",
                "decay_ms",
                "rdsm",
                "sharpness",

                # Slope features
                "max_upslope_mv_per_s",
                "max_downslope_mv_per_s",
                "slope_asymmetry",

                # Area under the wave
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
                # QTc (rate-corrected QT) calculations
                "QTc_Bazett_ms",
                "QTc_Fridericia_ms",
                "QTc_Framingham_ms",
            ]
            
            pairwise_diff_keys = [
                "R_minus_S_voltage_diff_signed",
                "R_minus_P_voltage_diff_signed",
                "T_minus_R_voltage_diff_signed",
            ]
            
            self.output_dict = self.initialize_output_dict(
                cycle_inds=np.arange(len(cycles)),
                components=component_keys,
                peak_features=peak_feature_keys,
                intervals=interval_keys,
                pairwise_differences=pairwise_diff_keys,
            )
            
            for cycle_idx, cycle_label in enumerate(cycles):
                one_cycle = epochs_df.loc[epochs_df["cycle"] == cycle_label]
                
                # Always log cycle processing start for cycles 50-60 to debug missing cycles 51-59
                if 50 <= cycle_idx <= 60:
                    logging.info(f"[FIT_CYCLE_START] Cycle {cycle_idx} (label {cycle_label}, type={type(cycle_label).__name__}): Starting processing, one_cycle len={len(one_cycle)}")
                    if len(one_cycle) == 0:
                        logging.warning(f"[FIT_EMPTY_CYCLE] Cycle {cycle_idx} (label {cycle_label}): one_cycle is EMPTY! Checking epochs_df cycle column...")
                        # Debug: Check what cycle values exist in epochs_df around this label
                        unique_cycles = epochs_df["cycle"].unique()
                        logging.warning(f"[FIT_EMPTY_CYCLE] Available cycle labels in epochs_df (sample): {sorted(unique_cycles)[50:61] if len(unique_cycles) > 60 else sorted(unique_cycles)[50:]}")
                        logging.warning(f"[FIT_EMPTY_CYCLE] Cycle label type in epochs_df: {type(epochs_df['cycle'].iloc[0]).__name__ if len(epochs_df) > 0 else 'N/A'}")
                
                # Always log cycle processing start (sample every 10 cycles to avoid spam, but always log errors)
                elif cycle_idx % 10 == 0 or cycle_idx < 20:
                    logging.info(f"[FIT_CYCLE_START] Cycle {cycle_idx} (label {cycle_label}): Starting processing, one_cycle len={len(one_cycle)}")
                
                # Debug: Track cycle processing (first 3 cycles only to avoid spam)
                if cycle_idx < 3:
                    logging.debug(f"[fit.py] Processing cycle {cycle_idx} (label {cycle_label}), verbose={self.verbose}")
                    if len(one_cycle) == 0:
                        logging.warning(f"[fit.py] Cycle {cycle_idx} is empty!")
                
                # Always log if cycle is empty (critical issue)
                if len(one_cycle) == 0:
                    logging.warning(f"[FIT_EMPTY_CYCLE] Cycle {cycle_idx} (label {cycle_label}): one_cycle is EMPTY! Skipping.")
                    continue
                
                try:
                    self.process_cycle_wrapper(
                        one_cycle, cycle_idx, 
                        precomputed_peaks=precomputed_peaks, 
                        full_derivative=full_derivative,
                        p_training_signal_peak=p_training_signal_peak,
                        p_training_noise_peak=p_training_noise_peak,
                    )
                    
                    # Log completion (sample every 10 cycles)
                    if cycle_idx % 10 == 0 or cycle_idx < 20:
                        if self.output_dict is not None:
                            r_val = self.output_dict.get("R_global_center_idx", [None])[cycle_idx] if cycle_idx < len(self.output_dict.get("R_global_center_idx", [])) else None
                            logging.info(f"[FIT_CYCLE_END] Cycle {cycle_idx} (label {cycle_label}): Completed processing, R={r_val}")
                    
                    # Debug: Check if peaks were stored after processing (first 3 cycles)
                    if cycle_idx < 3 and self.output_dict is not None:
                        r_val = self.output_dict.get("R_global_center_idx", [None])[cycle_idx] if cycle_idx < len(self.output_dict.get("R_global_center_idx", [])) else None
                        p_val = self.output_dict.get("P_global_center_idx", [None])[cycle_idx] if cycle_idx < len(self.output_dict.get("P_global_center_idx", [])) else None
                        logging.debug(f"[fit.py] After cycle {cycle_idx}: R={r_val}, P={p_val}")
                        
                except Exception as e:
                    # Always log errors, regardless of verbose setting
                    logging.error(f"[CYCLE_ERROR] Error processing cycle {cycle_idx} (label {cycle_label}): {e}")
                    import traceback
                    logging.error(f"[CYCLE_ERROR] Cycle {cycle_idx} traceback:\n{traceback.format_exc()}")
                    # Continue processing other cycles even if one fails
                    continue

            # Debug: Check R peaks in output_dict before DataFrame conversion
            if "R_global_center_idx" in self.output_dict:
                r_dict_values = np.array(self.output_dict["R_global_center_idx"])
                r_non_null = np.sum(~pd.isna(r_dict_values))
                logging.info(f"[FIT_DEBUG] Before DataFrame conversion: R_global_center_idx - total={len(r_dict_values)}, non-null={r_non_null}")
                if r_non_null > 0:
                    r_non_null_indices = np.where(~pd.isna(r_dict_values))[0]
                    logging.info(f"[FIT_DEBUG] Cycles with R peaks: {r_non_null_indices[:20]}... (first 20 of {len(r_non_null_indices)})")
            
            self.output_df = pd.DataFrame.from_dict(self.output_dict, orient="columns")
            self.output_df.index.name = "cycle_index"
            
            # Debug: Check R peaks after DataFrame conversion
            if "R_global_center_idx" in self.output_df.columns:
                r_df_non_null = self.output_df["R_global_center_idx"].notna().sum()
                logging.info(f"[FIT_DEBUG] After DataFrame conversion: R_global_center_idx - non-null in df={r_df_non_null}, total rows={len(self.output_df)}")
            
            # Debug: Check P values after DataFrame conversion
            if self.verbose and "P_global_center_idx" in self.output_dict:
                p_dict_values = self.output_dict["P_global_center_idx"][:min(5, len(self.output_dict["P_global_center_idx"]))]
                p_df_values = self.output_df["P_global_center_idx"].head(5).values if "P_global_center_idx" in self.output_df.columns else None
                print(f"[DEBUG] After DataFrame conversion: dict values={p_dict_values}, df values={p_df_values}")
            
            # Compute beat-to-beat variability metrics
            if len(self.output_df) > 0:
                try:
                    self.compute_variability_metrics()
                except Exception as e:
                    logging.warning(f"Error computing variability metrics: {e}")
                    self.variability_metrics = {}
    
            return self.output_df, self.epochs_df #output_df, epochs_df = analyzer.analyze_ecg(signal) return both for accessible unpacking

        
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
        - pNN50: percentage of successive differences >50 ms
        - SD1: Short-term HRV from Poincaré plot (perpendicular to line of identity)
        - SD2: Long-term HRV from Poincaré plot (along line of identity)
    
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

            average_heart_rate, sdnn, rmssd, nn50, pnn50, sd1, sd2 = calc_hrv_metrics(clean_rr_intervals)
            self.hrv_metrics = {
                "average_heart_rate": average_heart_rate,
                "sdnn": sdnn,
                "rmssd": rmssd,
                "nn50": nn50,
                "pnn50": pnn50,
                "sd1": sd1,
                "sd2": sd2,
            }

        except ValueError as ve:
            logging.warning(f"Validation Error in compute_hrv_metrics: {ve}")
            self.hrv_metrics = {}
        except Exception as e:
            logging.error(f"Unexpected error in compute_hrv_metrics: {e}")
            self.hrv_metrics = {}
    
    def compute_variability_metrics(self, priority_features: Optional[List[str]] = None):
        """
        Compute beat-to-beat variability metrics for key morphological features.
    
        Computes variability statistics (std, CV, IQR, MAD, range) across cycles for
        priority features such as QT intervals, QRS duration, wave amplitudes, etc.
    
        Parameters
        ----------
        priority_features : List[str], optional
            List of feature names to compute variability for.
            If None, uses default priority features (QT, QRS, PR, RR intervals,
            QTc values, wave amplitudes, etc.)
    
        Returns
        -------
        None
            Updates `self.variability_metrics` with computed values, or an empty dict
            if computation fails.
        """
        try:
            if self.output_dict is None:
                raise ValueError("output_dict is missing. Cannot compute variability metrics.")
            
            self.variability_metrics = compute_beat_to_beat_variability(
                self.output_dict,
                priority_features=priority_features
            )
            
            if self.verbose and len(self.variability_metrics) > 0:
                logging.info(f"Computed variability metrics for {len(self.variability_metrics) // 5} features")
        
        except Exception as e:
            logging.error(f"Error computing variability metrics: {e}")
            self.variability_metrics = {}
    
    def save_variability_metrics(self, file_id: str, results_dir: str):
        """
        Save computed variability metrics to a CSV file.
    
        Parameters
        ----------
        file_id : str
            Identifier for the ECG recording (used in filename).
        results_dir : str
            Directory where the CSV will be saved.
    
        Returns
        -------
        None
            Writes `{file_id}_variability_metrics.csv` to the results directory.
        """
        try:
            if not self.variability_metrics:
                logging.info(f"Variability metrics are empty for file {file_id}. Nothing to save.")
                return

            variability_df = pd.DataFrame([self.variability_metrics])
            output_path = f"{results_dir}/{file_id}_variability_metrics.csv"
            variability_df.to_csv(output_path, index=False)
            self._save_metadata(file_id, results_dir)
            logging.info(f"Variability metrics for {file_id} saved to {output_path}.")

        except Exception as e:
            logging.error(f"Unexpected error in save_variability_metrics for {file_id}: {e}")

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
