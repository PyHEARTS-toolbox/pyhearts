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


def _load_record_routing_table(path: str) -> dict[str, str]:
    """
    Load a per-record routing table mapping record -> route.

    Expected schema:
      { "records": [ { "record": "<name>", "route": "stpq_overwrite" | "per_cycle_t" }, ... ] }
    """
    p = Path(path)
    if not p.is_absolute():
        # Resolve relative to repo root (pyhearts/pyhearts/core/fit.py -> repo/)
        p = Path(__file__).resolve().parents[2] / p
    data = json.loads(p.read_text())
    rows = data.get("records", [])
    out: dict[str, str] = {}
    for r in rows:
        rec = r.get("record")
        route = r.get("route")
        if isinstance(rec, str) and isinstance(route, str):
            out[rec] = route
    return out


def _is_per_cycle_t_route(cfg, meta: Optional[dict], cache: dict) -> bool:
    """True when routing table sends this record to per-cycle T (skip STPQ post-passes)."""
    try:
        table_path = getattr(cfg, "record_stpq_routing_table", None)
        record_name = (meta or {}).get("record")
        if not table_path or not isinstance(record_name, str) or not record_name:
            return False
        if table_path not in cache:
            cache[table_path] = _load_record_routing_table(str(table_path))
        return cache[table_path].get(record_name) == "per_cycle_t"
    except Exception as e:
        logging.warning("Record routing table lookup failed: %s", e)
        return False


def _skip_record_stpq_postpass(cfg, meta: Optional[dict], cache: dict) -> bool:
    """Skip record STPQ overwrite passes (per-cycle T route or template-prior phase 1)."""
    if getattr(cfg, "record_template_prior_windows", False):
        return True
    return _is_per_cycle_t_route(cfg, meta, cache)


def _skip_rt_plausibility_gate(cfg) -> bool:
    """
    Template-prior presets use per-cycle + rescue plausibility; record-level RT MAD
    gating after rescue can null beats that rescue kept (median shifts when other
    beats are updated).
    """
    return bool(getattr(cfg, "record_template_prior_windows", False))


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
    
    species : {"human", "mouse"}, optional
        ``"human"`` → :meth:`~pyhearts.config.ProcessCycleConfig.for_human_unified`
        (v3.2.1 production). ``"mouse"`` → :meth:`~pyhearts.config.ProcessCycleConfig.for_mouse`.
        Pass ``cfg=`` explicitly to override (e.g. ``ProcessCycleConfig.for_mouse()``).

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
                base = ProcessCycleConfig.for_human_unified()
            else:
                base = ProcessCycleConfig()

        # 2) apply field-level overrides (ProcessCycleConfig fields only)
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
        self._wfdb_lead_index: Optional[int] = None
        self._wfdb_lead_name: Optional[str] = None
        self._lead_policy: Optional[str] = None
        self._manual_ann_ext: Optional[str] = None
        self._run_metadata: Optional[dict] = None
        self._candidate_ranker_model = None

    def set_run_metadata(
        self,
        *,
        record: Optional[str] = None,
        lead_index: Optional[int] = None,
        lead_name: Optional[str] = None,
        lead_policy: Optional[str] = None,
        manual_ann_ext: Optional[str] = None,
    ) -> None:
        """Attach WFDB lead / annotation metadata for the next :meth:`analyze_ecg` run."""
        self._wfdb_lead_index = lead_index
        self._wfdb_lead_name = lead_name
        self._lead_policy = lead_policy
        self._manual_ann_ext = manual_ann_ext
        self._run_metadata = {
            k: v
            for k, v in (
                ("record", record),
                ("lead_index", lead_index),
                ("lead_name", lead_name),
                ("lead_policy", lead_policy),
                ("manual_ann_ext", manual_ann_ext),
            )
            if v is not None
        }

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

    def process_cycle_wrapper(
        self,
        one_cycle: pd.DataFrame,
        cycle_idx: int,
        cycle_epoch_idx: int | None = None,
        precomputed_peaks: dict | None = None,
        full_derivative: np.ndarray | None = None,
        p_training_signal_peak: float | None = None,
        p_training_noise_peak: float | None = None,
        template_prior_windows=None,
    ):
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
        cycle_epoch_idx : int, optional
            Epoch label from ``epochs_df['cycle']`` (index into ``r_peak_indices``).
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
        r_peak_global_idx = None
        if hasattr(self, "r_peak_indices") and self.r_peak_indices is not None and cycle_epoch_idx is not None:
            epoch_i = int(cycle_epoch_idx)
            if 0 <= epoch_i < len(self.r_peak_indices):
                r_peak_global_idx = int(self.r_peak_indices[epoch_i])

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
            r_peak_global_idx=r_peak_global_idx,
            cycle_epoch_idx=cycle_epoch_idx,
            full_derivative=full_derivative,
            p_training_signal_peak=p_training_signal_peak,
            p_training_noise_peak=p_training_noise_peak,
            full_delineation_ecg=getattr(self, "_full_delineation_ecg", None),
            template_prior_windows=template_prior_windows,
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

    def _run_r_peak_detection(
        self,
        ecg_signal: np.ndarray,
        raw_ecg: Optional[np.ndarray],
    ) -> np.ndarray:
        """R-peak train for epoching (unified derivative + Phase A pipeline)."""
        trp_signal = raw_ecg if raw_ecg is not None else ecg_signal
        return r_peak_detection(
            ecg_signal,
            self.sampling_rate,
            cfg=self.cfg,
            plot=self.plot,
            sensitivity=self.sensitivity,
            raw_ecg=trp_signal,
        )

    def _lock_r_timing_to_detection(
        self,
        epochs_df: pd.DataFrame,
        cycles: np.ndarray,
    ) -> None:
        """Overwrite per-cycle R timing fiducials with detection indices (Tier A)."""
        if self.output_dict is None or self.r_peak_indices is None:
            return
        from pyhearts.processing.peaks import (
            cycle_rel_to_global_sample,
            global_index_to_cycle_relative,
            refine_r_peak_near_anchor,
        )

        def _finite(val) -> bool:
            return val is not None and not (
                isinstance(val, float) and np.isnan(val)
            )

        r_list = self.output_dict.get("R_global_center_idx", [])
        for cycle_idx, cycle_label in enumerate(cycles):
            if cycle_idx >= len(r_list):
                break
            epoch_i = int(cycle_label)
            if epoch_i < 0 or epoch_i >= len(self.r_peak_indices):
                continue
            r_det = int(self.r_peak_indices[epoch_i])
            one_cycle = epochs_df.loc[epochs_df["cycle"] == cycle_label].sort_values("index")
            if one_cycle.empty:
                continue
            if "index" in one_cycle.columns:
                xs = one_cycle["index"].values.astype(int)
            else:
                xs = one_cycle["signal_x"].values.astype(int)
            sig = one_cycle["signal_y"].values.astype(float)
            r_rel = global_index_to_cycle_relative(r_det, xs)
            if r_rel is None:
                continue
            r_rel_refined = refine_r_peak_near_anchor(
                sig,
                r_rel,
                self.sampling_rate,
                half_window_ms=self.cfg.r_anchor_refine_half_window_ms,
                refine_mode=self.cfg.r_anchor_refine_mode,
            )
            r_global = cycle_rel_to_global_sample(
                r_rel_refined,
                xs,
                sig,
                refine_subsample=self.cfg.use_subsample_peak_refinement,
            )
            old_r = r_list[cycle_idx]
            r_list[cycle_idx] = r_global
            self.output_dict["R_center_idx"][cycle_idx] = float(r_rel_refined)
            if _finite(old_r):
                delta = float(r_global) - float(old_r)
                if abs(delta) > 1e-9:
                    for wave in ("P", "T"):
                        gkey = f"{wave}_global_center_idx"
                        garr = self.output_dict.get(gkey)
                        if garr is None or cycle_idx >= len(garr):
                            continue
                        if _finite(garr[cycle_idx]):
                            garr[cycle_idx] = float(garr[cycle_idx]) + delta

    def analyze_ecg(
        self,
        ecg_signal: np.ndarray,
        verbose: Optional[bool] = None,
        plot: Optional[bool] = None,
        raw_ecg: Optional[np.ndarray] = None,
        *,
        run_metadata: Optional[dict] = None,
        signal_crop_fraction: Optional[float] = None,
        signal_crop_duration_s: Optional[float] = None,
        signal_crop_from_end: bool = False,
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
            Clinical (unfiltered) ECG for TRP/kurtosis and polarity. Required for best
            results with Rahul presets when ``ecg_signal`` is high-pass filtered.
            If None, clinical trace defaults to ``ecg_signal`` when record
            delineation uses median-record baseline removal.
        run_metadata : dict, optional
            Per-run provenance merged with :meth:`set_run_metadata` (e.g.
            ``lead_index``, ``lead_name``, ``lead_policy``, ``manual_ann_ext``).
        signal_crop_fraction : float, optional
            When set, analyze only this fraction of the recording (see *signal_crop_from_end*).
        signal_crop_duration_s : float, optional
            When set, analyze only this many seconds; overrides *signal_crop_fraction*.
        signal_crop_from_end : bool
            If False, use the first segment; if True, use the last segment.
    
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
            meta = dict(self._run_metadata or {})
            if run_metadata:
                meta.update(run_metadata)
            ecg_signal = np.asarray(ecg_signal, dtype=float)
            if raw_ecg is not None:
                raw_ecg = np.asarray(raw_ecg, dtype=float)
            elif (
                self.cfg.record_delineation
                and self.cfg.delineation_baseline_method == "median_record"
            ):
                raw_ecg = ecg_signal.copy()
            delim_ecg = raw_ecg if raw_ecg is not None else ecg_signal

            if signal_crop_fraction is not None or signal_crop_duration_s is not None:
                from pyhearts.processing.signal_crop import crop_signal

                ecg_signal, crop_start, crop_end = crop_signal(
                    ecg_signal,
                    self.sampling_rate,
                    fraction=signal_crop_fraction,
                    duration_s=signal_crop_duration_s,
                    from_end=signal_crop_from_end,
                )
                if raw_ecg is not None:
                    raw_ecg = raw_ecg[crop_start:crop_end]

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
            
            # R-peak detection
            #
            # BREAKING CHANGE NOTE:
            # Legacy alternate detectors (pan_tompkins, bandpass_energy) were removed.
            # Use the unified `pyhearts.processing.r_peak_detection`.
            filtered_r_peaks = self._run_r_peak_detection(ecg_signal, raw_ecg)
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

            from pyhearts.processing.delineation_signal import prepare_record_delineation_signal

            self._full_delineation_ecg = prepare_record_delineation_signal(
                delim_ecg,
                self.sampling_rate,
                self.cfg,
            )

            # Use the actual cycle labels from epochs_df (sorted for determinism)
            cycles = np.sort(epochs_df["cycle"].unique())

            # Compute P wave training phase thresholds (adaptive signal/noise separation)
            # Analyzes first 1-3 seconds to learn P wave signal vs noise characteristics
            p_training_signal_peak = None
            p_training_noise_peak = None
            from pyhearts.processing.pt_detection_mode import p_t_detection_is_record_only

            record_only_pt = p_t_detection_is_record_only(self.cfg)
            if len(cycles) > 0 and not self.cfg.lite_mode and not record_only_pt:
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
            if len(cycles) > 0 and not self.cfg.lite_mode and not record_only_pt:
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
                # ST segment features
                "ST_elevation_mv",
                "ST_slope_mv_per_s",
                "ST_deviation_mv",
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

            from pyhearts.processing.fiducial_provenance import (
                init_fiducial_provenance,
                set_run_lead_metadata,
            )
            from pyhearts.processing.t_timing_audit import init_t_timing_audit

            init_fiducial_provenance(self.output_dict, len(cycles))
            if self.cfg.record_delineation_t_timing_audit:
                init_t_timing_audit(self.output_dict, len(cycles))
            if meta.get("lead_index") is not None and meta.get("lead_name"):
                set_run_lead_metadata(
                    self.output_dict,
                    lead_index=int(meta["lead_index"]),
                    lead_name=str(meta["lead_name"]),
                    lead_policy=str(meta.get("lead_policy", "")),
                )

            if not hasattr(self, "_record_routing_cache"):
                self._record_routing_cache = {}
            per_cycle_t_route = _is_per_cycle_t_route(
                self.cfg, meta, self._record_routing_cache
            )
            skip_stpq_postpass = _skip_record_stpq_postpass(
                self.cfg, meta, self._record_routing_cache
            )
            if per_cycle_t_route and self.verbose:
                logging.info(
                    "Record routing: %s -> per_cycle_t "
                    "(skip STPQ delineation, RT gate, fill, smoothing)",
                    meta.get("record"),
                )
            if (
                self.cfg.record_template_prior_windows
                and not per_cycle_t_route
                and self.verbose
            ):
                logging.info(
                    "Record %s: template-prior windows (per-cycle T inside STPQ w1)",
                    meta.get("record"),
                )

            template_prior_by_cycle: dict = {}
            self._record_template_for_prior = None
            self._cluster_by_epoch: dict = {}
            self._cluster_templates: dict = {}
            if len(cycles) > 0 and self.cfg.record_template_prior_windows:
                from pyhearts.processing.template_prior_windows import (
                    compute_template_prior_windows,
                )

                clinical = raw_ecg if raw_ecg is not None else delim_ecg
                cluster_by_epoch: dict = {}
                cluster_templates: dict = {}
                cluster_k = int(
                    getattr(self.cfg, "record_template_prior_cluster_k", 0) or 0
                )
                if cluster_k > 1:
                    from pyhearts.processing.record_beat_clustering import (
                        build_cluster_templates,
                        cluster_epoch_indices,
                        extract_stpq_segments,
                        stpq_clusters_are_heterogeneous,
                    )

                    segments = extract_stpq_segments(
                        clinical,
                        self.r_peak_indices,
                        self.sampling_rate,
                        self.cfg,
                    )
                    if stpq_clusters_are_heterogeneous(segments, cluster_k):
                        cluster_by_epoch = cluster_epoch_indices(
                            segments, cluster_k, seed=0
                        )
                        cluster_templates = build_cluster_templates(
                            clinical,
                            self.r_peak_indices,
                            self.sampling_rate,
                            self.cfg,
                            cluster_by_epoch,
                            manual_ann_ext=self._manual_ann_ext,
                        )
                        self._cluster_by_epoch = cluster_by_epoch
                        self._cluster_templates = cluster_templates

                _tmpl, template_prior_by_cycle = compute_template_prior_windows(
                    clinical,
                    self.r_peak_indices,
                    list(cycles),
                    self.sampling_rate,
                    self.cfg,
                    manual_ann_ext=self._manual_ann_ext,
                    cluster_templates=cluster_templates or None,
                    cluster_by_epoch=cluster_by_epoch or None,
                )
                self._record_template_for_prior = _tmpl
                self.last_template_prior_stats = {
                    "template_valid": bool(_tmpl is not None and _tmpl.valid),
                    "n_cycles_with_windows": len(template_prior_by_cycle),
                    "n_cycles": len(cycles),
                    "n_clusters": len(cluster_templates),
                }
                if self.verbose:
                    logging.info(
                        "Template-prior windows: %s",
                        self.last_template_prior_stats,
                    )

            precomputed_peaks = None
            if (
                len(cycles) > 0
                and self.cfg.record_delineation
                and self.cfg.record_delineation_before_cycles
                and not per_cycle_t_route
            ):
                from pyhearts.processing.record_fiducial_seed import (
                    seed_record_fiducials_before_cycles,
                )

                clinical = raw_ecg if raw_ecg is not None else delim_ecg
                precomputed_peaks, seed_stats = seed_record_fiducials_before_cycles(
                    self.output_dict,
                    delim_ecg,
                    self.r_peak_indices,
                    epochs_df,
                    cycles,
                    self.sampling_rate,
                    self.cfg,
                    clinical_ecg=clinical,
                    expected_max_energy=getattr(self, "expected_max_energy", 0.0),
                    verbose=self.verbose,
                )
                self.last_record_fiducial_seed_stats = seed_stats
                if self.verbose and seed_stats.get("template_valid"):
                    logging.info("Record fiducial seed (before cycles): %s", seed_stats)
            
            for cycle_idx, cycle_label in enumerate(cycles):
                one_cycle = epochs_df.loc[epochs_df["cycle"] == cycle_label]

                if len(one_cycle) == 0:
                    logging.debug(
                        "[FIT_EMPTY_CYCLE] Cycle %s (label %s): empty epoch; skipping.",
                        cycle_idx,
                        cycle_label,
                    )
                    continue
                
                try:
                    self.process_cycle_wrapper(
                        one_cycle,
                        cycle_idx,
                        cycle_epoch_idx=int(cycle_label),
                        precomputed_peaks=precomputed_peaks,
                        full_derivative=full_derivative,
                        p_training_signal_peak=p_training_signal_peak,
                        p_training_noise_peak=p_training_noise_peak,
                        template_prior_windows=template_prior_by_cycle.get(cycle_idx),
                    )

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

            if len(cycles) > 0 and self.cfg.lock_r_global_to_detection:
                self._lock_r_timing_to_detection(epochs_df, cycles)

            if (
                len(cycles) > 0
                and self.cfg.record_template_prior_learned_ranker
                and template_prior_by_cycle
            ):
                from pyhearts.processing.t_candidate_ranker import (
                    apply_learned_ranker_pass,
                )

                clinical = raw_ecg if raw_ecg is not None else delim_ecg
                ranker_stats = apply_learned_ranker_pass(
                    self.output_dict,
                    clinical,
                    self.r_peak_indices,
                    list(cycles),
                    self.sampling_rate,
                    self.cfg,
                    template_prior_by_cycle,
                    getattr(self, "_record_template_for_prior", None),
                    ranker_model=getattr(self, "_candidate_ranker_model", None),
                )
                self.last_learned_ranker_stats = ranker_stats
                if self.verbose and ranker_stats.get("ranked", 0):
                    logging.info("Learned candidate ranker: %s", ranker_stats)

            if (
                len(cycles) > 0
                and self.cfg.record_template_prior_rescue
                and template_prior_by_cycle
                and not self.cfg.record_template_prior_learned_ranker
            ):
                from pyhearts.processing.template_prior_rescue import (
                    apply_template_prior_t_rescue_pass,
                )

                clinical = raw_ecg if raw_ecg is not None else delim_ecg
                rescue_stats, _ = apply_template_prior_t_rescue_pass(
                    self.output_dict,
                    clinical,
                    self.r_peak_indices,
                    list(cycles),
                    self.sampling_rate,
                    self.cfg,
                    template_prior_by_cycle,
                    getattr(self, "_record_template_for_prior", None),
                    cluster_templates=getattr(self, "_cluster_templates", None)
                    or None,
                    cluster_by_epoch=getattr(self, "_cluster_by_epoch", None) or None,
                )
                self.last_template_prior_rescue_stats = rescue_stats
                if self.verbose and rescue_stats.get("rescued", 0):
                    logging.info("Template-prior T rescue: %s", rescue_stats)

            if (
                len(cycles) > 0
                and getattr(self.cfg, "record_inverted_dzc_rescue", False)
                and template_prior_by_cycle
            ):
                from pyhearts.processing.t_inverted_dzc_rescue import (
                    apply_inverted_dzc_rescue_pass,
                )

                clinical = raw_ecg if raw_ecg is not None else delim_ecg
                inv_stats = apply_inverted_dzc_rescue_pass(
                    self.output_dict,
                    clinical,
                    self.r_peak_indices,
                    list(cycles),
                    self.sampling_rate,
                    self.cfg,
                    template_prior_by_cycle,
                    getattr(self, "_record_template_for_prior", None),
                )
                self.last_inverted_dzc_rescue_stats = inv_stats
                if self.verbose and inv_stats.get("rescued", 0):
                    logging.info("Inverted DZC rescue: %s", inv_stats)

            if (
                len(cycles) > 0
                and not per_cycle_t_route
                and not record_only_pt
                and (self.cfg.t_wave_use_record_prior or self.cfg.t_wave_use_secondary_detector)
            ):
                try:
                    from pyhearts.processing.t_wave_fusion import recover_missing_t_waves

                    t_rec_stats = recover_missing_t_waves(
                        self.output_dict,
                        epochs_df,
                        cycles,
                        self.sampling_rate,
                        self.cfg,
                        verbose=self.verbose,
                    )
                    if self.verbose and t_rec_stats.get("recovered", 0) > 0:
                        logging.info(
                            "T-wave recovery pass: %s",
                            t_rec_stats,
                        )
                except Exception as e:
                    logging.warning("T-wave recovery pass failed: %s", e)

            if (
                len(cycles) > 0
                and self.cfg.t_rt_plausibility_gate
                and not per_cycle_t_route
                and not record_only_pt
                and not self.cfg.record_delineation
                and not _skip_rt_plausibility_gate(self.cfg)
            ):
                try:
                    from pyhearts.processing.t_plausibility import apply_rt_plausibility_gate

                    rt_stats = apply_rt_plausibility_gate(
                        self.output_dict,
                        self.sampling_rate,
                        self.cfg,
                        verbose=self.verbose,
                    )
                    if self.verbose and (
                        rt_stats.get("rejected_bounds", 0)
                        or rt_stats.get("rejected_outlier", 0)
                    ):
                        logging.info("RT plausibility gate (pre-record): %s", rt_stats)
                except Exception as e:
                    logging.warning("RT plausibility gate failed: %s", e)

            if (
                len(cycles) > 0
                and self.cfg.record_delineation
                and not skip_stpq_postpass
                and not self.cfg.record_delineation_before_cycles
            ):
                try:
                    from pyhearts.processing.record_delineation import (
                        apply_record_level_delineation,
                    )

                    clinical = raw_ecg if raw_ecg is not None else delim_ecg
                    delim_stats = apply_record_level_delineation(
                        self.output_dict,
                        delim_ecg,
                        self.r_peak_indices,
                        epochs_df,
                        cycles,
                        self.sampling_rate,
                        self.cfg,
                        clinical_ecg=clinical,
                        expected_max_energy=getattr(
                            self, "expected_max_energy", 0.0
                        ),
                        verbose=self.verbose,
                        manual_ann_ext=self._manual_ann_ext,
                    )
                    self.last_record_delineation_stats = delim_stats
                    if self.verbose and delim_stats.get("template_valid", 0):
                        logging.info("Record-level delineation (B1): %s", delim_stats)
                except Exception as e:
                    logging.warning("Record-level delineation failed: %s", e)

            clinical_verify_after_fill = (
                self.cfg.record_clinical_verify
                and self.cfg.record_delineation_fill_missing_t
            )

            if (
                len(cycles) > 0
                and self.cfg.record_clinical_verify
                and not per_cycle_t_route
            ):
                if not clinical_verify_after_fill:
                    try:
                        from pyhearts.processing.clinical_fiducial_verify import (
                            apply_clinical_fiducial_verification,
                        )

                        clinical = raw_ecg if raw_ecg is not None else delim_ecg
                        verify_stats = apply_clinical_fiducial_verification(
                            self.output_dict,
                            epochs_df,
                            cycles,
                            clinical,
                            self.r_peak_indices,
                            self.sampling_rate,
                            self.cfg,
                            verbose=self.verbose,
                        )
                        if self.verbose and verify_stats.get("p_checked", 0):
                            logging.info("Clinical fiducial verify: %s", verify_stats)
                    except Exception as e:
                        logging.warning("Clinical fiducial verify failed: %s", e)

            if len(cycles) > 0 and self.cfg.t_rt_plausibility_gate:
                if (
                    not per_cycle_t_route
                    and (record_only_pt or self.cfg.record_delineation)
                    and not _skip_rt_plausibility_gate(self.cfg)
                ):
                    try:
                        from pyhearts.processing.t_plausibility import (
                            apply_rt_plausibility_gate,
                        )

                        rt_stats = apply_rt_plausibility_gate(
                            self.output_dict,
                            self.sampling_rate,
                            self.cfg,
                            verbose=self.verbose,
                        )
                        if self.verbose and (
                            rt_stats.get("rejected_bounds", 0)
                            or rt_stats.get("rejected_outlier", 0)
                        ):
                            logging.info("RT plausibility gate (post-delineation): %s", rt_stats)
                    except Exception as e:
                        logging.warning("RT plausibility gate failed: %s", e)

            if (
                len(cycles) > 0
                and self.cfg.record_delineation
                and self.cfg.record_delineation_fill_missing_t
                and not per_cycle_t_route
            ):
                try:
                    from pyhearts.processing.record_delineation import (
                        apply_record_fill_missing_t,
                    )

                    clinical = raw_ecg if raw_ecg is not None else delim_ecg
                    fill_stats = apply_record_fill_missing_t(
                        self.output_dict,
                        delim_ecg,
                        self.r_peak_indices,
                        epochs_df,
                        cycles,
                        self.sampling_rate,
                        self.cfg,
                        clinical_ecg=clinical,
                        expected_max_energy=getattr(
                            self, "expected_max_energy", 0.0
                        ),
                        verbose=self.verbose,
                    )
                    if self.verbose and fill_stats.get("t_fill_missing", 0):
                        logging.info(
                            "Record fill missing T (post-RT gate): %s", fill_stats
                        )
                except Exception as e:
                    logging.warning("Record fill missing T failed: %s", e)

            if (
                len(cycles) > 0
                and clinical_verify_after_fill
                and not per_cycle_t_route
            ):
                try:
                    from pyhearts.processing.clinical_fiducial_verify import (
                        apply_clinical_fiducial_verification,
                    )

                    clinical = raw_ecg if raw_ecg is not None else delim_ecg
                    verify_stats = apply_clinical_fiducial_verification(
                        self.output_dict,
                        epochs_df,
                        cycles,
                        clinical,
                        self.r_peak_indices,
                        self.sampling_rate,
                        self.cfg,
                        verbose=self.verbose,
                    )
                    if self.verbose and (
                        verify_stats.get("p_checked", 0)
                        or verify_stats.get("t_checked", 0)
                    ):
                        logging.info(
                            "Clinical fiducial verify (post-fill): %s", verify_stats
                        )
                except Exception as e:
                    logging.warning("Clinical fiducial verify failed: %s", e)

            if (
                len(cycles) > 0
                and self.cfg.record_fiducial_smoothing
                and not per_cycle_t_route
            ):
                try:
                    from pyhearts.processing.record_fiducial_smoothing import (
                        apply_record_fiducial_smoothing,
                    )

                    delim_stats = getattr(self, "last_record_delineation_stats", None) or {}
                    seed_stats = getattr(self, "last_record_fiducial_seed_stats", None) or {}
                    t_morph = delim_stats.get("t_morphology") or seed_stats.get(
                        "t_morphology"
                    )
                    smooth_stats = apply_record_fiducial_smoothing(
                        self.output_dict,
                        epochs_df,
                        cycles,
                        self.sampling_rate,
                        self.cfg,
                        record_t_morphology=t_morph,
                        verbose=self.verbose,
                    )
                    if self.verbose and (
                        smooth_stats.get("p_adjusted", 0) > 0
                        or smooth_stats.get("t_adjusted", 0) > 0
                    ):
                        logging.info("Record fiducial smoothing: %s", smooth_stats)
                except Exception as e:
                    logging.warning("Record fiducial smoothing failed: %s", e)

            if (
                len(cycles) > 0
                and self.cfg.record_biphasic_pm_early_t_guardrail
                and not skip_stpq_postpass
            ):
                try:
                    from pyhearts.processing.record_delineation import (
                        build_record_beat_template,
                        delineate_record_template,
                    )
                    from pyhearts.processing.record_template_biphasic import (
                        apply_biphasic_pm_early_t_guardrail,
                    )

                    clinical = raw_ecg if raw_ecg is not None else delim_ecg
                    raw_tmpl = build_record_beat_template(
                        clinical,
                        self.r_peak_indices,
                        self.sampling_rate,
                        self.cfg,
                    )
                    tmpl = delineate_record_template(
                        raw_tmpl,
                        self.sampling_rate,
                        self.cfg,
                        manual_ann_ext=self._manual_ann_ext,
                    )
                    guard_stats = apply_biphasic_pm_early_t_guardrail(
                        self.output_dict,
                        epochs_df,
                        cycles,
                        self.r_peak_indices,
                        delim_ecg,
                        tmpl,
                        self.sampling_rate,
                        self.cfg,
                    )
                    if self.verbose and guard_stats.get("adjusted", 0) > 0:
                        logging.info(
                            "Biphasic +− early-T guardrail (post-smoothing): %s",
                            guard_stats,
                        )
                except Exception as e:
                    logging.warning("Biphasic +− early-T guardrail failed: %s", e)

            if self.verbose and "R_global_center_idx" in self.output_dict:
                r_dict_values = np.array(self.output_dict["R_global_center_idx"])
                r_non_null = int(np.sum(~pd.isna(r_dict_values)))
                logging.debug(
                    "[FIT_DEBUG] Before DataFrame conversion: R non-null=%s / %s",
                    r_non_null,
                    len(r_dict_values),
                )

            self.output_df = pd.DataFrame.from_dict(self.output_dict, orient="columns")
            self.output_df.index.name = "cycle_index"

            from pyhearts.processing.fiducial_provenance import attach_provenance_to_dataframe

            attach_provenance_to_dataframe(self.output_df, meta)

            if self.verbose and "R_global_center_idx" in self.output_df.columns:
                r_df_non_null = int(self.output_df["R_global_center_idx"].notna().sum())
                logging.debug(
                    "[FIT_DEBUG] After DataFrame conversion: R non-null=%s / %s rows",
                    r_df_non_null,
                    len(self.output_df),
                )

            # Debug: Check P values after DataFrame conversion
            if self.verbose and "P_global_center_idx" in self.output_dict:
                p_dict_values = self.output_dict["P_global_center_idx"][:min(5, len(self.output_dict["P_global_center_idx"]))]
                p_df_values = self.output_df["P_global_center_idx"].head(5).values if "P_global_center_idx" in self.output_df.columns else None
                print(f"[DEBUG] After DataFrame conversion: dict values={p_dict_values}, df values={p_df_values}")
            
            # Compute beat-to-beat variability metrics
            if len(self.output_df) > 0 and not self.cfg.lite_mode:
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
