from __future__ import annotations

##PyHEARTS IMPORTS
import logging
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt


# Custom imports for PyHEARTS
from pyhearts.config import ProcessCycleConfig
from pyhearts.feature import calc_intervals, interval_ms, extract_shape_features
from pyhearts.feature.st_segment import extract_st_segment_features
from pyhearts.fitmetrics import calc_r_squared, calc_rmse
from pyhearts.plots import plot_fit, plot_labeled_peaks, plot_rise_decay
from .bounds import calc_bounds, calc_bounds_skewed
from .detrend import detrend_signal
from .gaussian import compute_gauss_std, gaussian_function, skewed_gaussian_function
from .peaks import find_peaks, find_peak_derivative_based, refine_peak_parabolic
from .validation import (
    validate_peaks,
    validate_cycle_physiology,
    validate_peak_temporal_order,
    validate_intervals_physiological,
)
from .waveletoffset import calc_wavelet_dynamic_offset
from .snrgate import gate_by_local_mad
from .adaptive_threshold import gate_by_adaptive_threshold
from .qrs_boundary_detection_v2 import detect_qrs_onset_derivative, detect_qrs_end_derivative
from .derivative_t_detection import (
    compute_filtered_derivative,
    detect_t_wave_derivative_based,
)
from .p_wave_detection_fixed_window import detect_p_wave_fixed_window
from .p_wave_detection_improved import detect_p_wave_improved
from .p_wave_detection_derivative_validated import (
    detect_p_wave_derivative_validated,
    thresholdcross,
    bandpass_filter_p_wave,
)


def bandpass_filter_pwave(
    signal: np.ndarray,
    sampling_rate: float,
    lowcut: float = 5.0,
    highcut: float = 15.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply band-pass filter for P-wave detection.
    
    Enhances P-wave visibility by filtering in the 5-15 Hz range, which:
    - Removes low-frequency baseline wander (<5 Hz)
    - Removes high-frequency noise (>15 Hz)
    - Preserves P-wave morphology
    
    Parameters
    ----------
    signal : np.ndarray
        Input ECG signal.
    sampling_rate : float
        Sampling rate in Hz.
    lowcut : float
        Low cutoff frequency in Hz (default 5.0).
    highcut : float
        High cutoff frequency in Hz (default 15.0).
    order : int
        Filter order (default 4).
    
    Returns
    -------
    np.ndarray
        Band-pass filtered signal.
    """
    nyq = sampling_rate / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    
    # Ensure cutoffs are valid
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    try:
        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        return filtered
    except Exception:
        # Fallback: return original signal if filtering fails
        return signal


def process_cycle(
    one_cycle,
    output_dict,
    sampling_rate,
    cycle_idx,
    previous_r_global_center_idx,
    previous_p_global_center_idx,
    previous_gauss_features=None,
    expected_max_energy=None,
    plot=False,
    verbose=False,
    cfg: ProcessCycleConfig | None = None,
    precomputed_peaks: dict | None = None,
    original_r_peaks: np.ndarray | None = None,
    full_derivative: np.ndarray | None = None,
    p_training_signal_peak: float | None = None,
    p_training_noise_peak: float | None = None,
):

    cfg = cfg or ProcessCycleConfig()  # safe default
    
    # CRITICAL: Always log entry to process_cycle for ALL cycles (sample every 10 to avoid spam, but always log errors)
    # This ensures we can detect if cycles are being processed at all
    # Always log for cycles 50-60 and 21-29 to debug missing cycles
    if 50 <= cycle_idx <= 60 or 21 <= cycle_idx <= 29 or cycle_idx % 10 == 0 or cycle_idx < 20:
        logging.info(f"[PROCESS_CYCLE_ENTRY] Cycle {cycle_idx}: Entered process_cycle function")
    
    # Debug: Always log cycle start for first few cycles (regardless of verbose)
    if cycle_idx < 3:
        logging.debug(f"[processcycle.py] Starting process_cycle for cycle {cycle_idx}, verbose={verbose}")
    
    if verbose or plot:
        print(f"[Cycle {cycle_idx}]: Starting process_cycle()")
        print("=" * 80)

    # Step 1: Basic Input Validation
    # Always log input validation (sample every 10 cycles to avoid spam, but always log errors)
    # Always log for cycles 50-60 and 21-29 to debug missing cycles
    if 50 <= cycle_idx <= 60 or 21 <= cycle_idx <= 29 or cycle_idx % 10 == 0 or cycle_idx < 20:
        logging.info(f"[INPUT_VALIDATION] Cycle {cycle_idx}: Checking input - one_cycle.empty={one_cycle.empty if one_cycle is not None else 'one_cycle is None'}, one_cycle len={len(one_cycle) if one_cycle is not None and not one_cycle.empty else 0}, has_nan={one_cycle['signal_y'].isnull().any() if one_cycle is not None and not one_cycle.empty else 'N/A'}")
    
    if one_cycle.empty or one_cycle["signal_y"].isnull().any():
        reason = "empty" if one_cycle.empty else "contains NaN values"
        # Always log validation failures (critical issue)
        logging.warning(f"[INPUT_VALIDATION_FAIL] Cycle {cycle_idx}: Input cycle {reason}. Skipping this cycle.")
        if verbose:
            print(f"[Cycle {cycle_idx}]: Input cycle {reason}. Skipping this cycle.")
        return output_dict, previous_r_global_center_idx, previous_p_global_center_idx, None, previous_gauss_features

    if cycle_idx < 3:
        logging.debug(f"[processcycle.py] Cycle {cycle_idx}: Input validation passed, cycle length={len(one_cycle)}")
    if verbose:
        print(f"[Cycle {cycle_idx}]: Input validation passed.")

    # Step 2: Prepare Signal Data
    # DEBUG: Track one_cycle at start
    if verbose:
        print(f"[Cycle {cycle_idx}]: DEBUG - one_cycle at start: len={len(one_cycle)}, index.iloc[0]={one_cycle['index'].iloc[0] if 'index' in one_cycle.columns and len(one_cycle) > 0 else 'N/A'}, index.iloc[-1]={one_cycle['index'].iloc[-1] if 'index' in one_cycle.columns and len(one_cycle) > 0 else 'N/A'}")
        if 'cycle' in one_cycle.columns:
            print(f"[Cycle {cycle_idx}]: DEBUG - one_cycle cycle column values: {one_cycle['cycle'].unique() if len(one_cycle) > 0 else 'N/A'}")
    
    cycle_start_from_index = int(one_cycle["index"].iloc[0])
    cycle_end_from_index = int(one_cycle["index"].iloc[-1]) + 1
    xs_samples = np.arange(cycle_start_from_index, cycle_end_from_index)
    xs_rel_idxs = np.arange(len(xs_samples))
    if verbose:
        print(f"[Cycle {cycle_idx}]: DEBUG - xs_samples created: start={cycle_start_from_index}, end={cycle_end_from_index}, len={len(xs_samples)}, xs_samples[0]={xs_samples[0] if len(xs_samples) > 0 else 'N/A'}, xs_samples[-1]={xs_samples[-1] if len(xs_samples) > 0 else 'N/A'}")
    sig = one_cycle["signal_y"].to_numpy()

    # Step 3: Detrend Signal
    # CRITICAL: Always log detrending for cycles 50-60 to debug missing cycles 51-59
    if 50 <= cycle_idx <= 60:
        logging.info(f"[BEFORE_DETREND] Cycle {cycle_idx}: About to detrend signal - sig len={len(sig)}, xs_rel_idxs len={len(xs_rel_idxs)}")
    
    try:
        sig_detrended, trend = detrend_signal(
            xs_rel_idxs, sig, sampling_rate=sampling_rate, window_ms=cfg.detrend_window_ms, cycle=cycle_idx, plot=plot
        )
        # CRITICAL: Always log after detrending for cycles 50-60
        if 50 <= cycle_idx <= 60:
            logging.info(f"[AFTER_DETREND] Cycle {cycle_idx}: Detrending complete - sig_detrended len={len(sig_detrended) if sig_detrended is not None else None}, has_nan={np.isnan(sig_detrended).any() if sig_detrended is not None and len(sig_detrended) > 0 else 'N/A'}")
    except Exception as e:
        # CRITICAL: Always log exceptions in detrending
        logging.error(f"[DETREND_ERROR] Cycle {cycle_idx}: Exception in detrend_signal: {e}")
        import traceback
        logging.error(f"[DETREND_ERROR] Cycle {cycle_idx} traceback:\n{traceback.format_exc()}")
        raise

    output_dict["cycle_trend"][cycle_idx] = trend

    if verbose:
        print(f"[Cycle {cycle_idx}]: Detrending complete and trend saved.")

    # ============================================================================================
    # ============================================================================================
    # Always run full peak detection in every cycle
    # If previous cycle available, use it as seeds/bounds for curve fitting (optimization)
    # If not available, compute seeds/bounds from detected peaks
    # ============================================================================================
    
    # Check if we have previous cycle Gaussian features to use as seeds/bounds
    has_previous_seeds = (
        previous_gauss_features is not None 
        and isinstance(previous_gauss_features, dict) 
        and len(previous_gauss_features) > 0
    )
    
    if has_previous_seeds:
        essential_peaks = ["P", "R", "T"]
        missing_essential = [peak for peak in essential_peaks if peak not in previous_gauss_features]
        if len(missing_essential) > 0:
            has_previous_seeds = False  # Previous cycle incomplete, don't use as seeds
            if verbose:
                print(f"[Cycle {cycle_idx}]: Previous cycle missing essential peaks: {missing_essential}. Computing seeds from detected peaks.")
        else:
            if verbose:
                print(f"[Cycle {cycle_idx}]: Using previous cycle Gaussian features as seeds/bounds for curve fitting.")
    else:
        if verbose:
            print(f"[Cycle {cycle_idx}]: No previous cycle available. Computing seeds from detected peaks.")
    
    # ============================================================================================
    # Full Peak Detection (runs in every cycle)
    # ============================================================================================
    
    if verbose:
        print(f"[Cycle {cycle_idx}]: Running full peak detection...")

    # ------------------------------------------------------
    # Step 1: R Peak Detection and Initial Checks
    # ------------------------------------------------------
    # Always log cycle start (sample every 10 cycles to avoid spam, but always log errors)
    # Always log for cycles 50-60 and 21-29 to debug missing cycles
    if 50 <= cycle_idx <= 60 or 21 <= cycle_idx <= 29 or cycle_idx % 10 == 0 or cycle_idx < 20:
        logging.info(f"[CYCLE_START] Cycle {cycle_idx}: Starting R peak detection - sig_detrended len={len(sig_detrended) if sig_detrended is not None else None}, one_cycle len={len(one_cycle) if one_cycle is not None else None}")
    
    if len(sig_detrended) == 0 or np.isnan(sig_detrended).any():
        # Always log invalid signal (critical issue)
        logging.warning(f"[SIG_INVALID] Cycle {cycle_idx}: sig_detrended is invalid (empty or NaNs). Skipping this cycle. len={len(sig_detrended)}, has_nan={np.isnan(sig_detrended).any() if len(sig_detrended) > 0 else 'N/A'}")
        if verbose:
            print(f"[Cycle {cycle_idx}]: sig_detrended is invalid (empty or NaNs). Skipping this cycle.")
        return output_dict, previous_r_global_center_idx, previous_p_global_center_idx, None, None

    # R Peak Detection: Handle both upright and inverted R peaks
    # Use absolute value to find the peak with maximum magnitude, regardless of polarity
    # This is more robust than argmax/argmin alone and handles variable R peak timing
    r_center_idx = None
    r_height = None
    r_detection_failed = False
    r_detection_failure_reason = None
    
    try:
        # Validate signal before detection
        if len(sig_detrended) == 0:
            r_detection_failed = True
            r_detection_failure_reason = "sig_detrended is empty"
            logging.warning(f"[R_DETECT_FAIL] Cycle {cycle_idx}: {r_detection_failure_reason}")
        elif np.all(np.isnan(sig_detrended)):
            r_detection_failed = True
            r_detection_failure_reason = "sig_detrended contains only NaN values"
            logging.warning(f"[R_DETECT_FAIL] Cycle {cycle_idx}: {r_detection_failure_reason}")
        elif np.all(sig_detrended == 0):
            r_detection_failed = True
            r_detection_failure_reason = "sig_detrended is all zeros"
            logging.warning(f"[R_DETECT_FAIL] Cycle {cycle_idx}: {r_detection_failure_reason}")
        else:
            sig_abs = np.abs(sig_detrended)
            max_abs_val = np.max(sig_abs)
            if max_abs_val == 0 or np.isnan(max_abs_val):
                r_detection_failed = True
                r_detection_failure_reason = f"max absolute value is invalid (max={max_abs_val})"
                logging.warning(f"[R_DETECT_FAIL] Cycle {cycle_idx}: {r_detection_failure_reason}")
            else:
                r_center_idx = int(np.argmax(sig_abs))
                r_height = sig_detrended[r_center_idx]
                
                # Validate detected R peak
                if r_center_idx < 0 or r_center_idx >= len(sig_detrended):
                    r_detection_failed = True
                    r_detection_failure_reason = f"r_center_idx out of bounds: {r_center_idx} (signal len={len(sig_detrended)})"
                    logging.warning(f"[R_DETECT_FAIL] Cycle {cycle_idx}: {r_detection_failure_reason}")
                    r_center_idx = None
                    r_height = None
                elif np.isnan(r_height):
                    r_detection_failed = True
                    r_detection_failure_reason = f"r_height is NaN at index {r_center_idx}"
                    logging.warning(f"[R_DETECT_FAIL] Cycle {cycle_idx}: {r_detection_failure_reason}")
                    r_center_idx = None
                    r_height = None
                else:
                    # Log R detection start (sample every 10 cycles, but always for cycles 50-60 and 21-29)
                    if 50 <= cycle_idx <= 60 or 21 <= cycle_idx <= 29 or cycle_idx % 10 == 0 or cycle_idx < 20:
                        logging.info(f"[R_DETECT_START] Cycle {cycle_idx}: R peak detection started - r_center_idx={r_center_idx}, r_height={r_height:.4f}, sig_detrended_len={len(sig_detrended)}")
    except Exception as e:
        # Always log exceptions (critical errors)
        r_detection_failed = True
        r_detection_failure_reason = f"Exception: {e}"
        logging.error(f"[R_DETECT_ERROR] Cycle {cycle_idx}: Exception in R peak detection: {e}")
        import traceback
        logging.error(f"[R_DETECT_ERROR] Cycle {cycle_idx} traceback:\n{traceback.format_exc()}")
        r_center_idx = None
        r_height = None
    
    # CRITICAL: Always log R detection result (success or failure) for ALL cycles
    # This ensures we can track why R detection fails for specific cycles
    # CRITICAL: This log MUST execute for all cycles - if it doesn't, there's a bug or exception
    try:
        if r_center_idx is None or r_height is None:
            # Always log failures (critical issue)
            logging.warning(f"[R_DETECT_RESULT] Cycle {cycle_idx}: R detection FAILED - r_center_idx={r_center_idx}, r_height={r_height}, reason={r_detection_failure_reason if 'r_detection_failure_reason' in locals() else 'unknown'}, sig_detrended_len={len(sig_detrended) if sig_detrended is not None else None}, sig_detrended_max={np.max(np.abs(sig_detrended)) if sig_detrended is not None and len(sig_detrended) > 0 else None}")
        else:
            # Always log success for cycles 50-60 and 21-29 to debug missing cycles
            if 50 <= cycle_idx <= 60 or 21 <= cycle_idx <= 29:
                logging.info(f"[R_DETECT_RESULT] Cycle {cycle_idx}: R detection SUCCESS - r_center_idx={r_center_idx}, r_height={r_height:.4f}")
            # Log success (sample every 10 cycles to avoid spam)
            elif cycle_idx % 10 == 0 or cycle_idx < 20:
                logging.info(f"[R_DETECT_RESULT] Cycle {cycle_idx}: R detection SUCCESS - r_center_idx={r_center_idx}, r_height={r_height:.4f}")
    except Exception as e:
        # CRITICAL: If we can't even log the result, something is very wrong
        logging.error(f"[R_DETECT_RESULT_LOG_ERROR] Cycle {cycle_idx}: Exception while logging R detection result: {e}")
        import traceback
        logging.error(f"[R_DETECT_RESULT_LOG_ERROR] Cycle {cycle_idx} traceback:\n{traceback.format_exc()}")
    
    # Optional: Use cycle center as a sanity check/refinement if available
    # This helps with very noisy signals but shouldn't override the absolute value method
    # CRITICAL: Wrap in try-except to catch any errors and always log for cycles 50-60
    try:
        cycle_start_global = int(one_cycle["index"].iloc[0])
        cycle_end_global = int(one_cycle["index"].iloc[-1])
        cycle_center_global = (cycle_start_global + cycle_end_global) // 2
        cycle_center_rel = cycle_center_global - cycle_start_global
        
        # If the absolute value peak is far from cycle center (>50ms), check if cycle center is better
        # This handles cases where detrending creates artifacts
        max_distance_samples = int(round(50.0 * sampling_rate / 1000.0))
        if r_center_idx is not None and abs(r_center_idx - cycle_center_rel) > max_distance_samples:
            if 0 <= cycle_center_rel < len(sig_detrended):
                cycle_center_abs = abs(sig_detrended[cycle_center_rel])
                # Only use cycle center if it's within 20% of the absolute peak
                # CRITICAL: sig_abs is only defined inside the try block, so we need to recompute it or check if it exists
                if 'sig_abs' in locals() and sig_abs is not None and r_center_idx < len(sig_abs):
                    if cycle_center_abs >= 0.8 * sig_abs[r_center_idx]:
                        if verbose:
                            print(f"[Cycle {cycle_idx}]: Using cycle center for R peak (abs peak too far: {abs(r_center_idx - cycle_center_rel)} samples)")
                        r_center_idx = cycle_center_rel
                        r_height = sig_detrended[r_center_idx]
    except Exception as e:
        # CRITICAL: Always log exceptions in cycle center refinement
        logging.error(f"[CYCLE_CENTER_REFINE_ERROR] Cycle {cycle_idx}: Exception in cycle center refinement: {e}")
        import traceback
        logging.error(f"[CYCLE_CENTER_REFINE_ERROR] Cycle {cycle_idx} traceback:\n{traceback.format_exc()}")
        # Don't fail the cycle - continue with original r_center_idx
    
    # Debug: Always log R peak detection (sample every 50 cycles to avoid spam)
    if cycle_idx % 50 == 0 or cycle_idx < 10:
        try:
            r_polarity = "INVERTED" if r_height < 0 else "UPRIGHT"
            r_global = int(xs_samples[r_center_idx]) if r_center_idx < len(xs_samples) else None
            logging.info(f"[R_DETECT] Cycle {cycle_idx}: R detected at rel_idx={r_center_idx}, global_idx={r_global}, height={r_height:.4f} mV ({r_polarity})")
        except Exception as e:
            if cycle_idx < 20 or cycle_idx % 10 == 0:
                logging.error(f"[R_DETECT_EXCEPTION] Cycle {cycle_idx}: Exception in R_DETECT logging: {e}")
                import traceback
                logging.error(traceback.format_exc())
        
        if verbose:
            r_polarity = "INVERTED" if r_height < 0 else "UPRIGHT"
            print(f"[Cycle {cycle_idx}]: R peak detected at {r_center_idx} (cycle-rel), value: {r_height:.4f} mV ({r_polarity})")

    # ------------------------------------------------------
    # Step 2: R Peak Stdev and Dynamic Offset
    # ------------------------------------------------------
    if verbose:
        print(f"[Cycle {cycle_idx}]: Finding half-height indices for R peak...")

    # Log before compute_gauss_std (sample every 10 cycles, but always for cycles 50-60 and 21-29)
    if 50 <= cycle_idx <= 60 or 21 <= cycle_idx <= 29 or cycle_idx % 10 == 0 or cycle_idx < 20:
        logging.info(f"[BEFORE_GAUSS_STD] Cycle {cycle_idx}: About to call compute_gauss_std - r_center_idx={r_center_idx}, r_height={r_height}, sig_detrended_len={len(sig_detrended) if sig_detrended is not None else None}")

    # Exception handling for compute_gauss_std
    r_std = None
    r_std_dict = {}
    try:
        r_guess = {"R": (r_center_idx, r_height)}
        # Always log for cycles 50-60 to debug missing cycles 51-59
        if 50 <= cycle_idx <= 60 or cycle_idx % 10 == 0 or cycle_idx < 20:
            logging.info(f"[GAUSS_STD_CALL] Cycle {cycle_idx}: Calling compute_gauss_std with r_guess={r_guess}")
        r_std_dict = compute_gauss_std(sig_detrended, r_guess)
        r_std = r_std_dict.get("R")
        
        # Log that compute_gauss_std completed (always for cycles 50-60 and 21-29)
        if 50 <= cycle_idx <= 60 or 21 <= cycle_idx <= 29 or cycle_idx % 10 == 0 or cycle_idx < 20:
            logging.info(f"[GAUSS_STD] Cycle {cycle_idx}: compute_gauss_std completed - r_std={r_std}, r_std_dict={r_std_dict}")
    except Exception as e:
        # Always log exceptions (critical errors)
        logging.error(f"[GAUSS_STD_ERROR] Cycle {cycle_idx}: Exception in compute_gauss_std: {e}")
        import traceback
        logging.error(f"[GAUSS_STD_ERROR] Cycle {cycle_idx} traceback:\n{traceback.format_exc()}")
        r_std = None
        r_std_dict = {}
    
    # Log after compute_gauss_std (always for cycles 50-60 and 21-29)
    if 50 <= cycle_idx <= 60 or 21 <= cycle_idx <= 29 or cycle_idx % 10 == 0 or cycle_idx < 20:
        logging.info(f"[AFTER_GAUSS_STD] Cycle {cycle_idx}: After compute_gauss_std - r_std={r_std}, r_std_dict={r_std_dict}")

    # CRITICAL: Always log r_center_idx status for cycles 50-60 before the check
    if 50 <= cycle_idx <= 60:
        logging.info(f"[BEFORE_R_CHECK] Cycle {cycle_idx}: About to check r_center_idx - r_center_idx={r_center_idx}, r_height={r_height if 'r_height' in locals() else 'N/A'}")
    
    if r_center_idx is None:
        # CRITICAL: Always log early return due to missing R (critical issue)
        # This should NEVER happen silently - if we reach here, R detection failed
        logging.warning(f"[EARLY_RETURN_R_NONE] Cycle {cycle_idx}: Returning early because r_center_idx is None. sig_detrended len={len(sig_detrended) if sig_detrended is not None else None}, sig_abs max={np.max(np.abs(sig_detrended)) if sig_detrended is not None and len(sig_detrended) > 0 else None}, sig_abs mean={np.mean(np.abs(sig_detrended)) if sig_detrended is not None and len(sig_detrended) > 0 else None}, failure_reason={r_detection_failure_reason if 'r_detection_failure_reason' in locals() else 'unknown'}")
        if verbose:
            print(f"[Cycle {cycle_idx}]: Error -  R peak not detected — skipping cycle.")
        return output_dict, previous_r_global_center_idx, previous_p_global_center_idx, None, previous_gauss_features
    
    # Log that we passed the r_center_idx check (always for cycles 50-60 and 21-29)
    if 50 <= cycle_idx <= 60 or 21 <= cycle_idx <= 29 or cycle_idx % 10 == 0 or cycle_idx < 20:
        logging.info(f"[PASSED_R_CHECK] Cycle {cycle_idx}: r_center_idx is not None, continuing processing")

    if r_std is None:
        # Always log when r_std is None (potential issue)
        logging.warning(f"[GAUSS_STD] Cycle {cycle_idx}: r_std is None - continuing anyway")
        if verbose:
            print(f"[Cycle {cycle_idx}]: Could not estimate std for R peak.")
    
    # Log before wavelet offset (always for cycles 50-60 and 21-29)
    if 50 <= cycle_idx <= 60 or 21 <= cycle_idx <= 29 or cycle_idx % 10 == 0 or cycle_idx < 20:
        logging.info(f"[BEFORE_WAVELET] Cycle {cycle_idx}: About to call calc_wavelet_dynamic_offset - r_center_idx={r_center_idx}, r_std={r_std}")

    # Dynamic offset (in samples)
    offset_samples = None
    q_start = None
    s_end = None
    try:
        # Always log for cycles 50-60 and 21-29
        if 50 <= cycle_idx <= 60 or 21 <= cycle_idx <= 29 or cycle_idx % 10 == 0 or cycle_idx < 20:
            logging.info(f"[WAVELET_CALL] Cycle {cycle_idx}: Calling calc_wavelet_dynamic_offset with r_center_idx={r_center_idx}, r_std={r_std}, sig_len={len(sig_detrended)}")
        offset_samples, _, _, q_start, s_end = calc_wavelet_dynamic_offset(
            ecg_signal=sig_detrended,
            sampling_rate=sampling_rate,
            expected_max_energy=expected_max_energy,
            xs=xs_rel_idxs,
            r_center_idx=r_center_idx,
            r_std=r_std,
            plot=plot,
            cfg=cfg,
        )
        # Log successful completion (always for cycles 50-60 and 21-29)
        if 50 <= cycle_idx <= 60 or 21 <= cycle_idx <= 29 or cycle_idx % 10 == 0 or cycle_idx < 20:
            logging.info(f"[WAVELET_OFFSET] Cycle {cycle_idx}: calc_wavelet_dynamic_offset completed successfully - offset_samples={offset_samples}, q_start={q_start}, s_end={s_end}")
    except Exception as e:
        # Always log exceptions (critical errors)
        logging.error(f"[WAVELET_OFFSET_ERROR] Cycle {cycle_idx}: Exception in calc_wavelet_dynamic_offset: {e}")
        import traceback
        logging.error(f"[WAVELET_OFFSET_ERROR] Cycle {cycle_idx} traceback:\n{traceback.format_exc()}")
        # Re-raise to prevent silent failure
        raise
        # Fallback: use default values
        offset_samples = None
        q_start = None
        s_end = len(sig_detrended) - 100 if len(sig_detrended) > 100 else len(sig_detrended) - 1
        # Always log fallback usage (potential issue)
        logging.warning(f"[WAVELET_OFFSET_FALLBACK] Cycle {cycle_idx}: Using fallback values - q_start={q_start}, s_end={s_end}, sig_len={len(sig_detrended)}")
    
    # Log that calc_wavelet_dynamic_offset completed (always for cycles 50-60 and 21-29)
    if 50 <= cycle_idx <= 60 or 21 <= cycle_idx <= 29 or cycle_idx % 10 == 0 or cycle_idx < 20:
        logging.info(f"[AFTER_WAVELET] Cycle {cycle_idx}: After calc_wavelet_dynamic_offset - offset_samples={offset_samples}, q_start={q_start}, s_end={s_end}")

    # ------------------------------------------------------
    # Step 3: Peak Detection - Q, P, S, T
    # ------------------------------------------------------
    
    # CRITICAL: Clamp s_end to signal length - THIS MUST ALWAYS RUN
    if s_end is None or s_end >= len(sig_detrended):
        s_end = len(sig_detrended) - 100
        if verbose:
            print(f"[Cycle {cycle_idx}]: Adjusted S max index to within signal bounds: {s_end}")
        # Always log when s_end is clamped (potential issue)
        logging.warning(f"[S_END_CLAMPED] Cycle {cycle_idx}: s_end was None or out of bounds, clamped to {s_end}")

    # Initialize Q and S center indices (will be set later)
    q_center_idx = None
    s_center_idx = None
    
    # Log that we're starting Q/P/S/T detection (always for cycles 50-60 and 21-29)
    if 50 <= cycle_idx <= 60 or 21 <= cycle_idx <= 29 or cycle_idx % 10 == 0 or cycle_idx < 20:
        logging.info(f"[PEAK_DETECT_START] Cycle {cycle_idx}: Starting Q/P/S/T detection after R detection and wavelet offset")
    
    # ==============================================================================
    # COMPUTE QRS BOUNDARIES EARLY (BEFORE P/T DETECTION)
    # ==============================================================================
    # CRITICAL: This code MUST ALWAYS RUN - it was incorrectly inside a conditional block
    # CRITICAL: Compute QRS boundaries BEFORE P/T detection so they can be used
    # for P/T search windows. This is essential because P/T detection depends on
    # accurate QRS boundaries, not just Q/S peak positions.
    # Note: Q and S peaks haven't been detected yet, so we'll compute boundaries
    # from R peak only, then update them after Q/S detection if available.
    qrs_onset_idx_early = None
    qrs_end_idx_early = None
    if r_center_idx is not None:
        # Compute QRS onset (from R peak, Q will be detected later)
        qrs_onset_idx_early = detect_qrs_onset_derivative(
            signal=sig_detrended,
            q_peak_idx=None,  # Q not detected yet, search from R
            r_peak_idx=r_center_idx,
            sampling_rate=sampling_rate,
            search_window_ms=100.0,
            verbose=verbose,
            cycle_idx=cycle_idx,
        )
        # Compute QRS end (from R peak, S will be detected later)
        qrs_end_idx_early = detect_qrs_end_derivative(
            signal=sig_detrended,
            s_peak_idx=None,  # S not detected yet, search from R
            r_peak_idx=r_center_idx,
            sampling_rate=sampling_rate,
            search_window_ms=100.0,
            verbose=verbose,
            cycle_idx=cycle_idx,
        )
        if verbose:
            print(f"[Cycle {cycle_idx}]: Early QRS boundaries: onset={qrs_onset_idx_early}, end={qrs_end_idx_early}")

    # --- Q Peak ---
    # Ensure Q search is before R (fix temporal order violations)
    # Also ensure search window is wide enough (Q waves can be up to 100-150ms before R)
    if q_start is not None and r_center_idx is not None:
        if q_start >= r_center_idx:
            # q_start is after R, which is invalid - search before R instead
            q_start = max(0, r_center_idx - int(round(0.1 * sampling_rate)))  # 100ms before R
            if verbose:
                print(f"[Cycle {cycle_idx}]: Adjusted q_start from invalid position to {q_start} (before R at {r_center_idx})")
        else:
            # Ensure search window is wide enough to capture Q waves up to 150ms before R
            # The wavelet-based q_start may be too close to R (only 20-60ms), missing earlier Q waves
            min_q_search_ms = 150.0  # Minimum search window: 150ms before R
            q_start_min = max(0, r_center_idx - int(round(min_q_search_ms * sampling_rate / 1000.0)))
            if q_start > q_start_min:
                # Current q_start is too close to R (larger index), expand the search window (use smaller index)
                q_start = q_start_min
                if verbose:
                    print(f"[Cycle {cycle_idx}]: Expanded Q search window to {q_start} (150ms before R at {r_center_idx})")
    
    # Ensure q_start is valid before calling find_peaks
    if q_start is None:
        # Fallback: use default search window before R
        if r_center_idx is not None:
            q_start = max(0, r_center_idx - int(round(0.15 * sampling_rate)))  # 150ms before R
        else:
            q_start = 0
    
    # Exception handling for Q peak detection
    q_center_idx, q_height = None, None
    try:
        q_center_idx, q_height, _ = find_peaks(
            sig_detrended,
            xs_rel_idxs,
            q_start,
            r_center_idx,
            mode="min",
            verbose=verbose,
            label="Q",
            cycle_idx=cycle_idx,
        )
        # Always log for cycles 21-25 to debug Q/S detection issue
        if cycle_idx < 20 or cycle_idx % 10 == 0 or (21 <= cycle_idx <= 25):
            if q_center_idx is not None:
                logging.info(f"[Q_DETECT] Cycle {cycle_idx}: Q peak detected at idx={q_center_idx}, height={q_height:.4f}")
            else:
                logging.warning(f"[Q_DETECT] Cycle {cycle_idx}: Q peak NOT detected - q_start={q_start}, r_center_idx={r_center_idx}, sig_len={len(sig_detrended)}")
    except Exception as e:
        if cycle_idx < 20 or cycle_idx % 10 == 0:
            logging.error(f"[Q_DETECT_ERROR] Cycle {cycle_idx}: Exception in Q peak detection: {e}")
            import traceback
            logging.error(traceback.format_exc())
        q_center_idx, q_height = None, None

    if q_center_idx is None and verbose:
        print(f"[Cycle {cycle_idx}]: Q peak rejected — not included in fit.")
    
    # For validation purposes, use detected Q
    q_center_idx_for_p_validation = q_center_idx

    # --- P Peak ---
    # Check if precomputed peaks are available
    p_center_idx, p_height = None, None
    p_onset_idx, p_offset_idx = None, None
    expected_polarity = "positive"  # Default, will be updated when P is detected
    
    if verbose:
        print(f"[Cycle {cycle_idx}]: P detection: Checking precomputed peaks. precomputed_peaks={precomputed_peaks is not None}, cycle_idx in precomputed_peaks={precomputed_peaks is not None and cycle_idx in precomputed_peaks if precomputed_peaks is not None else False}")
    
    if precomputed_peaks is not None and cycle_idx in precomputed_peaks:
        p_annotation = precomputed_peaks[cycle_idx].get('P')
        if verbose:
            print(f"[Cycle {cycle_idx}]: P detection: p_annotation={p_annotation is not None}")
        if p_annotation is not None:
            # Use precomputed P-wave annotation
            # Map from global signal indices to cycle-relative indices
            cycle_start_global = int(one_cycle["signal_x"].iloc[0])
            cycle_end_global = int(one_cycle["signal_x"].iloc[-1])
            cycle_length = len(one_cycle)
            
            # Check if peak is within cycle boundaries
            if cycle_start_global <= p_annotation.peak_idx <= cycle_end_global:
                p_center_idx = p_annotation.peak_idx - cycle_start_global
                # Ensure index is within cycle array bounds
                if 0 <= p_center_idx < cycle_length:
                    p_height = p_annotation.peak_amplitude
                    if p_annotation.onset_idx is not None and cycle_start_global <= p_annotation.onset_idx <= cycle_end_global:
                        p_onset_idx = p_annotation.onset_idx - cycle_start_global
                        p_onset_idx = max(0, min(p_onset_idx, cycle_length - 1))
                    if p_annotation.offset_idx is not None and cycle_start_global <= p_annotation.offset_idx <= cycle_end_global:
                        p_offset_idx = p_annotation.offset_idx - cycle_start_global
                        p_offset_idx = max(0, min(p_offset_idx, cycle_length - 1))
                    
                    if verbose:
                        print(f"[Cycle {cycle_idx}]: Using precomputed P-wave at global idx {p_annotation.peak_idx} (cycle-relative: {p_center_idx})")
                else:
                    # Peak outside cycle array bounds, skip precomputed detection
                    p_center_idx = None
                    if verbose:
                        print(f"[Cycle {cycle_idx}]: P detection: Precomputed P peak outside cycle array bounds, setting p_center_idx=None")
            else:
                # Peak outside cycle boundaries, skip precomputed detection
                p_center_idx = None
                if verbose:
                    print(f"[Cycle {cycle_idx}]: P detection: Precomputed P peak outside cycle boundaries, setting p_center_idx=None")
    
    if verbose:
        print(f"[Cycle {cycle_idx}]: P detection: After precomputed check, p_center_idx={p_center_idx}")
    
    # If precomputed peaks didn't provide a P-wave, use standard detection
    if p_center_idx is None:
        if verbose:
            print(f"[Cycle {cycle_idx}]: P detection: p_center_idx is None, attempting standard detection")
        
        # Initialize p_peak_end_idx for use in standard method (if needed)
        p_peak_end_idx = None
        
        # Try derivative-validated P wave detection first (if enabled), then improved, then fixed-window, then standard
        use_derivative_validated = cfg is not None and hasattr(cfg, 'p_use_derivative_validated_method') and cfg.p_use_derivative_validated_method
        use_improved = cfg is not None and hasattr(cfg, 'p_use_improved_method') and cfg.p_use_improved_method
        use_fixed_window = cfg is not None and cfg.p_use_fixed_window_method
        
        if use_derivative_validated or use_improved or use_fixed_window:
            # Get cycle start global index for converting results to global indices
            # Use "index" column (same as xs_samples) for consistency with global index conversion
            cycle_start_global = int(one_cycle["index"].iloc[0]) if "index" in one_cycle.columns and not one_cycle.empty else (int(one_cycle["signal_x"].iloc[0]) if not one_cycle.empty else 0)
            if verbose:
                print(f"[Cycle {cycle_idx}]: DEBUG - cycle_start_global calculated: {cycle_start_global}, one_cycle['index'].iloc[0]={one_cycle['index'].iloc[0] if 'index' in one_cycle.columns and len(one_cycle) > 0 else 'N/A'}, cycle_start_from_index={cycle_start_from_index if 'cycle_start_from_index' in locals() else 'N/A'}")
            
            # P wave detection methods use QRS onset - calculate actual QRS onset using derivative-based method
            # QRS onset detection method uses QRS1 which is QRS onset (start of QRS complex) detected via threshold crossing
            # Use derivative-based threshold crossing for accurate QRS onset detection
            qrs_onset_idx = None
            if r_center_idx is not None:
                # Use derivative-based QRS onset detection (similar to established methods)
                qrs_onset_idx = detect_qrs_onset_derivative(
                    signal=sig_detrended,
                    q_peak_idx=q_center_idx,
                    r_peak_idx=r_center_idx,
                    sampling_rate=sampling_rate,
                    search_window_ms=100.0,
                    verbose=verbose,
                    cycle_idx=cycle_idx,
                )
                if verbose:
                    print(f"[Cycle {cycle_idx}]: QRS onset detected at {qrs_onset_idx} (derivative-based)")
            else:
                # Fallback: estimate QRS onset as 40ms before R if R not available (shouldn't happen)
                qrs_onset_offset_ms = 40.0
                qrs_onset_offset_samples = int(round(qrs_onset_offset_ms * sampling_rate / 1000.0))
                qrs_onset_idx = max(0, r_center_idx - qrs_onset_offset_samples) if r_center_idx is not None else None
            
            if qrs_onset_idx is not None and qrs_onset_idx > 0:
                # Get previous T end if available (from previous cycle) - convert to cycle-relative
                previous_t_end_idx = None
                if previous_gauss_features is not None and "T" in previous_gauss_features:
                    # Extract T offset from previous cycle if available
                    # This is approximate - would need to track T offset from previous cycle
                    pass
                
                # Get previous P end if available - convert global to cycle-relative
                previous_p_end_idx = None
                if previous_p_global_center_idx is not None:
                    previous_p_end_idx = previous_p_global_center_idx - cycle_start_global
                    if previous_p_end_idx < 0:
                        previous_p_end_idx = None
                
                # Get max derivative from QRS region (not entire signal)
                # dermax is calculated from QRS complex region for more accurate thresholding
                # IMPORTANT: Calculate derivative from cycle segment, not full_derivative (which is global)
                max_derivative = None
                if r_center_idx is not None:
                    # Calculate derivative from cycle segment for accurate QRS region extraction
                    # Use the filtered signal for derivative calculation (same as P detection uses)
                    Xpb_cycle = bandpass_filter_p_wave(sig_detrended, sampling_rate, lowcut=1.0, highcut=60.0, order=2)
                    cycle_derivative = np.diff(Xpb_cycle)
                    
                    # Calculate dermax from QRS region (±70ms around R peak) using cycle-relative indices
                    qrs_window_ms = 70.0
                    qrs_window_samples = int(round(qrs_window_ms * sampling_rate / 1000.0))
                    qrs_start = max(0, r_center_idx - qrs_window_samples)
                    qrs_end = min(len(cycle_derivative), r_center_idx + qrs_window_samples)
                    qrs_derivative = cycle_derivative[qrs_start:qrs_end]
                    
                    if len(qrs_derivative) > 0:
                        max_derivative = float(np.max(np.abs(qrs_derivative)))
                        if verbose:
                            print(f"[Cycle {cycle_idx}]: dermax from QRS region: {max_derivative:.6f} (window: {qrs_start} to {qrs_end}, cycle-relative)")
                    else:
                        # Fallback: use entire cycle derivative
                        if len(cycle_derivative) > 0:
                            max_derivative = float(np.max(np.abs(cycle_derivative)))
                            if verbose:
                                print(f"[Cycle {cycle_idx}]: dermax from cycle derivative (fallback): {max_derivative:.6f}")
                elif full_derivative is not None:
                    # Fallback: use full signal if R peak not available
                    max_derivative = float(np.max(np.abs(full_derivative)))
                    if verbose:
                        print(f"[Cycle {cycle_idx}]: dermax from full signal (fallback): {max_derivative:.6f}")
                
                # Safety check: ensure dermax is not zero (would break validation)
                if max_derivative is not None and max_derivative == 0.0:
                    # Calculate from entire cycle derivative as last resort
                    Xpb_cycle = bandpass_filter_p_wave(sig_detrended, sampling_rate, lowcut=1.0, highcut=60.0, order=2)
                    cycle_derivative = np.diff(Xpb_cycle)
                    if len(cycle_derivative) > 0:
                        max_derivative = float(np.max(np.abs(cycle_derivative)))
                        if verbose:
                            print(f"[Cycle {cycle_idx}]: dermax was 0.0, recalculated from cycle: {max_derivative:.6f}")
                    if max_derivative == 0.0:
                        # Final fallback: use a small non-zero value to prevent division by zero
                        max_derivative = 0.001
                        if verbose:
                            print(f"[Cycle {cycle_idx}]: WARNING: dermax still 0.0, using fallback value 0.001")
                
                if use_derivative_validated:
                    if verbose:
                        print(f"[Cycle {cycle_idx}]: Using derivative-validated P wave detection")
                        print(f"[Cycle {cycle_idx}]: cycle_start_global={cycle_start_global}, qrs_onset_idx (rel)={qrs_onset_idx}, r_peak_idx (rel)={r_center_idx}")
                        print(f"[Cycle {cycle_idx}]: cycle length={len(sig_detrended)}, signal range: 0 to {len(sig_detrended)-1}")
                    
                    # Call derivative-validated P wave detection with CYCLE-RELATIVE indices
                    # The function works on sig_detrended (cycle segment) with cycle-relative indices
                    try:
                        p_peak_idx, p_amplitude, p_onset_idx, p_offset_idx = detect_p_wave_derivative_validated(
                            signal=sig_detrended,
                            qrs_onset_idx=qrs_onset_idx,  # Cycle-relative
                            r_peak_idx=r_center_idx,  # Cycle-relative
                            r_amplitude=r_height,
                            sampling_rate=sampling_rate,
                            previous_t_end_idx=previous_t_end_idx,  # Cycle-relative (or None)
                            previous_p_end_idx=previous_p_end_idx,  # Cycle-relative (or None)
                            max_derivative=max_derivative,
                            verbose=verbose,  # Use passed verbose flag
                            cycle_idx=cycle_idx,
                        )
                    except Exception as e:
                        # Log exception but don't crash - set P to None and continue
                        if cycle_idx < 20 or cycle_idx % 10 == 0:
                            logging.error(f"[P_DETECT_EXCEPTION] Cycle {cycle_idx}: Exception in detect_p_wave_derivative_validated: {e}")
                            import traceback
                            logging.error(traceback.format_exc())
                        p_peak_idx, p_amplitude, p_onset_idx, p_offset_idx = None, None, None, None
                    
                    if verbose:
                        if p_peak_idx is not None:
                            p_peak_global = p_peak_idx + cycle_start_global
                            print(f"[Cycle {cycle_idx}]: Derivative-validated P detection SUCCESS - peak (rel)={p_peak_idx}, peak (global)={p_peak_global}")
                        else:
                            print(f"[Cycle {cycle_idx}]: Derivative-validated P detection FAILED - returning None")
                    
                    # Results are cycle-relative - convert to global for storage in output_dict
                    # But keep cycle-relative for p_center_idx, p_onset_idx, p_offset_idx variables
                    if p_peak_idx is not None:
                        p_peak_idx_global = p_peak_idx + cycle_start_global
                    else:
                        p_peak_idx_global = None
                elif use_improved:
                    if verbose:
                        print(f"[Cycle {cycle_idx}]: Using improved P wave detection")
                    
                    # Call improved P wave detection
                    p_peak_idx, p_amplitude, p_onset_idx, p_offset_idx = detect_p_wave_improved(
                        signal=sig_detrended,
                        qrs_onset_idx=qrs_onset_idx,
                        r_peak_idx=r_center_idx,
                        r_amplitude=r_height,
                        sampling_rate=sampling_rate,
                        previous_t_end_idx=previous_t_end_idx,
                        previous_p_end_idx=previous_p_end_idx,
                        max_derivative=max_derivative,
                        verbose=verbose,
                        cycle_idx=cycle_idx,
                    )
                elif use_fixed_window:
                    if verbose:
                        print(f"[Cycle {cycle_idx}]: Using fixed-window P wave detection")
                    
                    # Call fixed-window P wave detection
                    p_peak_idx, p_amplitude, p_onset_idx, p_offset_idx = detect_p_wave_fixed_window(
                        signal=sig_detrended,
                        qrs_onset_idx=qrs_onset_idx,
                        r_peak_idx=r_center_idx,
                        r_amplitude=r_height,
                        sampling_rate=sampling_rate,
                        previous_t_end_idx=previous_t_end_idx,
                        verbose=verbose,
                        cycle_idx=cycle_idx,
                    )
                
                if p_peak_idx is not None:
                    p_center_idx = p_peak_idx  # Already cycle-relative from derivative-validated detection
                    p_height = p_amplitude
                    if p_onset_idx is not None:
                        p_onset_idx = p_onset_idx  # Already cycle-relative
                    if p_offset_idx is not None:
                        p_offset_idx = p_offset_idx  # Already cycle-relative
                    # Determine expected polarity from detected P wave
                    if p_height >= 0:
                        expected_polarity = "positive"
                    else:
                        expected_polarity = "negative"
                    if verbose:
                        print(f"[Cycle {cycle_idx}]: Derivative-validated P wave detected: peak={p_center_idx}, amplitude={p_height:.4f}, onset={p_onset_idx}, offset={p_offset_idx}")
                else:
                    if verbose:
                        print(f"[Cycle {cycle_idx}]: Derivative-validated P wave detection failed, falling back to standard method")
            else:
                if verbose:
                    print(f"[Cycle {cycle_idx}]: Cannot use fixed-window method - missing QRS onset (qrs_onset_idx={qrs_onset_idx})")
        
        # Use Q for P search window if available (from full detection or simplified validation)
        # This is critical: Q position helps bound P search window correctly
        # Use Q position to distinguish P from Q (helps bound search window correctly)
        # Only use standard method if derivative-validated/improved/fixed-window methods didn't find a peak
        if p_center_idx is None:  # Only use standard method if advanced methods didn't find a peak
            # CRITICAL: Use QRS onset boundary (computed early) instead of Q peak position
            # This ensures P detection works even when Q peak is not detected
            if qrs_onset_idx_early is not None:
                p_peak_end_idx = qrs_onset_idx_early
                if verbose:
                    print(f"[Cycle {cycle_idx}]: P detection: Using QRS onset boundary={qrs_onset_idx_early} (Q peak={q_center_idx})")
            else:
                # Fallback to Q peak if QRS onset not available
                p_peak_end_idx = q_center_idx if q_center_idx is not None else q_center_idx_for_p_validation
                if verbose:
                    print(f"[Cycle {cycle_idx}]: P detection: q_center_idx={q_center_idx}, q_center_idx_for_p_validation={q_center_idx_for_p_validation}, p_peak_end_idx={p_peak_end_idx}")
                if p_peak_end_idx is None:
                    # Fallback to R if no Q detected
                    p_peak_end_idx = r_center_idx
                    if verbose:
                        print(f"[Cycle {cycle_idx}]: P detection: Using R as fallback, r_center_idx={r_center_idx}, p_peak_end_idx={p_peak_end_idx}")
            
            if p_peak_end_idx is None or p_peak_end_idx < 2:
                if verbose:
                    print(f"[Cycle {cycle_idx}]: Cannot search for P peak — missing Q/R bounds (p_peak_end_idx={p_peak_end_idx}).")
                p_center_idx, p_height = None, None
            else:
                if verbose:
                    print(f"[Cycle {cycle_idx}]: P detection: p_peak_end_idx={p_peak_end_idx}, proceeding with P search")
                pre_qrs_len = int(p_peak_end_idx)
                
                # Standard P wave detection:
                # 1. Search from previous R peak (or use wider window if not available)
                # 2. End search with minimum distance from current R peak (60-80ms safety margin)
                # 3. This prevents P from encroaching on R peak and improves timing accuracy
                
                # Minimum distance from R peak (safety margin to prevent encroachment)
                min_p_to_r_distance_ms = 60.0  # Minimum 60ms before R peak
                min_p_to_r_distance_samples = int(round(min_p_to_r_distance_ms * sampling_rate / 1000.0))
                
                # End search window before R peak with safety margin
                search_end_idx = max(0, pre_qrs_len - min_p_to_r_distance_samples)
                
                # Determine search start: prefer previous R peak, fallback to wider window
                # Search from previous R to current R (entire R-R interval)
                search_start_idx = None  # Initialize to ensure it's always set
                if previous_r_global_center_idx is not None:
                    # Map previous R peak to cycle-relative index
                    cycle_start_global = int(one_cycle["index"].iloc[0]) if "index" in one_cycle.columns else int(one_cycle["signal_x"].iloc[0])
                    previous_r_cycle_relative = previous_r_global_center_idx - cycle_start_global
                    
                    if previous_r_cycle_relative >= 0 and previous_r_cycle_relative < len(sig_detrended):
                        # Search from previous R peak (full R-R interval)
                        search_start_idx = previous_r_cycle_relative
                        if verbose:
                            print(f"[Cycle {cycle_idx}]: P search from previous R peak at cycle idx {search_start_idx} (full R-R interval)")
                    else:
                        # Previous R outside cycle, estimate RR interval or use physiological maximum
                        estimated_rr_ms = None
                        if original_r_peaks is not None and len(original_r_peaks) >= 2:
                            # Estimate RR interval from detected R peaks
                            rr_intervals = np.diff(original_r_peaks) / sampling_rate * 1000.0
                            # Use median RR interval, clamped to physiological range (500-1500ms)
                            estimated_rr_ms = np.clip(np.median(rr_intervals), 500.0, 1500.0)
                        
                        max_p_window_ms = estimated_rr_ms if estimated_rr_ms is not None else cfg.shape_max_window_ms.get("P", 1200)
                        max_p_window_samples = int(round(max_p_window_ms * sampling_rate / 1000.0))
                        search_start_idx = max(0, search_end_idx - max_p_window_samples)
                        if verbose:
                            if estimated_rr_ms is not None:
                                print(f"[Cycle {cycle_idx}]: P search from estimated RR interval ({estimated_rr_ms:.1f}ms) - previous R outside cycle")
                            else:
                                print(f"[Cycle {cycle_idx}]: P search from max window ({max_p_window_ms:.1f}ms) - previous R outside cycle")
                
                if search_start_idx is None:
                    # No previous R available, estimate RR interval or use physiological maximum
                    estimated_rr_ms = None
                    if original_r_peaks is not None and len(original_r_peaks) >= 2:
                        # Estimate RR interval from detected R peaks
                        rr_intervals = np.diff(original_r_peaks) / sampling_rate * 1000.0
                        # Use median RR interval, clamped to physiological range (500-1500ms)
                        estimated_rr_ms = np.clip(np.median(rr_intervals), 500.0, 1500.0)
                    
                    max_p_window_ms = estimated_rr_ms if estimated_rr_ms is not None else cfg.shape_max_window_ms.get("P", 1200)
                    max_p_window_samples = int(round(max_p_window_ms * sampling_rate / 1000.0))
                    search_start_idx = max(0, search_end_idx - max_p_window_samples)
                    if verbose:
                        if estimated_rr_ms is not None:
                            print(f"[Cycle {cycle_idx}]: P search from estimated RR interval ({estimated_rr_ms:.1f}ms) - no previous R available")
                        else:
                            print(f"[Cycle {cycle_idx}]: P search from max window ({max_p_window_ms:.1f}ms) - no previous R available")
                
                start_idx = search_start_idx
                pre_qrs_len = search_end_idx  # Update to use safety-margin-adjusted end
            
                if pre_qrs_len - start_idx < 2:
                    if verbose:
                        print(f"[Cycle {cycle_idx}]: P window too small (start={start_idx}, end={pre_qrs_len}).")
                    p_center_idx, p_height = None, None
                else:
                    # Try band-pass filtered signal first (if enabled), then fallback to unfiltered
                    sig_for_p_detection = sig_detrended
                    
                    if cfg.pwave_use_bandpass:
                        # Filter a larger segment to avoid edge artifacts, then extract the original window
                        # Edge artifacts occur when filtering short segments, so we pad the segment
                        filter_padding_ms = 50  # Extra samples on each side to avoid edge artifacts
                        filter_padding_samples = int(round(filter_padding_ms * sampling_rate / 1000.0))
                    else:
                        filter_padding_samples = 0
                    filter_start = max(0, start_idx - filter_padding_samples)
                    filter_end = min(len(sig_detrended), pre_qrs_len + filter_padding_samples)
                    
                    # Filter larger segment
                    p_search_segment_large = sig_detrended[filter_start:filter_end]
                    
                    if len(p_search_segment_large) >= 10:
                        # Apply band-pass filter to larger segment
                        p_search_filtered_large = bandpass_filter_pwave(
                            p_search_segment_large,
                            sampling_rate,
                            lowcut=cfg.pwave_bandpass_low_hz,
                            highcut=cfg.pwave_bandpass_high_hz,
                            order=cfg.pwave_bandpass_order,
                        )
                        
                        # Extract original window from filtered segment (ignore edge padding)
                        # Calculate padding actually used (may be less if at signal boundaries)
                        actual_padding_left = start_idx - filter_start
                        actual_padding_right = filter_end - pre_qrs_len
                        
                        # Extract the portion corresponding to the original search window
                        extract_start = actual_padding_left
                        extract_end = len(p_search_filtered_large) - actual_padding_right
                        
                        if extract_end > extract_start:
                            p_search_filtered = p_search_filtered_large[extract_start:extract_end]
                        else:
                            p_search_filtered = p_search_filtered_large
                        
                        # Ensure extracted segment matches expected window size
                        expected_size = pre_qrs_len - start_idx
                        if len(p_search_filtered) != expected_size:
                            if verbose:
                                print(f"[Cycle {cycle_idx}]: Warning: Filtered segment size ({len(p_search_filtered)}) doesn't match expected window size ({expected_size}), adjusting...")
                            # Adjust to match expected size
                            if len(p_search_filtered) > expected_size:
                                p_search_filtered = p_search_filtered[:expected_size]
                            else:
                                # Pad if too short (shouldn't happen, but handle gracefully)
                                padding_needed = expected_size - len(p_search_filtered)
                                p_search_filtered = np.pad(p_search_filtered, (0, padding_needed), mode='edge')
                        
                        # Map back to full signal indices for find_peaks
                        sig_for_p_detection = sig_detrended.copy()
                        sig_for_p_detection[start_idx:pre_qrs_len] = p_search_filtered
                        
                        # Try finding P-wave in filtered signal
                        # Step 5: Try both positive and negative P-waves (inverted leads)
                        p_center_idx, p_height = None, None
                        
                        # First try positive P-wave (normal)
                        # Use derivative-based detection for better accuracy
                        p_center_idx_pos, p_height_pos, _ = find_peaks(
                            signal=sig_for_p_detection,
                            xs=xs_rel_idxs,
                            start_idx=start_idx,
                            end_idx=pre_qrs_len,
                        mode="max",
                        verbose=verbose,
                        label="P",
                        cycle_idx=cycle_idx,
                        use_derivative=True,  # Use derivative-based detection
                    )
                    
                    # Fallback: if derivative-based detection fails, try simple argmax
                    if p_center_idx_pos is None:
                        p_segment = sig_for_p_detection[start_idx:pre_qrs_len]
                        if len(p_segment) > 0:
                            p_local_max = int(np.argmax(p_segment))
                            p_center_idx_pos = start_idx + p_local_max
                            p_height_pos = sig_for_p_detection[p_center_idx_pos]
                            if verbose:
                                print(f"[Cycle {cycle_idx}]: P peak found via fallback argmax at index {p_center_idx_pos}")
                    
                    # Also try negative P-wave (inverted lead)
                    p_center_idx_neg, p_height_neg, _ = find_peaks(
                        signal=sig_for_p_detection,
                        xs=xs_rel_idxs,
                        start_idx=start_idx,
                        end_idx=pre_qrs_len,
                        mode="min",
                        verbose=verbose,
                        label="P",
                        cycle_idx=cycle_idx,
                        use_derivative=True,  # Use derivative-based detection
                    )
                    
                    # Fallback: if derivative-based detection fails, try simple argmin
                    if p_center_idx_neg is None:
                        p_segment = sig_for_p_detection[start_idx:pre_qrs_len]
                        if len(p_segment) > 0:
                            p_local_min = int(np.argmin(p_segment))
                            p_center_idx_neg = start_idx + p_local_min
                            p_height_neg = sig_for_p_detection[p_center_idx_neg]
                            if verbose:
                                print(f"[Cycle {cycle_idx}]: P peak found via fallback argmin at index {p_center_idx_neg}")
                    
                    # Choose the one with larger absolute amplitude
                    if p_center_idx_pos is not None and p_center_idx_neg is not None:
                        if abs(p_height_pos) >= abs(p_height_neg):
                            p_center_idx, p_height = p_center_idx_pos, p_height_pos
                        else:
                            p_center_idx, p_height = p_center_idx_neg, p_height_neg
                    elif p_center_idx_pos is not None:
                        p_center_idx, p_height = p_center_idx_pos, p_height_pos
                    elif p_center_idx_neg is not None:
                        p_center_idx, p_height = p_center_idx_neg, p_height_neg
                    
                    # Refine peak position in unfiltered signal to avoid filter phase shifts
                    if p_center_idx is not None:
                        # Use larger window and derivative-based detection for better accuracy
                        refine_window_samples = int(round(40 * sampling_rate / 1000.0))  # ±40ms window for better phase shift correction
                        refine_start = max(0, p_center_idx - refine_window_samples)
                        refine_end = min(len(sig_detrended), p_center_idx + refine_window_samples + 1)
                        if refine_end > refine_start:
                            # Use derivative-based peak finding in unfiltered signal (more accurate)
                            p_center_idx_refined, p_height_refined = find_peak_derivative_based(
                                sig_detrended, refine_start, refine_end, expected_polarity, verbose, "P", cycle_idx
                            )
                            if p_center_idx_refined is not None:
                                # Apply parabolic interpolation for sub-sample accuracy
                                refined_subsample = refine_peak_parabolic(sig_detrended, p_center_idx_refined)
                                p_center_idx_refined = int(np.round(refined_subsample))
                                p_center_idx_refined = np.clip(p_center_idx_refined, 0, len(sig_detrended) - 1)
                                p_height_refined = sig_detrended[p_center_idx_refined]
                                
                                if verbose and abs(p_center_idx_refined - p_center_idx) > 2:
                                    print(f"[Cycle {cycle_idx}]: P peak refined from filtered position {p_center_idx} to unfiltered position {p_center_idx_refined} (offset: {p_center_idx_refined - p_center_idx} samples)")
                                p_center_idx = p_center_idx_refined
                                p_height = p_height_refined
                            # If derivative method fails, fallback to simple argmax/argmin
                            elif refine_end - refine_start >= 3:
                                refine_seg = sig_detrended[refine_start:refine_end]
                                if expected_polarity == "positive":
                                    refine_local_idx = int(np.argmax(refine_seg))
                                else:
                                    refine_local_idx = int(np.argmin(refine_seg))
                                p_center_idx_refined = refine_start + refine_local_idx
                                p_center_idx = p_center_idx_refined
                                p_height = sig_detrended[p_center_idx]
            
            # Fallback: if band-pass failed or not enabled, try unfiltered signal
            if p_center_idx is None:
                # Ensure start_idx and pre_qrs_len are set (they should be set earlier, but check to be safe)
                if 'start_idx' not in locals() or 'pre_qrs_len' not in locals():
                    # Fallback: use default values if not set
                    pre_qrs_len = q_center_idx if q_center_idx is not None else r_center_idx
                    if pre_qrs_len is None or pre_qrs_len < 2:
                        pre_qrs_len = len(sig_detrended) // 2
                    start_idx = max(0, pre_qrs_len - int(round(1200 * sampling_rate / 1000.0)))  # 1200ms window
                
                sig_for_p_detection = sig_detrended
                
                # Try both positive and negative P-waves
                # Use derivative-based detection for better accuracy
                p_center_idx_pos, p_height_pos, _ = find_peaks(
                    signal=sig_for_p_detection,
                    xs=xs_rel_idxs,
                    start_idx=start_idx,
                    end_idx=pre_qrs_len,
                    mode="max",
                    verbose=verbose,
                    label="P",
                    cycle_idx=cycle_idx,
                    use_derivative=True,  # Use derivative-based detection
                )
                
                # Fallback: if derivative-based detection fails, try simple argmax
                if p_center_idx_pos is None:
                    p_segment = sig_for_p_detection[start_idx:pre_qrs_len]
                    if len(p_segment) > 0:
                        p_local_max = int(np.argmax(p_segment))
                        p_center_idx_pos = start_idx + p_local_max
                        p_height_pos = sig_for_p_detection[p_center_idx_pos]
                        if verbose:
                            print(f"[Cycle {cycle_idx}]: P peak found via fallback argmax at index {p_center_idx_pos}")
                
                p_center_idx_neg, p_height_neg, _ = find_peaks(
                    signal=sig_for_p_detection,
                    xs=xs_rel_idxs,
                    start_idx=start_idx,
                    end_idx=pre_qrs_len,
                    mode="min",
                    verbose=verbose,
                    label="P",
                    cycle_idx=cycle_idx,
                    use_derivative=True,  # Use derivative-based detection
                )
                
                # Fallback: if derivative-based detection fails, try simple argmin
                if p_center_idx_neg is None:
                    p_segment = sig_for_p_detection[start_idx:pre_qrs_len]
                    if len(p_segment) > 0:
                        p_local_min = int(np.argmin(p_segment))
                        p_center_idx_neg = start_idx + p_local_min
                        p_height_neg = sig_for_p_detection[p_center_idx_neg]
                        if verbose:
                            print(f"[Cycle {cycle_idx}]: P peak found via fallback argmin at index {p_center_idx_neg}")
                
                # Choose the one with larger absolute amplitude
                if p_center_idx_pos is not None and p_center_idx_neg is not None:
                    if abs(p_height_pos) >= abs(p_height_neg):
                        p_center_idx, p_height = p_center_idx_pos, p_height_pos
                    else:
                        p_center_idx, p_height = p_center_idx_neg, p_height_neg
                elif p_center_idx_pos is not None:
                    p_center_idx, p_height = p_center_idx_pos, p_height_pos
                elif p_center_idx_neg is not None:
                    p_center_idx, p_height = p_center_idx_neg, p_height_neg
                
                # Check SNR gate for unfiltered detection
                # Step 5: Determine polarity from detected P-wave
                if p_center_idx is not None:
                    # Store original detected position BEFORE SNR gate (for accurate timing)
                    p_center_idx_detected = p_center_idx
                    
                    seg = sig_detrended[start_idx:pre_qrs_len]
                    # Determine expected polarity from detected P-wave
                    p_detected_value = seg[p_center_idx - start_idx] if (p_center_idx - start_idx) < len(seg) else seg[0]
                    expected_polarity = "positive" if p_detected_value >= 0 else "negative"
                    
                    # Get peak height from segment
                    p_rel_idx = p_center_idx - start_idx
                    if 0 <= p_rel_idx < len(seg):
                        p_h = seg[p_rel_idx]
                    else:
                        p_h = seg[0] if len(seg) > 0 else 0.0
                    
                    # Validation: use training thresholds as PRIMARY if configured, else as SECONDARY
                    keep = True
                    if cfg is not None and cfg.p_use_training_as_primary and p_training_noise_peak is not None:
                        # PRIMARY validation: Training thresholds (adaptive signal/noise separation)
                        if abs(p_h) < p_training_noise_peak:
                            if verbose:
                                print(f"[Cycle {cycle_idx}]: P wave rejected by training phase (PRIMARY, unfiltered): |p_height|={abs(p_h):.4f} < noise_peak={p_training_noise_peak:.4f}")
                            keep = False
                        else:
                            # SECONDARY validation: Local MAD (optional, if training passed)
                            if keep:
                                keep, _, p_h_mad = gate_by_local_mad(
                                    seg, sampling_rate,
                                    comp="P",
                                    cand_rel_idx=p_rel_idx,
                                    expected_polarity=expected_polarity,
                                    cfg=cfg,
                                    baseline_mode="rolling",
                                )
                                if p_h_mad is not None:
                                    p_h = p_h_mad
                    else:
                        # PRIMARY validation: Local MAD (current default)
                        keep, rel_idx, p_h = gate_by_local_mad(
                            seg, sampling_rate,
                            comp="P",
                            cand_rel_idx=p_rel_idx,
                            expected_polarity=expected_polarity,
                            cfg=cfg,
                            baseline_mode="rolling",
                        )
                        
                        # SECONDARY validation: Training thresholds (if available)
                        if keep and p_training_noise_peak is not None and p_h is not None:
                            if abs(p_h) < p_training_noise_peak:
                                if verbose:
                                    print(f"[Cycle {cycle_idx}]: P wave rejected by training phase (SECONDARY, unfiltered): |p_height|={abs(p_h):.4f} < noise_peak={p_training_noise_peak:.4f}")
                                keep = False
                    
                    if keep:
                        # Use original detected position for timing (SNR gate only validates, doesn't refine position for P)
                        p_center_idx = p_center_idx_detected
                        p_height = p_h
                        # For unfiltered detection, peak is already in unfiltered signal (no refinement needed)
                    else:
                        p_center_idx, p_height = None, None
            
    
    if p_center_idx is None:
        # Debug: Log why P detection failed (sample every 10 cycles)
        if cycle_idx < 20 or cycle_idx % 10 == 0:
            logging.warning(f"[P_DETECT] Cycle {cycle_idx}: P peak NOT detected - p_center_idx=None, p_height=None")
        if verbose:
            print(f"[Cycle {cycle_idx}]: P peak rejected — not included in fit.")

     
    # --- S Peak ---
    # S Peak Detection
    # Start S search AFTER R peak (r_center_idx + 1) to avoid detecting R and S at same index
    s_search_start = r_center_idx + 1 if r_center_idx is not None else None
    s_center_idx, s_height = None, None
    
    if r_center_idx is None or s_end is None or s_end <= s_search_start:
        if verbose:
            print(f"[Cycle {cycle_idx}]: Cannot search for S peak — invalid R or S window.")
        if cycle_idx < 20 or cycle_idx % 10 == 0:
            logging.warning(f"[S_DETECT] Cycle {cycle_idx}: Cannot search for S peak - r_center_idx={r_center_idx}, s_end={s_end}, s_search_start={s_search_start}, sig_len={len(sig_detrended)}")
        s_center_idx, s_height = None, None
    else:
        # Exception handling for S peak detection
        try:
            s_center_idx, s_height, _ = find_peaks(
                sig_detrended,
                xs_rel_idxs,
                s_search_start,
                s_end,
                mode="min",
                verbose=verbose,
                label="S",
                cycle_idx=cycle_idx,
            )
            # Always log for cycles 21-25 to debug Q/S detection issue
            if cycle_idx < 20 or cycle_idx % 10 == 0 or (21 <= cycle_idx <= 25):
                if s_center_idx is not None:
                    logging.info(f"[S_DETECT] Cycle {cycle_idx}: S peak detected at idx={s_center_idx}, height={s_height:.4f}")
                else:
                    logging.warning(f"[S_DETECT] Cycle {cycle_idx}: S peak NOT detected - s_search_start={s_search_start}, s_end={s_end}, sig_len={len(sig_detrended)}")
        except Exception as e:
            if cycle_idx < 20 or cycle_idx % 10 == 0:
                logging.error(f"[S_DETECT_ERROR] Cycle {cycle_idx}: Exception in S peak detection: {e}")
                import traceback
                logging.error(traceback.format_exc())
            s_center_idx, s_height = None, None

    if s_center_idx is None and verbose:
        print(f"[Cycle {cycle_idx}]: S peak rejected — not included in fit.")

    # # --- T Peak ---
    # Check if precomputed peaks are available
    t_center_idx, t_height = None, None
    t_onset_idx, t_offset_idx = None, None
    
    if precomputed_peaks is not None and cycle_idx in precomputed_peaks:
        t_annotation = precomputed_peaks[cycle_idx].get('T')
        if t_annotation is not None:
            # Use precomputed T-wave annotation
            # Map from global signal indices to cycle-relative indices
            # Use "index" column for global indices, not "signal_x" (which is relative time)
            cycle_start_global = int(one_cycle["index"].iloc[0]) if "index" in one_cycle.columns else int(one_cycle["signal_x"].iloc[0])
            cycle_end_global = int(one_cycle["index"].iloc[-1]) if "index" in one_cycle.columns else int(one_cycle["signal_x"].iloc[-1])
            cycle_length = len(one_cycle)
            
            # Check if peak is within cycle boundaries
            if cycle_start_global <= t_annotation.peak_idx <= cycle_end_global:
                t_center_idx = t_annotation.peak_idx - cycle_start_global
                # Ensure index is within cycle array bounds
                if 0 <= t_center_idx < cycle_length:
                    t_height = t_annotation.peak_amplitude
                    if t_annotation.onset_idx is not None and cycle_start_global <= t_annotation.onset_idx <= cycle_end_global:
                        t_onset_idx = t_annotation.onset_idx - cycle_start_global
                        t_onset_idx = max(0, min(t_onset_idx, cycle_length - 1))
                    if t_annotation.offset_idx is not None and cycle_start_global <= t_annotation.offset_idx <= cycle_end_global:
                        t_offset_idx = t_annotation.offset_idx - cycle_start_global
                        t_offset_idx = max(0, min(t_offset_idx, cycle_length - 1))
                    
                    if verbose:
                        print(f"[Cycle {cycle_idx}]: Using precomputed T-wave at global idx {t_annotation.peak_idx} (cycle-relative: {t_center_idx})")
                else:
                    # Peak outside cycle array bounds, skip precomputed detection
                    t_center_idx = None
            else:
                # Peak outside cycle boundaries, skip precomputed detection
                t_center_idx = None
    
    # If precomputed peaks didn't provide a T-wave, use derivative-based detection
    if t_center_idx is None:
        # T wave search starts from R peak, not S end
        # Start search 100ms after R peak (bwind=100ms)
        # This is more accurate than starting from S end
        if r_center_idx is None:
            if verbose:
                print(f"[Cycle {cycle_idx}]: Cannot search for T — missing R peak.")
        else:
            n = len(sig_detrended)
        
            # T search starts 100ms after R peak (base: bwind=100ms)
            # But adjust to S wave end + 20ms if that's later
            # This ensures search starts AFTER QRS complex ends, avoiding ST segment
            t_start_offset_ms = 100.0
            t_start_from_r = r_center_idx + int(round(t_start_offset_ms * sampling_rate / 1000.0))
            
            # CRITICAL: Use QRS end boundary (computed early) instead of S peak position
            # This ensures T detection works even when S peak is not detected
            if qrs_end_idx_early is not None:
                kdis_ms = 20.0  # 20ms margin (kdis parameter)
                t_start_from_qrs_end = qrs_end_idx_early + int(round(kdis_ms * sampling_rate / 1000.0))
                t_start_idx = max(t_start_from_r, t_start_from_qrs_end)  # Use later of the two
                if verbose:
                    print(f"[Cycle {cycle_idx}]: T search start adjusted: R+100ms={t_start_from_r}, QRS_end+20ms={t_start_from_qrs_end}, using={t_start_idx} (S peak={s_center_idx})")
            elif s_center_idx is not None:
                # Fallback to S peak if QRS end not available
                kdis_ms = 20.0  # 20ms margin (kdis parameter)
                t_start_from_s = s_center_idx + int(round(kdis_ms * sampling_rate / 1000.0))
                t_start_idx = max(t_start_from_r, t_start_from_s)  # Use later of the two
                if verbose:
                    print(f"[Cycle {cycle_idx}]: T search start adjusted: R+100ms={t_start_from_r}, S+20ms={t_start_from_s}, using={t_start_idx}")
            else:
                # Fallback to fixed 100ms if S wave not detected
                t_start_idx = t_start_from_r
                if verbose:
                    print(f"[Cycle {cycle_idx}]: T search start: R+100ms={t_start_from_r} (S wave not detected)")
            
            t_start_idx = max(0, min(t_start_idx, n - 2))
            
            # T search end window: fixed 450ms
            t_end_offset_ms = 450.0
            t_end_idx = r_center_idx + int(round(t_end_offset_ms * sampling_rate / 1000.0))
            t_end_idx = min(n - 1, t_end_idx)
            
            # Ensure minimum window size
            min_t_window_ms = 100.0
            min_t_window_samples = int(round(min_t_window_ms * sampling_rate / 1000.0))
            if t_end_idx - t_start_idx < min_t_window_samples:
                t_end_idx = min(n - 1, t_start_idx + min_t_window_samples)

            if verbose:
                print(f"[Cycle {cycle_idx}]: T search window: start={t_start_idx}, end={t_end_idx}, size={t_end_idx - t_start_idx}, signal_len={n}")

            if t_end_idx - t_start_idx < 3:
                if verbose:
                    print(f"[Cycle {cycle_idx}]: T window too small (start={t_start_idx}, end={t_end_idx}, signal_len={n}).")
            else:
                # T-wave detection: filtered derivative + zero-crossing
                if verbose:
                    print(f"[Cycle {cycle_idx}]: Using derivative-based T wave detection")
                
                # Compute filtered derivative (dbuf: derivative buffer)
                # Use full-signal filtering to avoid edge artifacts from filtering cycle segments
                if full_derivative is not None:
                    # Extract cycle segment from full-signal derivative
                    cycle_start_global = int(one_cycle["index"].iloc[0]) if "index" in one_cycle.columns else int(one_cycle["signal_x"].iloc[0])
                    cycle_end_global = int(one_cycle["index"].iloc[-1]) if "index" in one_cycle.columns else int(one_cycle["signal_x"].iloc[-1])
                    
                    if cycle_start_global < len(full_derivative) and cycle_end_global < len(full_derivative):
                        derivative = full_derivative[cycle_start_global:cycle_end_global+1]
                        # Ensure length matches sig_detrended (should match, but handle edge cases)
                        if len(derivative) != len(sig_detrended):
                            min_len = min(len(derivative), len(sig_detrended))
                            derivative = derivative[:min_len]
                    else:
                        # Fallback to cycle-segment filtering if indices are out of bounds
                        derivative = compute_filtered_derivative(
                            sig_detrended,
                            sampling_rate,
                            lowpass_cutoff=40.0,
                        )
                else:
                    # Fallback to cycle-segment filtering (backward compatibility)
                    derivative = compute_filtered_derivative(
                        sig_detrended,
                        sampling_rate,
                        lowpass_cutoff=40.0,  # 40 Hz low-pass filter
                    )
                
                # Detect T wave using derivative-based method
                # s_end_idx should be cycle-relative (not global)
                s_end_for_t = s_center_idx if s_center_idx is not None else None
                
                t_peak_idx, t_start_boundary, t_end_boundary, t_peak_amplitude, morphology = (
                    detect_t_wave_derivative_based(
                        signal=sig_detrended,
                        derivative=derivative,
                        search_start=t_start_idx,
                        search_end=t_end_idx,
                        s_end_idx=s_end_for_t,
                        sampling_rate=sampling_rate,
                        verbose=verbose,
                        r_peak_idx=r_center_idx,
                        r_peak_value=r_height,
                    )
                )
                
                if t_peak_idx is not None:
                    # T wave detected successfully
                    t_center_idx = t_peak_idx
                    t_height = t_peak_amplitude
                    
                    if verbose:
                        print(f"[Cycle {cycle_idx}]: T wave detected via derivative-based method: "
                              f"peak={t_center_idx}, amplitude={t_height:.4f}, "
                              f"morphology={morphology}")
                else:
                    # No T wave detected
                    t_center_idx, t_height = None, None
                    if verbose:
                        print(f"[Cycle {cycle_idx}]: No T wave detected via derivative-based method")
                    
    if t_center_idx is None and verbose:
        print(f"[Cycle {cycle_idx}]: T peak rejected — not included in fit.")


    # ------------------------------------------------------
    # Step 4: Sanity Checks - Distance and Prominence
    # ------------------------------------------------------
    # Package detections
    if verbose:
        print(f"[Cycle {cycle_idx}]: DEBUG - Packaging detections: P={p_center_idx}, Q={q_center_idx}, R={r_center_idx}, S={s_center_idx}, T={t_center_idx}")
    peaks = {
        "P": (p_center_idx, p_height),
        "Q": (q_center_idx, q_height),
        "S": (s_center_idx, s_height),
        "T": (t_center_idx, t_height),
    }
    
    # Morphology-based validation for P waves (before standard validation)
    # This helps distinguish P waves from Q peaks even when Q detection fails
    # Skip morphology validation for derivative-validated P waves
    # They already have comprehensive validation built-in:
    # - Amplitude ratio checks (vs R peak)
    # - Derivative extrema validation
    # - Noise level checks (2 separate checks)
    # - Temporal constraints (P-R separation, duration limits)
    # - Zero-crossing based peak localization
    # Morphology validation can be disabled via config to match high-sensitivity detection style
    enable_morphology = getattr(cfg, "p_enable_morphology_validation", True) if cfg is not None else True
    skip_morphology_validation = not enable_morphology  # Skip if disabled in config
    
    if p_center_idx is not None and p_height is not None and not skip_morphology_validation:
        from pyhearts.processing.validation import validate_p_wave_morphology
        p_morphology_valid, p_morphology_features = validate_p_wave_morphology(
            signal=sig_detrended,
            p_center_idx=p_center_idx,
            p_height=p_height,
            sampling_rate=sampling_rate,
            r_center_idx=r_center_idx,
            verbose=verbose,
            cycle_idx=cycle_idx,
        )
        if not p_morphology_valid:
            # Morphology validation failed - reject P wave
            if verbose:
                print(f"[Cycle {cycle_idx}]: P wave rejected by morphology validation")
            p_center_idx, p_height = None, None
            peaks["P"] = (None, None)  # Update peaks dict
    elif skip_morphology_validation and verbose:
        print(f"[Cycle {cycle_idx}]: Skipping morphology validation for derivative-validated P wave")
    
    # Validate peak (pass Q center for P wave validation)
    if verbose:
        print(f"[Cycle {cycle_idx}]: DEBUG - Before validation: peaks['P']={peaks.get('P')}")
    validated = validate_peaks(
        peaks=peaks,
        r_center_idx=r_center_idx,
        r_height=r_height,
        sampling_rate=sampling_rate,
        verbose=verbose,
        cycle_idx=cycle_idx,
        cfg=cfg,
        q_center_idx_for_validation=q_center_idx_for_p_validation,  # Pass Q for P validation
    )
    if verbose:
        print(f"[Cycle {cycle_idx}]: DEBUG - After validation: validated.get('P')={validated.get('P')}")
    
    # Build final guess dict (add R, filter invalid, keep verbose logging)
    components = {**validated, "R": (r_center_idx, r_height)}
    if verbose:
        print(f"[Cycle {cycle_idx}]: DEBUG - After adding R: components.get('P')={components.get('P')}")
    guess_idxs = {}
    # Store original peak indices before Gaussian fitting (for accurate timing after fit)
    original_peak_indices = {}
    for label, (center, height) in components.items():
        if center is None or height is None or not np.isfinite(height):
            if verbose:
                print(f"[Cycle {cycle_idx}]: Excluding {label} from Gaussian fit due to missing or invalid values.")
            continue
        guess_idxs[label] = (int(center), float(height))
        original_peak_indices[label] = int(center)
    
    # Debug: Log guess_idxs status (sample every 10 cycles)
    if cycle_idx < 20 or cycle_idx % 10 == 0:
        logging.info(f"[GUESS_IDXS] Cycle {cycle_idx}: After building guess_idxs - keys={list(guess_idxs.keys())}, R in guess_idxs={'R' in guess_idxs}")  # Store original peak for timing accuracy
    if verbose:
        print(f"[Cycle {cycle_idx}]: DEBUG - After building guess_idxs: guess_idxs.get('P')={guess_idxs.get('P')}, original_peak_indices.get('P')={original_peak_indices.get('P')}")

    # ------------------------------------------------------
    # Step 5: Compute Gaussian Guess features
    # ------------------------------------------------------
    if verbose:
        print(f"[Cycle {cycle_idx}]: Computing initial Gaussian featureeter guesses...")

    # Estimate standard deviations
    std_dict = compute_gauss_std(sig_detrended, guess_idxs)
    
    # Determine if using skewed Gaussian
    use_skewed = cfg.use_skewed_gaussian
    params_per_peak = 4 if use_skewed else 3

    guess_dict = {}
    for comp, (center, height) in guess_idxs.items():
        # Use previous cycle features as seeds if available, otherwise compute from detected peaks
        if has_previous_seeds and comp in previous_gauss_features:
            # Use previous cycle as seed (but keep detected center/height for accuracy)
            prev_feat = previous_gauss_features[comp]
            if len(prev_feat) >= 3:
                prev_std = prev_feat[2]
                prev_alpha = prev_feat[3] if len(prev_feat) >= 4 and use_skewed else 0.0
            else:
                # Fallback to computed std
                prev_std = std_dict.get(comp)
                prev_alpha = 0.0
            
            std_guess = max(prev_std, 0.5) if prev_std is not None else std_dict.get(comp)
            
            # CRITICAL FIX: For R, ALWAYS ensure it's included in the fit when detected
            # Priority: 1) r_std (pre-computed, most reliable), 2) prev_std/std_dict, 3) default fallback
            if comp == "R":
                if r_std is not None:
                    # Always prefer r_std for R (computed early when signal is known-good)
                    std_guess = r_std
                    if verbose:
                        print(f"[Cycle {cycle_idx}]: Using pre-computed r_std={std_guess:.2f} for R (preferred over prev_std/std_dict)")
                    if cycle_idx < 20 or cycle_idx % 10 == 0:
                        logging.info(f"[R_STD_PREFERRED] Cycle {cycle_idx}: Using pre-computed r_std={std_guess:.2f} for R (previous cycle path, ensuring R is always in fit)")
                elif std_guess is None:
                    # If r_std, prev_std, and std_dict are all None, use a reasonable default
                    # Default std should be proportional to signal characteristics
                    default_r_std = max(2.0, abs(r_height) * 0.1) if r_height is not None else 2.5
                    std_guess = default_r_std
                    if verbose:
                        print(f"[Cycle {cycle_idx}]: Using default r_std={std_guess:.2f} for R (r_std, prev_std, and std_dict all None)")
                    if cycle_idx < 20 or cycle_idx % 10 == 0:
                        logging.warning(f"[R_STD_DEFAULT] Cycle {cycle_idx}: Using default r_std={std_guess:.2f} for R (previous cycle path, all std estimates None)")
            
            if std_guess is not None:
                std_guess = max(std_guess, 0.5)
                if verbose:
                    print(f"[Cycle {cycle_idx}]: Using {comp} with previous cycle seed: center={int(round(center))}, height={float(height):.4f}, std={std_guess:.2f}")
                
                if use_skewed:
                    guess_dict[comp] = [int(round(center)), float(height), std_guess, prev_alpha]
                else:
                    guess_dict[comp] = [int(round(center)), float(height), std_guess]
            else:
                if verbose:
                    print(f"[Cycle {cycle_idx}]: Skipping {comp}. No std estimate available.")
        else:
            # No previous cycle available, compute from detected peaks
            std_guess = std_dict.get(comp)
            
            # CRITICAL FIX: For R, ALWAYS ensure it's included in the fit when detected
            # Priority: 1) r_std (pre-computed, most reliable), 2) std_dict value, 3) default fallback
            if comp == "R":
                if r_std is not None:
                    # Always prefer r_std for R (computed early when signal is known-good)
                    std_guess = r_std
                    if verbose:
                        print(f"[Cycle {cycle_idx}]: Using pre-computed r_std={std_guess:.2f} for R (preferred over std_dict)")
                    if cycle_idx < 20 or cycle_idx % 10 == 0:
                        logging.info(f"[R_STD_PREFERRED] Cycle {cycle_idx}: Using pre-computed r_std={std_guess:.2f} for R (ensuring R is always in fit)")
                elif std_guess is None:
                    # If both r_std and std_dict are None, use a reasonable default based on R height
                    # Default std should be proportional to signal characteristics
                    # Use a conservative estimate: ~2-3 samples for typical R peaks
                    default_r_std = max(2.0, abs(r_height) * 0.1) if r_height is not None else 2.5
                    std_guess = default_r_std
                    if verbose:
                        print(f"[Cycle {cycle_idx}]: Using default r_std={std_guess:.2f} for R (r_std and std_dict both None)")
                    if cycle_idx < 20 or cycle_idx % 10 == 0:
                        logging.warning(f"[R_STD_DEFAULT] Cycle {cycle_idx}: Using default r_std={std_guess:.2f} for R (r_std=None, std_dict.get('R')=None)")
        
            if std_guess is not None:
                # Optional: enforce only a numerical stability floor
                std_guess = max(std_guess, 0.5)
    
                if verbose:
                    print(f"[Cycle {cycle_idx}]: Using {comp} std guess: {std_guess:.2f} samples (no clamping)")
    
                if use_skewed:
                    # Add alpha=0.0 as initial guess (symmetric)
                    guess_dict[comp] = [int(round(center)), float(height), std_guess, 0.0]
                else:
                    guess_dict[comp] = [int(round(center)), float(height), std_guess]
            else:
                if verbose:
                    print(f"[Cycle {cycle_idx}]: Skipping {comp}. No std estimate available.")
                if cycle_idx < 20 or cycle_idx % 10 == 0 and comp == "R":
                    logging.warning(f"[R_SKIPPED] Cycle {cycle_idx}: R skipped from Gaussian fit - std_guess=None, r_std={r_std}")


    # Build guess array for curve fitting
    guess_list = list(guess_dict.values())
    guess = np.array(guess_list)

    if verbose:
        print(f"[Cycle {cycle_idx}]: Initial Gaussian guess shape: {guess.shape} ({'skewed' if use_skewed else 'symmetric'})")
        print(f"[Cycle {cycle_idx}]: Components in guess_dict: {list(guess_dict.keys())}")

    # Determine valid components (keys)
    valid_components = list(guess_dict.keys())

    # Debug: Always log guess_dict status (sample every 10 cycles)
    if cycle_idx < 20 or cycle_idx % 10 == 0:
        logging.info(f"[GUESS_DICT] Cycle {cycle_idx}: guess_dict keys={list(guess_dict.keys())}, valid_components={valid_components}, R in guess_dict={'R' in guess_dict}")

    if verbose:
        print(f"[Cycle {cycle_idx}]: Valid components for fitting: {valid_components}")

    # Filter valid guesses
    valid_guess_list = [guess_dict[comp] for comp in valid_components if comp in guess_dict]

    if not valid_guess_list:
        # CRITICAL: If no valid guesses, we still need to ensure R is in peak_data
        if cycle_idx < 20 or cycle_idx % 10 == 0:
            logging.warning(f"[GUESS_DICT] Cycle {cycle_idx}: No valid guesses found! guess_dict={guess_dict}, guess_idxs={guess_idxs if 'guess_idxs' in locals() else 'N/A'}, R in guess_idxs={'R' in guess_idxs if 'guess_idxs' in locals() else 'N/A'}")
        if verbose:
            print(f"[Cycle {cycle_idx}]: No valid guesses found for curve_fit.")
        valid_guess = np.empty((0, params_per_peak))
        fitting_success = False
        # Initialize p0 and fit_func to avoid UnboundLocalError later
        p0 = np.array([])
        fit_func = gaussian_function  # Default to symmetric Gaussian
    else:
        valid_guess = np.array(valid_guess_list)
        if verbose:
            print(f"[Cycle {cycle_idx}]: Valid Gaussian guesses prepared: shape={valid_guess.shape}")

        # Compute curve_fit bounds
        if verbose:
            print(f"[Cycle {cycle_idx}]: Calculating bounds for Gaussian components...")

        bound_factor = cfg.bound_factor
        if use_skewed:
            valid_gaus_bounds = [
                calc_bounds_skewed(center, height, std, alpha, bound_factor, cfg.skew_bounds)
                for center, height, std, alpha in valid_guess
            ]
        else:
            valid_gaus_bounds = [calc_bounds(center, height, std, bound_factor) for center, height, std in valid_guess]
        lower_bounds, upper_bounds = zip(*valid_gaus_bounds)
        bounds = (np.array(lower_bounds).flatten(), np.array(upper_bounds).flatten())

        if verbose:
            print(f"[Cycle {cycle_idx}]: Bounds computed")

        # Select fitting function
        fit_func = skewed_gaussian_function if use_skewed else gaussian_function

        # Perform curve fitting
        p0 = valid_guess.flatten()
        
        # Clamp guess within bounds to avoid "x0 is infeasible" error
        # This ensures the initial guess is within bounds, especially important for
        # std values where float std_guess can exceed truncated int bounds
        epsilon = 1e-8  # small number to avoid edge
        p0 = np.clip(p0, bounds[0] + epsilon, bounds[1] - epsilon)
        
        if verbose:
            print(f"[Cycle {cycle_idx}]: Preparing to run curve_fit...")
        try:
            gaussian_features_fit, _ = curve_fit(
                fit_func,
                xs_rel_idxs,
                sig_detrended,
                p0=p0,
                bounds=bounds,
                method="trf",
                maxfev=cfg.maxfev
            )
            fitting_success = True
            if verbose:
                print(f"[Cycle {cycle_idx}]: Curve fitting succeeded.")
        except (ValueError, RuntimeError) as e:
            fitting_success = False
            gaussian_features_fit = np.full((len(p0),), np.nan)
            print(f"[Cycle {cycle_idx}]: Error - Gaussian fitting failed: {e}")


    # --------------------------------------------
    # Post-Fit: Handle fitted  features
    # --------------------------------------------
    if fitting_success:
        gaussian_features_reshape = gaussian_features_fit.reshape(-1, params_per_peak)
        if use_skewed:
            gauss_center_idxs = gaussian_features_reshape[:, 0]
            gauss_heights = gaussian_features_reshape[:, 1]
            gauss_stdevs = gaussian_features_reshape[:, 2]
            gauss_alphas = gaussian_features_reshape[:, 3]
            previous_gauss_features = {
                comp: [center, height, std, alpha]
                for comp, center, height, std, alpha in zip(
                    valid_components, gauss_center_idxs, gauss_heights, gauss_stdevs, gauss_alphas
                )
            }
        else:
            # When not using skewed, only extract first 3 columns (center, height, std)
            # Handle case where reshape might have more columns than expected
            if gaussian_features_reshape.shape[1] >= 3:
                gauss_center_idxs = gaussian_features_reshape[:, 0]
                gauss_heights = gaussian_features_reshape[:, 1]
                gauss_stdevs = gaussian_features_reshape[:, 2]
            else:
                # Fallback if shape is unexpected
                gauss_center_idxs = gaussian_features_reshape[:, 0] if gaussian_features_reshape.shape[1] >= 1 else np.array([])
                gauss_heights = gaussian_features_reshape[:, 1] if gaussian_features_reshape.shape[1] >= 2 else np.array([])
                gauss_stdevs = gaussian_features_reshape[:, 2] if gaussian_features_reshape.shape[1] >= 3 else np.array([])
            gauss_alphas = None
            previous_gauss_features = {
                comp: [center, height, std]
                for comp, center, height, std in zip(
                    valid_components, gauss_center_idxs, gauss_heights, gauss_stdevs
                )
            }
        if verbose:
            print(f"[Cycle {cycle_idx}]: Updated 'previous_gauss_features': {list(previous_gauss_features.keys())}")
        gaussian_features_to_use = gaussian_features_fit
    
    else:
        previous_gauss_features = None
        gauss_center_idxs = np.array([])
        gauss_heights = np.array([])
        gauss_stdevs = np.array([])
        gauss_alphas = None
        gaussian_features_to_use = np.full((len(p0),), np.nan)

    # Ensure flat array for use in Gaussian function
    if isinstance(gaussian_features_to_use, np.ndarray) and gaussian_features_to_use.ndim == 3:
        gaussian_features_to_use = gaussian_features_to_use.flatten()

    if verbose:
        print(f"[Cycle {cycle_idx}]: Generating fitted signal...")
    # Initialize fit to ensure it's always defined, even if fit generation fails
    fit = None
    try:
        # Only try to generate fit if we have valid features and fit_func is defined
        if 'fit_func' in locals() and fit_func is not None and 'xs_rel_idxs' in locals() and xs_rel_idxs is not None and len(xs_rel_idxs) > 0:
            if len(gaussian_features_to_use) > 0:
                fit = fit_func(xs_rel_idxs, *gaussian_features_to_use)
            else:
                # No features to fit, create empty fit
                if 'xs_rel_idxs' in locals() and xs_rel_idxs is not None and len(xs_rel_idxs) > 0:
                    fit = np.zeros_like(xs_rel_idxs)
                elif 'sig_detrended' in locals() and sig_detrended is not None and len(sig_detrended) > 0:
                    fit = np.zeros_like(sig_detrended)
                else:
                    fit = np.array([])
        else:
            # fit_func not available or xs_rel_idxs not available, create empty fit
            if 'xs_rel_idxs' in locals() and xs_rel_idxs is not None and len(xs_rel_idxs) > 0:
                fit = np.zeros_like(xs_rel_idxs)
            elif 'sig_detrended' in locals() and sig_detrended is not None and len(sig_detrended) > 0:
                fit = np.zeros_like(sig_detrended)
            else:
                fit = np.array([])
    except Exception as e:
        # If fit generation fails (e.g., NaN in gaussian_features_to_use), create empty fit
        if cycle_idx < 20 or cycle_idx % 10 == 0:
            logging.warning(f"[FIT_GEN] Cycle {cycle_idx}: fit_func failed: {e}. Creating empty fit.")
        if 'xs_rel_idxs' in locals() and xs_rel_idxs is not None and len(xs_rel_idxs) > 0:
            fit = np.zeros_like(xs_rel_idxs)
        elif 'sig_detrended' in locals() and sig_detrended is not None and len(sig_detrended) > 0:
            fit = np.zeros_like(sig_detrended)
        else:
            fit = np.array([])
    
    # Final safety check: ensure fit is always defined
    if fit is None:
        if 'xs_rel_idxs' in locals() and xs_rel_idxs is not None and len(xs_rel_idxs) > 0:
            fit = np.zeros_like(xs_rel_idxs)
        elif 'sig_detrended' in locals() and sig_detrended is not None and len(sig_detrended) > 0:
            fit = np.zeros_like(sig_detrended)
        else:
            fit = np.array([])
       
    if plot:
        plot_fit(xs_rel_idxs, sig_detrended, fit)
    if verbose:
        print(f"[Cycle {cycle_idx}]: Fit generation complete.")

    # --- Convert fitted features to arrays once ---
    centers_arr = np.atleast_1d(np.asarray(gauss_center_idxs, dtype=float))
    heights_arr = np.atleast_1d(np.asarray(gauss_heights, dtype=float))
    stdevs_arr  = np.atleast_1d(np.asarray(gauss_stdevs,  dtype=float)) if gauss_stdevs is not None else None
    
    SAMPLE_TO_MS = 1000.0 / sampling_rate
    
    gauss_idxs = {}
    
    # Debug: Always log fitting status for first few cycles (regardless of verbose)
    if cycle_idx < 3:
        logging.debug(f"[processcycle.py] After Gaussian fitting: fitting_success={fitting_success}, valid_components={valid_components}, len(centers_arr)={len(centers_arr) if 'centers_arr' in locals() else 'N/A'}")
        if not fitting_success:
            logging.warning(f"[processcycle.py] Cycle {cycle_idx}: Gaussian fitting FAILED!")
    
    if verbose:
        print(f"[Cycle {cycle_idx}]: DEBUG - After Gaussian fitting: valid_components={valid_components}")
    for i, comp in enumerate(valid_components):
        if i >= centers_arr.size:
            if verbose:
                print(f"[Cycle {cycle_idx}]: Component {comp} skipped — no Gaussian center available")
            continue
    
        # --- center index (discrete) ---
        c_val = centers_arr[i]
        gauss_center_idx = int(np.round(c_val)) if np.isfinite(c_val) else None
        
        # For timing accuracy: use original detected peak position instead of Gaussian-refined center
        # This avoids timing bias from Gaussian fitting while keeping Gaussian params for morphology
        # For P-waves especially, this improves timing accuracy
        if comp in original_peak_indices:
            center_idx = original_peak_indices[comp]  # Use original peak for timing
            if verbose and center_idx != gauss_center_idx:
                print(f"[Cycle {cycle_idx}]: {comp} timing using original peak {center_idx} (Gaussian center: {gauss_center_idx})")
        else:
            # Fallback: use Gaussian center if original not available (shouldn't happen for components in fit)
            center_idx = gauss_center_idx
        
        corrected_center_idx = center_idx  # Use original peak for timing, no post-fit refinement
    
        # map to global sample index (if available)
        global_center_idx = (
            int(xs_samples[corrected_center_idx])
            if corrected_center_idx is not None and corrected_center_idx < len(xs_samples)
            else None
        )
    
        # --- per-peak height ---
        height_i = float(heights_arr[i]) if i < heights_arr.size and np.isfinite(heights_arr[i]) else np.nan
    
        # --- per-peak σ and FWHM (samples + ms) ---
        gauss_stdev_samples = None
        gauss_fwhm_samples = None
        gauss_stdev_ms = None
        gauss_fwhm_ms = None
    
        if stdevs_arr is not None and i < stdevs_arr.size:
            s = stdevs_arr[i]  # σ from curve_fit, in samples
            if np.isfinite(s) and s > 0:
                gauss_stdev_samples = float(s)
                gauss_fwhm_samples  = float(2.0 * np.sqrt(2.0 * np.log(2.0)) * s)
                gauss_stdev_ms      = gauss_stdev_samples * SAMPLE_TO_MS
                gauss_fwhm_ms       = gauss_fwhm_samples  * SAMPLE_TO_MS
    
        # --- store both units ---
        if corrected_center_idx is not None and np.isfinite(height_i):
            gauss_idxs[comp] = {
                "global_center_idx": global_center_idx,
                "center_idx": corrected_center_idx,
                "gauss_center": float(c_val) if np.isfinite(c_val) else None,  # samples
                "gauss_height": height_i,                                      # mV
                "gauss_stdev_samples": gauss_stdev_samples,
                "gauss_fwhm_samples": gauss_fwhm_samples,
                "gauss_stdev_ms": gauss_stdev_ms,
                "gauss_fwhm_ms": gauss_fwhm_ms,
            }
            if verbose and comp == "P":
                print(f"[Cycle {cycle_idx}]: DEBUG - P stored in gauss_idxs: center_idx={corrected_center_idx}, global_center_idx={global_center_idx}")
        else:
            if verbose:
                print(f"[Cycle {cycle_idx}]: Component {comp} skipped due to invalid center/height")

    # # Build peak_data from filtered components
    def _safe_int(x):
        x = np.asarray(x)
        if x.size == 0: 
            return None
        x = x.item() if x.ndim == 0 else x.squeeze()
        return int(x) if x is not None and np.isfinite(x) else None
    
    def _safe_float(x):
        x = np.asarray(x)
        if x.size == 0:
            return None
        x = x.item() if x.ndim == 0 else x.squeeze()
        return float(x) if x is not None and np.isfinite(x) else None
    
    peak_data = {
        comp: {
            "global_center_idx": _safe_int(vals.get("global_center_idx")),
            "center_idx": _safe_int(vals.get("center_idx")),
            "gauss_center": _safe_float(vals.get("gauss_center")),
            "gauss_height": _safe_float(vals.get("gauss_height")),
            "gauss_stdev_samples": _safe_float(vals.get("gauss_stdev_samples")),
            "gauss_fwhm_samples": _safe_float(vals.get("gauss_fwhm_samples")),
            "gauss_stdev_ms": _safe_float(vals.get("gauss_stdev_ms")),
            "gauss_fwhm_ms": _safe_float(vals.get("gauss_fwhm_ms")),
            
        }
        for comp, vals in gauss_idxs.items()
        if vals is not None
    }
    
    # CRITICAL: Ensure R is ALWAYS in peak_data if it was detected, with valid std for morphological features
    # This MUST run for every cycle where R is detected, regardless of fitting success
    # IMPORTANT: R must have a valid gauss_stdev_samples so morphological features can be computed
    # This is the PRIMARY fallback - it should handle all cases where R is detected
    if r_center_idx is not None and r_height is not None:
        r_center_idx_rel = r_center_idx  # Already cycle-relative
        r_global_center_idx = (
            int(xs_samples[r_center_idx_rel])
            if r_center_idx_rel is not None and r_center_idx_rel < len(xs_samples)
            else None
        )
        if r_global_center_idx is not None:
            # Determine std to use: prefer r_std, then std_dict, then default
            r_std_for_features = r_std
            if r_std_for_features is None:
                # Try to get from std_dict (computed from all components)
                r_std_for_features = std_dict.get("R") if 'std_dict' in locals() else None
            if r_std_for_features is None:
                # Use default based on R height (conservative estimate)
                r_std_for_features = max(2.0, abs(r_height) * 0.1) if r_height is not None else 2.5
            
            # Compute derived values from std
            r_stdev_ms = r_std_for_features * (1000.0 / sampling_rate) if r_std_for_features is not None else None
            r_fwhm_samples = 2.0 * np.sqrt(2.0 * np.log(2.0)) * r_std_for_features if r_std_for_features is not None else None
            r_fwhm_ms = r_fwhm_samples * (1000.0 / sampling_rate) if r_fwhm_samples is not None else None
            
            # If R is not in peak_data, add it with valid std
            if "R" not in peak_data:
                peak_data["R"] = {
                    "global_center_idx": r_global_center_idx,
                    "center_idx": r_center_idx_rel,
                    "gauss_center": float(r_center_idx_rel),  # Use detected center as gauss_center
                    "gauss_height": float(r_height),
                    "gauss_stdev_samples": float(r_std_for_features),
                    "gauss_fwhm_samples": float(r_fwhm_samples) if r_fwhm_samples is not None else None,
                    "gauss_stdev_ms": float(r_stdev_ms) if r_stdev_ms is not None else None,
                    "gauss_fwhm_ms": float(r_fwhm_ms) if r_fwhm_ms is not None else None,
                }
                if cycle_idx < 20 or cycle_idx % 10 == 0 or (21 <= cycle_idx <= 25):
                    logging.info(f"[PEAK_DATA_R_FALLBACK] Cycle {cycle_idx}: Added R to peak_data with std={r_std_for_features:.2f} (R was missing from gauss_idxs)")
            else:
                # If R is in peak_data but missing std, update it
                # Check for None or NaN (since _safe_float can return None for NaN values)
                r_current_std = peak_data["R"].get("gauss_stdev_samples")
                if r_current_std is None or (isinstance(r_current_std, float) and not np.isfinite(r_current_std)):
                    peak_data["R"]["gauss_stdev_samples"] = float(r_std_for_features)
                    peak_data["R"]["gauss_stdev_ms"] = float(r_stdev_ms) if r_stdev_ms is not None else None
                    peak_data["R"]["gauss_fwhm_samples"] = float(r_fwhm_samples) if r_fwhm_samples is not None else None
                    peak_data["R"]["gauss_fwhm_ms"] = float(r_fwhm_ms) if r_fwhm_ms is not None else None
                    if peak_data["R"].get("gauss_center") is None or (isinstance(peak_data["R"].get("gauss_center"), float) and not np.isfinite(peak_data["R"].get("gauss_center"))):
                        peak_data["R"]["gauss_center"] = float(r_center_idx_rel)
                    if peak_data["R"].get("gauss_height") is None or (isinstance(peak_data["R"].get("gauss_height"), float) and not np.isfinite(peak_data["R"].get("gauss_height"))):
                        peak_data["R"]["gauss_height"] = float(r_height)
                    if cycle_idx < 20 or cycle_idx % 10 == 0 or (21 <= cycle_idx <= 25):
                        logging.info(f"[PEAK_DATA_R_FALLBACK] Cycle {cycle_idx}: Updated R in peak_data with std={r_std_for_features:.2f} (fitting failed or std missing)")
    
    # CRITICAL: Ensure Q is always in peak_data if it was detected
    # This handles cases where Q was detected but not included in the Gaussian fit
    if "Q" not in peak_data and q_center_idx is not None and q_height is not None:
        q_center_idx_rel = q_center_idx  # Already cycle-relative
        q_global_center_idx = (
            int(xs_samples[q_center_idx_rel])
            if q_center_idx_rel is not None and q_center_idx_rel < len(xs_samples)
            else None
        )
        if q_global_center_idx is not None:
            peak_data["Q"] = {
                "global_center_idx": q_global_center_idx,
                "center_idx": q_center_idx_rel,
                "gauss_center": None,  # Not in Gaussian fit
                "gauss_height": float(q_height),
                "gauss_stdev_samples": None,
                "gauss_fwhm_samples": None,
                "gauss_stdev_ms": None,
                "gauss_fwhm_ms": None,
            }
            if cycle_idx < 20 or cycle_idx % 10 == 0 or (21 <= cycle_idx <= 25):
                logging.info(f"[PEAK_DATA_Q_FALLBACK] Cycle {cycle_idx}: Added Q to peak_data (Q was detected but not in Gaussian fit): center_idx={q_center_idx_rel}, global={q_global_center_idx}, height={q_height:.4f}")
    
    # CRITICAL: Ensure S is always in peak_data if it was detected
    # This handles cases where S was detected but not included in the Gaussian fit
    if "S" not in peak_data and s_center_idx is not None and s_height is not None:
        s_center_idx_rel = s_center_idx  # Already cycle-relative
        s_global_center_idx = (
            int(xs_samples[s_center_idx_rel])
            if s_center_idx_rel is not None and s_center_idx_rel < len(xs_samples)
            else None
        )
        if s_global_center_idx is not None:
            peak_data["S"] = {
                "global_center_idx": s_global_center_idx,
                "center_idx": s_center_idx_rel,
                "gauss_center": None,  # Not in Gaussian fit
                "gauss_height": float(s_height),
                "gauss_stdev_samples": None,
                "gauss_fwhm_samples": None,
                "gauss_stdev_ms": None,
                "gauss_fwhm_ms": None,
            }
            if cycle_idx < 20 or cycle_idx % 10 == 0 or (21 <= cycle_idx <= 25):
                logging.info(f"[PEAK_DATA_S_FALLBACK] Cycle {cycle_idx}: Added S to peak_data (S was detected but not in Gaussian fit): center_idx={s_center_idx_rel}, global={s_global_center_idx}, height={s_height:.4f}")
    
    # Log before building peak_data
    if cycle_idx < 20 or cycle_idx % 10 == 0 or (26 <= cycle_idx <= 29):
        logging.info(f"[BEFORE_PEAK_DATA_BUILD] Cycle {cycle_idx}: About to build peak_data - gauss_idxs keys={list(gauss_idxs.keys()) if gauss_idxs else 'None'}, fitting_success={fitting_success if 'fitting_success' in locals() else 'N/A'}, valid_components={valid_components if 'valid_components' in locals() else 'N/A'}")
    
    # Debug: Check if peak_data is empty after building from gauss_idxs
    if cycle_idx < 20 or cycle_idx % 10 == 0 or (26 <= cycle_idx <= 29):
        if not peak_data:
            logging.warning(f"[PEAK_DATA_EMPTY] Cycle {cycle_idx}: peak_data is EMPTY after building from gauss_idxs! gauss_idxs={gauss_idxs}, fitting_success={fitting_success if 'fitting_success' in locals() else 'N/A'}, valid_components={valid_components if 'valid_components' in locals() else 'N/A'}")
    
    # Debug: Always log peak_data status (sample every 50 cycles, and cycles 26-29)
    if cycle_idx % 50 == 0 or cycle_idx < 10 or (26 <= cycle_idx <= 29):
        logging.info(f"[PEAK_DATA] Cycle {cycle_idx}: After building from gauss_idxs - peak_data keys={list(peak_data.keys())}, gauss_idxs keys={list(gauss_idxs.keys()) if gauss_idxs else 'None'}")
        if "R" in peak_data:
            r_data = peak_data['R']
            logging.info(f"[PEAK_DATA] Cycle {cycle_idx}: R in peak_data - center_idx={r_data.get('center_idx')}, global={r_data.get('global_center_idx')}")
        else:
            logging.warning(f"[PEAK_DATA] Cycle {cycle_idx}: R NOT in peak_data after building from gauss_idxs!")
    
    # Log after R fallback
    if cycle_idx < 20 or cycle_idx % 10 == 0 or (26 <= cycle_idx <= 29):
        logging.info(f"[AFTER_R_FALLBACK] Cycle {cycle_idx}: After R fallback - R in peak_data={'R' in peak_data}, peak_data keys={list(peak_data.keys())}")
    
    if verbose:
        print(f"[Cycle {cycle_idx}]: DEBUG - After building peak_data: 'P' in peak_data={('P' in peak_data)}, peak_data.get('P')={peak_data.get('P')}")
    
    # Add P waves detected by derivative_validated method if they weren't in Gaussian fit
    # This ensures P waves are stored even if they were skipped from fitting (e.g., no std estimate)
    if "P" not in peak_data and p_center_idx is not None and p_height is not None:
        # P wave was detected but not in Gaussian fit - add it to peak_data
        p_center_idx_rel = p_center_idx  # Already cycle-relative
        if verbose:
            print(f"[Cycle {cycle_idx}]: DEBUG - Adding P to peak_data: p_center_idx={p_center_idx_rel}, xs_samples[0]={xs_samples[0] if len(xs_samples) > 0 else 'N/A'}, xs_samples[-1]={xs_samples[-1] if len(xs_samples) > 0 else 'N/A'}, len(xs_samples)={len(xs_samples)}")
        p_global_center_idx = (
            int(xs_samples[p_center_idx_rel])
            if p_center_idx_rel is not None and p_center_idx_rel < len(xs_samples)
            else None
        )
        if verbose:
            print(f"[Cycle {cycle_idx}]: DEBUG - xs_samples[{p_center_idx_rel}] = {p_global_center_idx}, expected = {cycle_start_global + p_center_idx_rel if p_center_idx_rel is not None else 'N/A'}")
        
        if p_global_center_idx is not None:
            peak_data["P"] = {
                "global_center_idx": p_global_center_idx,
                "center_idx": p_center_idx_rel,
                "gauss_center": None,  # Not in Gaussian fit
                "gauss_height": float(p_height),
                "gauss_stdev_samples": None,
                "gauss_fwhm_samples": None,
                "gauss_stdev_ms": None,
                "gauss_fwhm_ms": None,
            }
            
            # Add onset/offset if available from derivative_validated detection
            if p_onset_idx is not None:
                p_onset_idx_rel = p_onset_idx  # Already cycle-relative
                if 0 <= p_onset_idx_rel < len(sig_detrended):
                    peak_data["P"]["le_idx"] = _safe_int(p_onset_idx_rel)
            if p_offset_idx is not None:
                p_offset_idx_rel = p_offset_idx  # Already cycle-relative
                if 0 <= p_offset_idx_rel < len(sig_detrended):
                    peak_data["P"]["ri_idx"] = _safe_int(p_offset_idx_rel)
            
            if verbose:
                print(f"[Cycle {cycle_idx}]: DEBUG - Added P to peak_data (not in Gaussian fit): center_idx={p_center_idx_rel}, global={p_global_center_idx}")
    
    # NOTE: R fallback with std is already handled comprehensively above at line 1756-1807
    # This duplicate fallback is kept for safety but should rarely be needed
    # If it does run, ensure it also includes std values for morphological features
    if "R" not in peak_data:
        if r_center_idx is not None and r_height is not None:
            # This should not happen if the comprehensive fallback above worked
            # But if it does, use the same logic to ensure R has valid std
            r_center_idx_rel = r_center_idx
            r_global_center_idx = (
                int(xs_samples[r_center_idx_rel])
                if r_center_idx_rel is not None and r_center_idx_rel < len(xs_samples)
                else None
            )
            
            if r_global_center_idx is not None:
                # Use same std logic as comprehensive fallback
                r_std_for_features = r_std
                if r_std_for_features is None:
                    r_std_for_features = std_dict.get("R") if 'std_dict' in locals() else None
                if r_std_for_features is None:
                    r_std_for_features = max(2.0, abs(r_height) * 0.1) if r_height is not None else 2.5
                
                r_stdev_ms = r_std_for_features * (1000.0 / sampling_rate) if r_std_for_features is not None else None
                r_fwhm_samples = 2.0 * np.sqrt(2.0 * np.log(2.0)) * r_std_for_features if r_std_for_features is not None else None
                r_fwhm_ms = r_fwhm_samples * (1000.0 / sampling_rate) if r_fwhm_samples is not None else None
                
                peak_data["R"] = {
                    "global_center_idx": r_global_center_idx,
                    "center_idx": r_center_idx_rel,
                    "gauss_center": float(r_center_idx_rel),
                    "gauss_height": float(r_height),
                    "gauss_stdev_samples": float(r_std_for_features),
                    "gauss_fwhm_samples": float(r_fwhm_samples) if r_fwhm_samples is not None else None,
                    "gauss_stdev_ms": float(r_stdev_ms) if r_stdev_ms is not None else None,
                    "gauss_fwhm_ms": float(r_fwhm_ms) if r_fwhm_ms is not None else None,
                }
                if cycle_idx % 50 == 0 or cycle_idx < 10:
                    logging.warning(f"[R_FALLBACK_DUPLICATE] Cycle {cycle_idx}: Added R to peak_data in duplicate fallback (should not happen): center_idx={r_center_idx_rel}, global={r_global_center_idx}, std={r_std_for_features:.2f}")
            else:
                if cycle_idx % 50 == 0 or cycle_idx < 10:
                    logging.error(f"[R_FALLBACK] Cycle {cycle_idx}: Failed to compute r_global_center_idx - r_center_idx_rel={r_center_idx_rel}, len(xs_samples)={len(xs_samples)}")
        else:
            if cycle_idx % 50 == 0 or cycle_idx < 10:
                logging.error(f"[R_FALLBACK] Cycle {cycle_idx}: Cannot add R to peak_data - r_center_idx={r_center_idx}, r_height={r_height}")
    
    # Add precomputed T/P waves to peak_data if they weren't in Gaussian fit
    # Also preserve precomputed onset/offset indices if available
    if precomputed_peaks is not None and cycle_idx in precomputed_peaks:
        cycle_start_global = int(one_cycle["signal_x"].iloc[0]) if not one_cycle.empty else 0
        
        # Handle T-wave
        t_annotation = precomputed_peaks[cycle_idx].get('T')
        if t_annotation is not None:
            if "T" not in peak_data:
                # T-wave not in Gaussian fit, add it from precomputed
                t_center_idx_rel = t_annotation.peak_idx - cycle_start_global
                if 0 <= t_center_idx_rel < len(sig_detrended):
                    peak_data["T"] = {
                        "global_center_idx": t_annotation.peak_idx,
                        "center_idx": t_center_idx_rel,
                        "gauss_center": None,
                        "gauss_height": t_annotation.peak_amplitude,
                        "gauss_stdev_samples": None,
                        "gauss_fwhm_samples": None,
                        "gauss_stdev_ms": None,
                        "gauss_fwhm_ms": None,
                    }
            # Add onset/offset indices from precomputed annotation
            if "T" in peak_data:
                if t_annotation.onset_idx is not None:
                    t_onset_idx_rel = t_annotation.onset_idx - cycle_start_global
                    if 0 <= t_onset_idx_rel < len(sig_detrended):
                        peak_data["T"]["le_idx"] = _safe_int(t_onset_idx_rel)
                if t_annotation.offset_idx is not None:
                    t_offset_idx_rel = t_annotation.offset_idx - cycle_start_global
                    if 0 <= t_offset_idx_rel < len(sig_detrended):
                        peak_data["T"]["ri_idx"] = _safe_int(t_offset_idx_rel)
        
        # Handle P-wave
        p_annotation = precomputed_peaks[cycle_idx].get('P')
        if p_annotation is not None:
            if "P" not in peak_data:
                p_center_idx_rel = p_annotation.peak_idx - cycle_start_global
                if 0 <= p_center_idx_rel < len(sig_detrended):
                    peak_data["P"] = {
                        "global_center_idx": p_annotation.peak_idx,
                        "center_idx": p_center_idx_rel,
                        "gauss_center": None,
                        "gauss_height": p_annotation.peak_amplitude,
                        "gauss_stdev_samples": None,
                        "gauss_fwhm_samples": None,
                        "gauss_stdev_ms": None,
                        "gauss_fwhm_ms": None,
                    }
            if "P" in peak_data:
                if p_annotation.onset_idx is not None:
                    p_onset_idx_rel = p_annotation.onset_idx - cycle_start_global
                    if 0 <= p_onset_idx_rel < len(sig_detrended):
                        peak_data["P"]["le_idx"] = _safe_int(p_onset_idx_rel)
                if p_annotation.offset_idx is not None:
                    p_offset_idx_rel = p_annotation.offset_idx - cycle_start_global
                    if 0 <= p_offset_idx_rel < len(sig_detrended):
                        peak_data["P"]["ri_idx"] = _safe_int(p_offset_idx_rel)

    if plot:
        plot_labeled_peaks(xs_rel_idxs, sig_detrended, peak_data)

    #########################################################################################################
    # END FULL ESTIMATION PROCEEDURE
    #########################################################################################################

    # CRITICAL FINAL CHECK: Ensure R is in peak_data before checking if it's empty
    # This handles cases where peak_data is empty (fitting failed) but R was detected
    if r_center_idx is not None and r_height is not None:
        if "peak_data" not in locals() or not peak_data or "R" not in peak_data:
            # Initialize peak_data if needed
            if "peak_data" not in locals() or not peak_data:
                peak_data = {}
        
        # Add R with valid std for morphological features
        r_center_idx_rel = r_center_idx
        r_global_center_idx = (
            int(xs_samples[r_center_idx_rel]) if r_center_idx_rel is not None and r_center_idx_rel < len(xs_samples) else None
        )
        if r_global_center_idx is not None:
            # Use same std logic as comprehensive fallback
            r_std_for_features = r_std
            if r_std_for_features is None:
                r_std_for_features = std_dict.get("R") if 'std_dict' in locals() else None
            if r_std_for_features is None:
                r_std_for_features = max(2.0, abs(r_height) * 0.1) if r_height is not None else 2.5
            
            r_stdev_ms = r_std_for_features * (1000.0 / sampling_rate) if r_std_for_features is not None else None
            r_fwhm_samples = 2.0 * np.sqrt(2.0 * np.log(2.0)) * r_std_for_features if r_std_for_features is not None else None
            r_fwhm_ms = r_fwhm_samples * (1000.0 / sampling_rate) if r_fwhm_samples is not None else None
            
            peak_data["R"] = {
                "global_center_idx": r_global_center_idx,
                "center_idx": r_center_idx_rel,
                "gauss_center": float(r_center_idx_rel),
                "gauss_height": float(r_height),
                "gauss_stdev_samples": float(r_std_for_features),
                "gauss_fwhm_samples": float(r_fwhm_samples) if r_fwhm_samples is not None else None,
                "gauss_stdev_ms": float(r_stdev_ms) if r_stdev_ms is not None else None,
                "gauss_fwhm_ms": float(r_fwhm_ms) if r_fwhm_ms is not None else None,
            }
            if cycle_idx < 20 or cycle_idx % 10 == 0:
                logging.warning(f"[FINAL_R_FALLBACK] Cycle {cycle_idx}: Added R to peak_data in final safety check: std={r_std_for_features:.2f}")
    elif "R" in peak_data and (peak_data["R"].get("gauss_stdev_samples") is None or (isinstance(peak_data["R"].get("gauss_stdev_samples"), float) and not np.isfinite(peak_data["R"].get("gauss_stdev_samples")))):
        # R is in peak_data but missing std - update it
        r_std_for_features = r_std
        if r_std_for_features is None:
            r_std_for_features = std_dict.get("R") if 'std_dict' in locals() else None
        if r_std_for_features is None:
            r_std_for_features = max(2.0, abs(r_height) * 0.1) if r_height is not None else 2.5
        
        r_stdev_ms = r_std_for_features * (1000.0 / sampling_rate) if r_std_for_features is not None else None
        r_fwhm_samples = 2.0 * np.sqrt(2.0 * np.log(2.0)) * r_std_for_features if r_std_for_features is not None else None
        r_fwhm_ms = r_fwhm_samples * (1000.0 / sampling_rate) if r_fwhm_samples is not None else None
        
        peak_data["R"]["gauss_stdev_samples"] = float(r_std_for_features)
        peak_data["R"]["gauss_stdev_ms"] = float(r_stdev_ms) if r_stdev_ms is not None else None
        peak_data["R"]["gauss_fwhm_samples"] = float(r_fwhm_samples) if r_fwhm_samples is not None else None
        peak_data["R"]["gauss_fwhm_ms"] = float(r_fwhm_ms) if r_fwhm_ms is not None else None
        if cycle_idx < 20 or cycle_idx % 10 == 0:
            logging.warning(f"[FINAL_R_FALLBACK] Cycle {cycle_idx}: Updated R std in peak_data in final safety check: std={r_std_for_features:.2f}")

    if "peak_data" not in locals() or not peak_data:
        # CRITICAL: If peak_data is missing, we still need to assign R peak to output_dict!
        # R peak should always be detected, so add it even if peak_data is missing
        if r_center_idx is not None and r_height is not None:
            r_global_center_idx = (
                int(xs_samples[r_center_idx]) if r_center_idx is not None and r_center_idx < len(xs_samples) else None
            )
            if r_global_center_idx is not None:
                output_dict["R_global_center_idx"][cycle_idx] = r_global_center_idx
                if cycle_idx < 20 or cycle_idx % 10 == 0:
                    logging.warning(f"[EARLY_RETURN] Cycle {cycle_idx}: peak_data missing but assigned R={r_global_center_idx} before early return")
        
        if verbose:
            print(f"[Cycle {cycle_idx}]: peak_data missing or empty — skipping shape feature extraction.")
        # Initialize fit to empty array to avoid UnboundLocalError later when computing r_squared/rmse
        # This must be done before the return statement
        if 'xs_rel_idxs' in locals() and xs_rel_idxs is not None and len(xs_rel_idxs) > 0:
            fit = np.zeros_like(xs_rel_idxs)
        elif 'sig_detrended' in locals() and sig_detrended is not None and len(sig_detrended) > 0:
            fit = np.zeros_like(sig_detrended)
        else:
            fit = np.array([])
        return output_dict, previous_r_global_center_idx, previous_p_global_center_idx, sig_detrended, None

    # ==============================================================================
    # COMPUTE QRS BOUNDARIES BEFORE SHAPE FEATURE EXTRACTION
    # ==============================================================================
    # Compute QRS boundaries using derivative-based method BEFORE shape extraction
    # so that shape features use the accurate boundaries
    # CRITICAL: Compute QRS boundaries even if Q or S peaks are not detected,
    # as P and T wave detection depends on these boundaries
    if r_center_idx is not None:
        # Update QRS onset (Q left edge) using derivative-based method
        # Use Q peak if available, otherwise search from R peak
        qrs_onset_detected = None
    try:
        if "Q" in peak_data and q_center_idx is not None:
            qrs_onset_detected = detect_qrs_onset_derivative(
                signal=sig_detrended,
                q_peak_idx=q_center_idx,
                r_peak_idx=r_center_idx,
                sampling_rate=sampling_rate,
                search_window_ms=100.0,
                verbose=verbose,
                cycle_idx=cycle_idx,
            )
            peak_data["Q"]["le_idx"] = float(qrs_onset_detected)
            if verbose:
                print(f"[Cycle {cycle_idx}]: Pre-computed Q le_idx (QRS onset) = {qrs_onset_detected} (derivative-based, Q peak available)")
        else:
            # Q peak not detected, but we still need QRS onset for P wave detection
            # Search from R peak backwards
            qrs_onset_detected = detect_qrs_onset_derivative(
                signal=sig_detrended,
                q_peak_idx=None,  # No Q peak, search from R
                r_peak_idx=r_center_idx,
                sampling_rate=sampling_rate,
                search_window_ms=100.0,
                verbose=verbose,
                cycle_idx=cycle_idx,
            )
            # Store in peak_data even if Q is not detected
            if "Q" not in peak_data:
                peak_data["Q"] = {}
            peak_data["Q"]["le_idx"] = float(qrs_onset_detected)
            if verbose:
                print(f"[Cycle {cycle_idx}]: Pre-computed Q le_idx (QRS onset) = {qrs_onset_detected} (derivative-based, Q peak NOT available)")
    except Exception as e:
        if cycle_idx < 20 or cycle_idx % 10 == 0:
            logging.error(f"[QRS_ONSET_ERROR] Cycle {cycle_idx}: Exception in QRS onset detection: {e}")
            import traceback
            logging.error(traceback.format_exc())
        # Fallback: use fixed offset
        if qrs_onset_detected is None and r_center_idx is not None:
            qrs_onset_detected = max(0, r_center_idx - int(round(40 * sampling_rate / 1000.0)))
            if "Q" not in peak_data:
                peak_data["Q"] = {}
            peak_data["Q"]["le_idx"] = float(qrs_onset_detected)
            if cycle_idx < 20 or cycle_idx % 10 == 0:
                logging.warning(f"[QRS_ONSET_FALLBACK] Cycle {cycle_idx}: Using fallback QRS onset = {qrs_onset_detected}")
    
    # Update QRS end (S right edge) using derivative-based method
    # Use S peak if available, otherwise search from R peak
    qrs_end_detected = None
    try:
        if "S" in peak_data and s_center_idx is not None:
            qrs_end_detected = detect_qrs_end_derivative(
                signal=sig_detrended,
                s_peak_idx=s_center_idx,
                r_peak_idx=r_center_idx,
                sampling_rate=sampling_rate,
                search_window_ms=100.0,
                verbose=verbose,
                cycle_idx=cycle_idx,
            )
            peak_data["S"]["ri_idx"] = float(qrs_end_detected)
            if verbose:
                print(f"[Cycle {cycle_idx}]: Pre-computed S ri_idx (QRS end) = {qrs_end_detected} (derivative-based, S peak available)")
        else:
            # S peak not detected, but we still need QRS end for T wave detection
            # Search from R peak forwards
            qrs_end_detected = detect_qrs_end_derivative(
                signal=sig_detrended,
                s_peak_idx=None,  # No S peak, search from R
                r_peak_idx=r_center_idx,
                sampling_rate=sampling_rate,
                search_window_ms=100.0,
                verbose=verbose,
                cycle_idx=cycle_idx,
            )
            # Store in peak_data even if S is not detected
            if "S" not in peak_data:
                peak_data["S"] = {}
            peak_data["S"]["ri_idx"] = float(qrs_end_detected)
            if verbose:
                print(f"[Cycle {cycle_idx}]: Pre-computed S ri_idx (QRS end) = {qrs_end_detected} (derivative-based, S peak NOT available)")
    except Exception as e:
        if cycle_idx < 20 or cycle_idx % 10 == 0:
            logging.error(f"[QRS_END_ERROR] Cycle {cycle_idx}: Exception in QRS end detection: {e}")
            import traceback
            logging.error(traceback.format_exc())
        # Fallback: use fixed offset
        if qrs_end_detected is None and r_center_idx is not None:
            qrs_end_detected = min(len(sig_detrended) - 1, r_center_idx + int(round(40 * sampling_rate / 1000.0)))
            if "S" not in peak_data:
                peak_data["S"] = {}
            peak_data["S"]["ri_idx"] = float(qrs_end_detected)
            if cycle_idx < 20 or cycle_idx % 10 == 0:
                logging.warning(f"[QRS_END_FALLBACK] Cycle {cycle_idx}: Using fallback QRS end = {qrs_end_detected}")

    # ==============================================================================
    # EXTRACT SHAPE FEATURES
    # ==============================================================================
    component_labels = list(peak_data.keys())

    def extract_feature_array(peak_data, component_labels, key):
        """Extract feature array with proper type conversion to handle non-numeric values."""
        values = []
        for comp in component_labels:
            val = peak_data[comp].get(key, np.nan)
            # Convert to float, handling None, strings, and other non-numeric types
            try:
                val_float = float(val) if val is not None and not isinstance(val, str) else np.nan
            except (ValueError, TypeError):
                val_float = np.nan
            values.append(val_float)
        return np.array(values, dtype=float)

    gauss_center = extract_feature_array(peak_data, component_labels, "gauss_center")
    gauss_height = extract_feature_array(peak_data, component_labels, "gauss_height")
    gauss_stdev_samples = extract_feature_array(peak_data, component_labels, "gauss_stdev_samples")

    # Extract R height for dynamic thresholding
    r_height = peak_data["R"]["gauss_height"] if "R" in peak_data else np.nan

    if verbose:
        print(f"[Cycle {cycle_idx}]: Extracting shape features...")
        print(f"[Cycle {cycle_idx}]: Components to process: {component_labels}")

    # Compute shape features (asymmetric, threshold-based)
    # Pass pre-computed boundaries for Q and S if available
    # Q: we have le_idx (QRS onset), need to compute ri_idx
    # S: we have ri_idx (QRS end), need to compute le_idx
    precomputed_bounds = {}
    if "Q" in peak_data and "le_idx" in peak_data["Q"] and not np.isnan(peak_data["Q"]["le_idx"]):
        q_le_idx = int(peak_data["Q"]["le_idx"])
        q_center_idx_for_bounds = int(peak_data["Q"]["center_idx"]) if "center_idx" in peak_data["Q"] and not np.isnan(peak_data["Q"]["center_idx"]) else None
        if q_center_idx_for_bounds is not None:
            precomputed_bounds["Q"] = (q_le_idx, q_center_idx_for_bounds)  # (left, center) - right will be computed
    if "S" in peak_data and "ri_idx" in peak_data["S"] and not np.isnan(peak_data["S"]["ri_idx"]):
        s_ri_idx = int(peak_data["S"]["ri_idx"])
        s_center_idx_for_bounds = int(peak_data["S"]["center_idx"]) if "center_idx" in peak_data["S"] and not np.isnan(peak_data["S"]["center_idx"]) else None
        if s_center_idx_for_bounds is not None:
            precomputed_bounds["S"] = (s_center_idx_for_bounds, s_ri_idx)  # (center, right) - left will be computed
    
    # CRITICAL: Ensure fit is initialized before extract_shape_features
    if 'fit' not in locals() or fit is None:
        # Initialize fit if it wasn't created earlier
        if 'xs_rel_idxs' in locals() and xs_rel_idxs is not None and len(xs_rel_idxs) > 0:
            fit = np.zeros_like(xs_rel_idxs)
        elif 'sig_detrended' in locals() and sig_detrended is not None and len(sig_detrended) > 0:
            fit = np.zeros_like(sig_detrended)
        else:
            fit = np.array([])
        if cycle_idx < 20 or cycle_idx % 10 == 0:
            logging.warning(f"[FIT_SAFETY] Cycle {cycle_idx}: fit was not initialized, created empty fit")
    
    try:
        shape = extract_shape_features(
            signal=fit,
            gauss_centers=gauss_center,
            gauss_stdevs=gauss_stdev_samples,
            gauss_heights=gauss_height,
            component_labels=component_labels,
            r_height=r_height,
            sampling_rate=sampling_rate,
            cfg=cfg,
            verbose=verbose,
            precomputed_bounds=precomputed_bounds,  # Pass pre-computed boundaries
        )

        valid_components = shape["valid_components"]
        shape_features_array = shape.get("array", np.empty((0, 7)))  # optional, for existing code
        
        if len(valid_components) == 0:
            if verbose:
                print(f"[Cycle {cycle_idx}]: No valid components after shape extraction.")
            return output_dict, previous_r_global_center_idx, previous_p_global_center_idx, sig_detrended, previous_gauss_features
    
    except Exception as e:
        print(f"[ERROR] [Cycle {cycle_idx}]:  extract_shape_features() failed: {e}")
        shape = {"valid_components": [], "per_component": {}, "global_metrics": {}, "array": np.empty((0, 10))}
        valid_components = []
    
    shape_feature_keys = ["duration_ms", "ri_idx", "le_idx", "rise_ms", "decay_ms", "rdsm", "sharpness",
                          "max_upslope_mv_per_s", "max_downslope_mv_per_s", "slope_asymmetry"]
    
    # Add shape features to each component in peak_data
    for comp in valid_components:
        comp_dict = shape["per_component"].get(comp, {})
        for key in shape_feature_keys:
            peak_data[comp][key] = comp_dict.get(key, np.nan)
        peak_data[comp]["voltage_integral_uv_ms"] = comp_dict.get("voltage_integral_uv_ms", np.nan)
    
    # QRS boundaries are now computed BEFORE shape extraction, so they're already in peak_data
    # and were used during shape feature extraction. No need to override again.
    
    # ms equivalents (unchanged)
    for comp in valid_components:
        try:
            center_idx = peak_data[comp].get("center_idx")
            le_idx = peak_data[comp].get("le_idx")
            ri_idx = peak_data[comp].get("ri_idx")
    
            peak_data[comp]["center_ms"] = xs_samples[int(center_idx)] / sampling_rate * 1000 if center_idx is not None and not np.isnan(center_idx) and int(center_idx) < len(xs_samples) else np.nan
            peak_data[comp]["le_ms"]     = xs_samples[int(le_idx)]    / sampling_rate * 1000 if le_idx is not None and not np.isnan(le_idx) and int(le_idx) < len(xs_samples) else np.nan
            peak_data[comp]["ri_ms"]     = xs_samples[int(ri_idx)]    / sampling_rate * 1000 if ri_idx is not None and not np.isnan(ri_idx) and int(ri_idx) < len(xs_samples) else np.nan
        except Exception as e:
            print(f"[Cycle {cycle_idx}]: Failed to compute *_ms for {comp}: {e}")
            peak_data[comp]["center_ms"] = np.nan
            peak_data[comp]["le_ms"] = np.nan
            peak_data[comp]["ri_ms"] = np.nan
    
    # if plot:
    #     plot_rise_decay(xs_samples, sig, peak_data)
        
    if plot:
        plot_rise_decay(
            xs=xs_samples,
            sig=sig,
            peak_data=peak_data,
            show=True  # default; can be omitted
        )
    
    # Store results back into output_dict (same as before)
    for comp in valid_components:
        comp_vals = shape["per_component"].get(comp, {})
        for key in shape_feature_keys:
            dict_key = f"{comp}_{key}"
            if dict_key not in output_dict:
                print(f"WARNING: [Cycle {cycle_idx}]: Missing key in output_dict: {dict_key}")
                continue
            try:
                value = float(comp_vals.get(key, np.nan))
                output_dict[dict_key][cycle_idx] = None if np.isnan(value) else round(value, 5)
            except Exception as e:
                print(f"ERROR: [Cycle {cycle_idx}]: Error storing {dict_key}: {e}")

    # ==============================================================================
    # STORE VALS
    # ==============================================================================
    # CRITICAL: Ensure Q and S are in peak_data even if peaks weren't detected,
    # so that QRS boundaries (le_idx for Q, ri_idx for S) are stored in output_dict
    # This is essential for P and T wave detection which depend on QRS boundaries
    if "Q" not in peak_data:
        peak_data["Q"] = {}
    if "S" not in peak_data:
        peak_data["S"] = {}

    for comp in peak_data.keys():
        center_idx = peak_data[comp].get("center_idx", np.nan)
        le_idx     = peak_data[comp].get("le_idx", np.nan)
        ri_idx     = peak_data[comp].get("ri_idx", np.nan)
        gauss_center = peak_data[comp].get("gauss_center", np.nan)
        gauss_fwhm_samples = peak_data[comp].get("gauss_fwhm_samples", np.nan)

        # Global center idx
        if center_idx is not None and not np.isnan(center_idx) and int(center_idx) < len(xs_samples):
            global_center_idx = xs_samples[int(center_idx)]
        else:
            global_center_idx = np.nan
        output_dict[f"{comp}_global_center_idx"][cycle_idx] = global_center_idx

        # Global le and ri idx
        if le_idx is not None and not np.isnan(le_idx) and int(le_idx) < len(xs_samples):
            global_le_idx = xs_samples[int(le_idx)]
        else:
            global_le_idx = np.nan
        output_dict[f"{comp}_global_le_idx"][cycle_idx] = global_le_idx


        if ri_idx is not None and not np.isnan(ri_idx) and int(ri_idx) < len(xs_samples):
            global_ri_idx = xs_samples[int(ri_idx)]
        else:
            global_ri_idx = np.nan
        output_dict[f"{comp}_global_ri_idx"][cycle_idx] = global_ri_idx

        # --- FWHM-based boundary indices (inner width around the Gaussian peak) ---
        # These complement the derivative/threshold-based le/ri indices by providing
        # a purely morphology-derived width (similar to classical FWHM).
        fwhm_le_idx = np.nan
        fwhm_ri_idx = np.nan
        fwhm_le_ms = np.nan
        fwhm_ri_ms = np.nan
        fwhm_global_le_idx = np.nan
        fwhm_global_ri_idx = np.nan

        if (
            gauss_center is not None
            and gauss_fwhm_samples is not None
            and not np.isnan(gauss_center)
            and not np.isnan(gauss_fwhm_samples)
        ):
            half_width = float(gauss_fwhm_samples) / 2.0
            # Local (cycle-relative) indices
            left = int(round(gauss_center - half_width))
            right = int(round(gauss_center + half_width))

            # Clamp to valid range of xs_samples / sig_detrended
            left = max(0, min(left, len(xs_samples) - 1))
            right = max(0, min(right, len(xs_samples) - 1))

            if right >= left:
                fwhm_le_idx = float(left)
                fwhm_ri_idx = float(right)
                # Convert to ms using sampling_rate
                fwhm_le_ms = (left / sampling_rate) * 1000.0
                fwhm_ri_ms = (right / sampling_rate) * 1000.0

                # Map to global sample indices via xs_samples
                fwhm_global_le_idx = float(xs_samples[left])
                fwhm_global_ri_idx = float(xs_samples[right])

        # Store FWHM-based metrics into output_dict
        output_dict[f"{comp}_fwhm_le_idx"][cycle_idx] = fwhm_le_idx
        output_dict[f"{comp}_fwhm_ri_idx"][cycle_idx] = fwhm_ri_idx
        output_dict[f"{comp}_fwhm_le_ms"][cycle_idx] = fwhm_le_ms
        output_dict[f"{comp}_fwhm_ri_ms"][cycle_idx] = fwhm_ri_ms
        output_dict[f"{comp}_fwhm_global_le_idx"][cycle_idx] = fwhm_global_le_idx
        output_dict[f"{comp}_fwhm_global_ri_idx"][cycle_idx] = fwhm_global_ri_idx


        # Center voltage
        if center_idx is not None and not np.isnan(center_idx) and int(center_idx) < len(sig_detrended):
            output_dict[f"{comp}_center_voltage"][cycle_idx] = sig_detrended[int(center_idx)]
        else:
            output_dict[f"{comp}_center_voltage"][cycle_idx] = np.nan
    
        # Left voltage
        if le_idx is not None and not np.isnan(le_idx) and int(le_idx) < len(sig_detrended):
            output_dict[f"{comp}_le_voltage"][cycle_idx] = sig_detrended[int(le_idx)]
        else:
            output_dict[f"{comp}_le_voltage"][cycle_idx] = np.nan
    
        # Right voltage
        if ri_idx is not None and not np.isnan(ri_idx) and int(ri_idx) < len(sig_detrended):
            output_dict[f"{comp}_ri_voltage"][cycle_idx] = sig_detrended[int(ri_idx)]
        else:
            output_dict[f"{comp}_ri_voltage"][cycle_idx] = np.nan
    
        # Save features
        output_dict[f"{comp}_global_center_idx"][cycle_idx] = global_center_idx 
        output_dict[f"{comp}_center_idx"][cycle_idx] = center_idx
        output_dict[f"{comp}_le_idx"][cycle_idx] = le_idx
        output_dict[f"{comp}_ri_idx"][cycle_idx] = ri_idx
        output_dict[f"{comp}_center_ms"][cycle_idx] = peak_data[comp].get("center_ms", np.nan)
        output_dict[f"{comp}_le_ms"][cycle_idx] = peak_data[comp].get("le_ms", np.nan)
        output_dict[f"{comp}_ri_ms"][cycle_idx] = peak_data[comp].get("ri_ms", np.nan)
        output_dict[f"{comp}_gauss_center"][cycle_idx] = peak_data[comp].get("gauss_center", np.nan)
        output_dict[f"{comp}_gauss_height"][cycle_idx] = peak_data[comp].get("gauss_height", np.nan)
        output_dict[f"{comp}_gauss_stdev_samples"][cycle_idx] = peak_data[comp].get("gauss_stdev_samples", np.nan)
        output_dict[f"{comp}_gauss_fwhm_samples"][cycle_idx]  = peak_data[comp].get("gauss_fwhm_samples", np.nan)
        output_dict[f"{comp}_gauss_stdev_ms"][cycle_idx]      = peak_data[comp].get("gauss_stdev_ms", np.nan)
        output_dict[f"{comp}_gauss_fwhm_ms"][cycle_idx]       = peak_data[comp].get("gauss_fwhm_ms", np.nan)

        output_dict[f"{comp}_voltage_integral_uv_ms"][cycle_idx] = peak_data[comp].get("voltage_integral_uv_ms", np.nan)

        # --- Save pairwise differences (per-cycle, not per-component) ---
        diffs = shape.get("pairwise_differences")
        if diffs is None:
            # Back-compat with older return structure
            diffs = shape.get("global_metrics", {}).get("interdeflection_voltage_differences", {})

        for diff_name, diff_value in diffs.items():
            if diff_name not in output_dict:
                print(f"WARNING: [Cycle {cycle_idx}]: Missing key in output_dict: {diff_name}")
                continue
            try:
                if diff_value is None or (isinstance(diff_value, float) and np.isnan(diff_value)):
                    output_dict[diff_name][cycle_idx] = None
                else:
                    output_dict[diff_name][cycle_idx] = round(float(diff_value), 5)
            except Exception as e:
                print(f"ERROR: [Cycle {cycle_idx}]: Error storing {diff_name}: {e}")

    # ==============================================================================
    # GLOBAL METRICS: R-squared, RMSE, Intervals
    # ==============================================================================
    # CRITICAL: Ensure fit is initialized before computing metrics
    if 'fit' not in locals() or fit is None:
        # Initialize fit if it wasn't created earlier
        if 'xs_rel_idxs' in locals() and xs_rel_idxs is not None and len(xs_rel_idxs) > 0:
            fit = np.zeros_like(xs_rel_idxs)
        elif 'sig_detrended' in locals() and sig_detrended is not None and len(sig_detrended) > 0:
            fit = np.zeros_like(sig_detrended)
        else:
            fit = np.array([])
        if cycle_idx < 20 or cycle_idx % 10 == 0:
            logging.warning(f"[FIT_SAFETY] Cycle {cycle_idx}: fit was not initialized before metrics, created empty fit")
    
    output_dict["r_squared"][cycle_idx] = calc_r_squared(sig_detrended, fit)
    output_dict["rmse"][cycle_idx] = calc_rmse(sig_detrended, fit)

    if verbose:
        print(f"[Cycle {cycle_idx}]: Fit metrics stored")

    # Perform interval calculations
    INTERVAL_PEAK_KEYS = [
        "P_le_idx",
        "P_ri_idx",
        "Q_le_idx",
        "Q_ri_idx",
        "R_le_idx",
        "R_ri_idx",
        "S_le_idx",
        "S_ri_idx",
        "T_le_idx",
        "T_ri_idx",
    ]

    peak_series_dict = {key: output_dict.get(key, []) for key in INTERVAL_PEAK_KEYS}
    interval_results = calc_intervals(
        all_peak_series=peak_series_dict, cycle_idx=cycle_idx, sampling_rate=sampling_rate, window_size=3
    )
    for interval_name, value in interval_results.items():
        output_dict[interval_name][cycle_idx] = value
    
    # Extract ST segment features (elevation, slope, deviation)
    try:
        s_ri_idx = None
        t_le_idx = None
        p_ri_idx = None
        q_le_idx = None
        
        if 'S_ri_idx' in peak_series_dict and cycle_idx < len(peak_series_dict['S_ri_idx']):
            s_ri_val = peak_series_dict['S_ri_idx'][cycle_idx]
            if not np.isnan(s_ri_val):
                s_ri_idx = int(s_ri_val)
        
        if 'T_le_idx' in peak_series_dict and cycle_idx < len(peak_series_dict['T_le_idx']):
            t_le_val = peak_series_dict['T_le_idx'][cycle_idx]
            if not np.isnan(t_le_val):
                t_le_idx = int(t_le_val)
        
        if 'P_ri_idx' in peak_series_dict and cycle_idx < len(peak_series_dict['P_ri_idx']):
            p_ri_val = peak_series_dict['P_ri_idx'][cycle_idx]
            if not np.isnan(p_ri_val):
                p_ri_idx = int(p_ri_val)
        
        if 'Q_le_idx' in peak_series_dict and cycle_idx < len(peak_series_dict['Q_le_idx']):
            q_le_val = peak_series_dict['Q_le_idx'][cycle_idx]
            if not np.isnan(q_le_val):
                q_le_idx = int(q_le_val)
        
        st_features = extract_st_segment_features(
            signal=sig_detrended,
            s_ri_idx=s_ri_idx,
            t_le_idx=t_le_idx,
            p_ri_idx=p_ri_idx,
            q_le_idx=q_le_idx,
            sampling_rate=sampling_rate,
            j_point_offset_ms=60.0,
            verbose=verbose,
        )
        
        # Store ST segment features in output_dict
        for feature_name, value in st_features.items():
            if feature_name not in output_dict:
                # Initialize if not present
                output_dict[feature_name] = [np.nan] * len(output_dict.get("cycle_trend", []))
            if cycle_idx < len(output_dict[feature_name]):
                output_dict[feature_name][cycle_idx] = value
    except Exception as e:
        if verbose:
            print(f"[Cycle {cycle_idx}]: Error extracting ST segment features: {e}")
        # Ensure ST features are initialized even if extraction fails
        for feature_name in ['ST_elevation_mv', 'ST_slope_mv_per_s', 'ST_deviation_mv']:
            if feature_name not in output_dict:
                output_dict[feature_name] = [np.nan] * len(output_dict.get("cycle_trend", []))
            if cycle_idx < len(output_dict[feature_name]):
                output_dict[feature_name][cycle_idx] = np.nan

    for key in INTERVAL_PEAK_KEYS:
        idx = peak_series_dict[key][cycle_idx] if cycle_idx < len(peak_series_dict[key]) else None
        voltage_key = key.replace("_idx", "_voltage")

        if idx is not None and not np.isnan(idx):
            idx = int(idx)
            if 0 <= idx < len(sig_detrended):
                output_dict[voltage_key][cycle_idx] = sig_detrended[idx]
                continue
        output_dict[voltage_key][cycle_idx] = np.nan

    # Compute absolute sample indices
    r_center_idx = peak_data.get("R", {}).get("center_idx")
    p_center_idx = peak_data.get("P", {}).get("center_idx")
    
    # Log that we reached the output assignment section
    if cycle_idx < 20 or cycle_idx % 10 == 0 or (26 <= cycle_idx <= 29):
        logging.info(f"[BEFORE_OUTPUT_ASSIGN] Cycle {cycle_idx}: Reached output assignment section - peak_data keys={list(peak_data.keys()) if 'peak_data' in locals() and peak_data else 'peak_data not defined'}")
    
    # CRITICAL: Always log if R is missing at output assignment (critical issue)
    # This should NEVER happen silently - if we reach here without R, something went wrong
    if r_center_idx is None:
        # Always log missing R at output assignment (critical issue)
        logging.warning(f"[OUTPUT_ASSIGN_R_MISSING] Cycle {cycle_idx}: r_center_idx is None at output assignment! peak_data keys={list(peak_data.keys()) if 'peak_data' in locals() and peak_data else 'peak_data not defined'}, peak_data.get('R')={peak_data.get('R') if 'peak_data' in locals() and peak_data else 'N/A'}, failure_reason={r_detection_failure_reason if 'r_detection_failure_reason' in locals() else 'unknown'}")
    
    # Debug: Always log peak_data status for first few cycles and cycles 26-29
    if cycle_idx < 3 or (26 <= cycle_idx <= 29):
        logging.debug(f"[processcycle.py] Cycle {cycle_idx}: peak_data keys={list(peak_data.keys()) if 'peak_data' in locals() and peak_data else 'peak_data not defined'}, R_center_idx={r_center_idx}, P_center_idx={p_center_idx}")
    
    if verbose:
        print(f"[Cycle {cycle_idx}]: DEBUG - Before output_dict assignment: p_center_idx={p_center_idx}, 'P' in peak_data={('P' in peak_data)}")

    # Log before computing global indices
    if cycle_idx < 20 or cycle_idx % 10 == 0 or (26 <= cycle_idx <= 29):
        logging.info(f"[BEFORE_GLOBAL_IDX] Cycle {cycle_idx}: About to compute global indices - r_center_idx={r_center_idx}, xs_samples len={len(xs_samples) if 'xs_samples' in locals() else 'xs_samples not defined'}")

    r_global_center_idx = (
        int(xs_samples[r_center_idx]) if r_center_idx is not None and r_center_idx < len(xs_samples) else None
    )
    p_global_center_idx = (
        int(xs_samples[p_center_idx]) if p_center_idx is not None and p_center_idx < len(xs_samples) else None
    )
    
    # Log after computing global indices
    if cycle_idx < 20 or cycle_idx % 10 == 0 or (26 <= cycle_idx <= 29):
        logging.info(f"[AFTER_GLOBAL_IDX] Cycle {cycle_idx}: Computed global indices - r_global_center_idx={r_global_center_idx}, p_global_center_idx={p_global_center_idx}")
    
    if verbose:
        print(f"[Cycle {cycle_idx}]: DEBUG - After computing global indices: p_global_center_idx={p_global_center_idx}")

    # Assign values - THIS MUST ALWAYS RUN REGARDLESS OF VERBOSE
    # CRITICAL: Always log for cycles 50-60 to debug missing cycles 51-59
    if 50 <= cycle_idx <= 60:
        logging.info(f"[BEFORE_OUTPUT_DICT_ASSIGN] Cycle {cycle_idx}: About to assign to output_dict - r_center_idx={r_center_idx}, r_global_center_idx={r_global_center_idx}, xs_samples len={len(xs_samples) if 'xs_samples' in locals() else 'xs_samples not defined'}")
    elif cycle_idx < 20 or cycle_idx % 10 == 0 or (26 <= cycle_idx <= 29):
        logging.info(f"[BEFORE_OUTPUT_DICT_ASSIGN] Cycle {cycle_idx}: About to assign to output_dict - r_global_center_idx={r_global_center_idx}")
    
    for comp, val in zip(["R_global_center_idx", "P_global_center_idx"], [r_global_center_idx, p_global_center_idx]):
        output_dict[comp][cycle_idx] = val
        # CRITICAL: Always log R assignment failures (when val is None) for ALL cycles
        if comp == "R_global_center_idx":
            if val is None:
                # Always log when R_global_center_idx is None (critical issue)
                logging.warning(f"[OUTPUT_ASSIGN_R_NONE] Cycle {cycle_idx}: R_global_center_idx is None! r_center_idx={r_center_idx}, r_height={r_height if 'r_height' in locals() else 'N/A'}, xs_samples len={len(xs_samples) if 'xs_samples' in locals() else 'xs_samples not defined'}, r_center_idx_valid={r_center_idx is not None and r_center_idx < len(xs_samples) if 'xs_samples' in locals() and r_center_idx is not None else False}")
            # Log successful assignment for cycles 50-60 to debug missing cycles 51-59
            elif 50 <= cycle_idx <= 60:
                logging.info(f"[OUTPUT_ASSIGN] Cycle {cycle_idx}: Assigned R_global_center_idx[{cycle_idx}] = {val}")
            # Debug: Log assignment (sample every 50 cycles, always for R, and cycles 26-29)
            elif cycle_idx % 50 == 0 or cycle_idx < 10 or (26 <= cycle_idx <= 29):
                logging.info(f"[OUTPUT_ASSIGN] Cycle {cycle_idx}: Assigned R_global_center_idx[{cycle_idx}] = {val}")
    
    # Log after output assignment
    if cycle_idx < 20 or cycle_idx % 10 == 0 or (26 <= cycle_idx <= 29):
        logging.info(f"[AFTER_OUTPUT_ASSIGN] Cycle {cycle_idx}: Completed output assignment - output_dict['R_global_center_idx'][{cycle_idx}]={output_dict.get('R_global_center_idx', [None])[cycle_idx] if cycle_idx < len(output_dict.get('R_global_center_idx', [])) else 'index out of range'}")
        if cycle_idx < 3 and comp == "P_global_center_idx":
            logging.debug(f"[OUTPUT_ASSIGN] Assigned {comp}[{cycle_idx}] = {val}")
        if verbose and comp == "P_global_center_idx":
            print(f"[Cycle {cycle_idx}]: DEBUG - Assigned to output_dict['P_global_center_idx'][{cycle_idx}] = {val}")

    if r_global_center_idx is None:
        if verbose:
            print(f"[Cycle {cycle_idx}]:  R peak not available — RR interval set to NaN")
    if previous_r_global_center_idx is None:
        if verbose:
            print(f"[Cycle {cycle_idx}]:  Previous R peak not available — RR interval set to NaN")

    if p_global_center_idx is None:
        if verbose:
            print(f"[Cycle {cycle_idx}]:  P peak not available — PP interval set to NaN")
    if previous_p_global_center_idx is None:
        if verbose:
            print(f"[Cycle {cycle_idx}]:  Previous P peak not available — PP interval set to NaN")

    # ---- Interval calc (no self, no globals) ----
    ms_per_sample = 1000.0 / sampling_rate
    lo_rr_ms, hi_rr_ms = cfg.rr_bounds_ms
    lo_pp_ms, hi_pp_ms = (cfg.pp_bounds_ms or cfg.rr_bounds_ms)

    rr_val_ms = interval_ms(r_global_center_idx, previous_r_global_center_idx,
                            lo_rr_ms, hi_rr_ms, ms_per_sample)
    pp_val_ms = interval_ms(p_global_center_idx, previous_p_global_center_idx,
                            lo_pp_ms, hi_pp_ms, ms_per_sample)

    # Write into preallocated arrays/lists (NOT setdefault to dict)
    output_dict["RR_interval_ms"][cycle_idx] = rr_val_ms
    output_dict["PP_interval_ms"][cycle_idx] = pp_val_ms

    # Calculate QTc values (requires both QT and RR intervals)
    qt_ms = interval_results.get('QT_interval_ms', np.nan)
    if np.isfinite(qt_ms) and np.isfinite(rr_val_ms) and rr_val_ms > 0:
        from pyhearts.feature.intervals import calc_qtc_all_formulas
        qtc_results = calc_qtc_all_formulas(qt_ms, rr_val_ms)
        for qtc_name, qtc_value in qtc_results.items():
            if qtc_name in output_dict:
                output_dict[qtc_name][cycle_idx] = qtc_value
            else:
                if cycle_idx < 20 or cycle_idx % 10 == 0:
                    logging.warning(f"[QTC_MISSING_KEY] Cycle {cycle_idx}: QTc key {qtc_name} not in output_dict")
    else:
        # Set QTc to NaN if QT or RR is missing
        for qtc_name in ['QTc_Bazett_ms', 'QTc_Fridericia_ms', 'QTc_Framingham_ms']:
            if qtc_name in output_dict:
                output_dict[qtc_name][cycle_idx] = np.nan

    # ==============================================================================
    # PHYSIOLOGICAL VALIDATION
    # ==============================================================================
    # Combine all interval results for validation
    all_interval_results = {**interval_results, "RR_interval_ms": rr_val_ms, "PP_interval_ms": pp_val_ms}
    
    # Validate peak temporal ordering and intervals
    is_valid, validation_errors = validate_cycle_physiology(
        peak_data=peak_data,
        interval_results=all_interval_results,
        sampling_rate=sampling_rate,
        verbose=verbose,
        cycle_idx=cycle_idx,
    )
    
    if not is_valid:
        if verbose:
            print(f"[Cycle {cycle_idx}]: ⚠️  Physiological validation FAILED - invalidating problematic values")
            print(f"[Cycle {cycle_idx}]: DEBUG - Validation errors: {validation_errors}")
        
        # If peak ordering is violated, invalidate only the problematic peaks
        if validation_errors.get('peak_ordering'):
            if verbose:
                print(f"[Cycle {cycle_idx}]: Invalidating peaks due to ordering violations")
                print(f"[Cycle {cycle_idx}]: DEBUG - Before invalidation: P_global_center_idx={output_dict.get('P_global_center_idx', [None])[cycle_idx] if cycle_idx < len(output_dict.get('P_global_center_idx', [])) else None}")
            
            # Parse error messages to identify which specific components are problematic
            # Error format: "Temporal order violation: P (idx=147) comes after or at R (idx=147)"
            problematic_components = set()
            for error_msg in validation_errors['peak_ordering']:
                # Extract the component name from error message
                # Format: "Temporal order violation: COMPONENT (idx=...) comes after or at ..."
                if "Temporal order violation:" in error_msg:
                    parts = error_msg.split("Temporal order violation:")[1].strip()
                    # Get the first component mentioned (the one that's out of order)
                    comp_name = parts.split(" (idx=")[0].strip()
                    if comp_name in ["P", "Q", "R", "S", "T"]:
                        problematic_components.add(comp_name)
            
            # Also check if the NEXT component in the violation should be invalidated
            # For example, if P comes after R, we might want to invalidate P but not R or T
            # Only invalidate the component that's actually out of order
            # Never invalidate R (it's critical)
            problematic_components.discard("R")
            
            if len(problematic_components) == 0:
                # If we couldn't parse, fall back to invalidating all except R (conservative)
                problematic_components = {"P", "Q", "S", "T"}
                if verbose:
                    print(f"[Cycle {cycle_idx}]: Could not parse problematic components, invalidating all except R")
            
            # Only invalidate the specific problematic components
            for comp in problematic_components:
                if comp in peak_data:
                    # Set center indices and related values to NaN in output_dict
                    for key_suffix in ["_global_center_idx", "_global_le_idx", "_global_ri_idx", 
                                       "_center_voltage", "_le_voltage", "_ri_voltage"]:
                        key = f"{comp}{key_suffix}"
                        if key in output_dict:
                            if verbose:
                                print(f"[Cycle {cycle_idx}]: Setting {key} to NaN due to ordering violation")
                                if comp == "P":
                                    print(f"[Cycle {cycle_idx}]: DEBUG - INVALIDATING P PEAK! Previous value was {output_dict[key][cycle_idx]}")
                            output_dict[key][cycle_idx] = np.nan
        
        # If intervals are out of physiological range, set them to NaN
        if validation_errors.get('intervals'):
            for error_msg in validation_errors['intervals']:
                # Extract interval name from error message
                # Format: "PR_interval_ms = 600.00 ms is outside physiological range [50, 500] ms"
                if '_ms' in error_msg:
                    interval_name = error_msg.split('_ms')[0] + '_ms'
                    if interval_name in output_dict:
                        if verbose:
                            print(f"[Cycle {cycle_idx}]: Setting {interval_name} to NaN due to physiological violation")
                        output_dict[interval_name][cycle_idx] = np.nan
        
        # Log summary
        if verbose:
            total_errors = len(validation_errors.get('peak_ordering', [])) + len(validation_errors.get('intervals', []))
            print(f"[Cycle {cycle_idx}]: Total validation errors: {total_errors}")
    
    # Update previous_gauss_features to only include peaks that are still valid after validation
    # This ensures that if a peak (e.g., P) is invalidated, the next cycle won't use CASE 1
    # Check which peaks are valid by looking at output_dict (after validation)
    if previous_gauss_features is not None:
        essential_peaks = ["P", "R", "T"]
        valid_peaks_in_output = []
        for peak in essential_peaks:
            global_center_key = f"{peak}_global_center_idx"
            if global_center_key in output_dict:
                center_val = output_dict[global_center_key][cycle_idx]
                # Check if peak is valid (not None and not NaN)
                if center_val is not None and not (isinstance(center_val, float) and np.isnan(center_val)):
                    valid_peaks_in_output.append(peak)
        
        # Only keep peaks in previous_gauss_features that are still valid
        if len(valid_peaks_in_output) < len(essential_peaks):
            # Some essential peaks were invalidated - remove them from previous_gauss_features
            missing_peaks = [p for p in essential_peaks if p not in valid_peaks_in_output]
            if verbose:
                print(f"[Cycle {cycle_idx}]: Removing invalidated peaks from previous_gauss_features: {missing_peaks}")
            # Create new previous_gauss_features without invalidated peaks
            updated_previous_gauss_features = {k: v for k, v in previous_gauss_features.items() if k in valid_peaks_in_output or k not in essential_peaks}
            previous_gauss_features = updated_previous_gauss_features if len(updated_previous_gauss_features) > 0 else None
            if verbose and previous_gauss_features is None:
                print(f"[Cycle {cycle_idx}]: All essential peaks invalidated - next cycle will use CASE 2")
    
    # Update prevs for next cycle
    previous_r_global_center_idx = r_global_center_idx
    previous_p_global_center_idx = p_global_center_idx
    
    if verbose:
        final_p_value = output_dict.get("P_global_center_idx", [None])[cycle_idx] if cycle_idx < len(output_dict.get("P_global_center_idx", [])) else None
        print(f"[Cycle {cycle_idx}]: DEBUG - FINAL CHECK before return: P_global_center_idx[{cycle_idx}] = {final_p_value}")

    return (output_dict,
            previous_r_global_center_idx,
            previous_p_global_center_idx,
            sig_detrended,
            previous_gauss_features)
