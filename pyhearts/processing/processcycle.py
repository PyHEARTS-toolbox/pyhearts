from __future__ import annotations
from typing import Optional

##PyHEARTS IMPORTS
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt


# Custom imports for PyHEARTS
from pyhearts.config import ProcessCycleConfig
from pyhearts.feature import calc_intervals, interval_ms, extract_shape_features
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
from .adaptive_threshold import gate_by_adaptive_threshold_ecgpuwave_style
from .ecgpuwave_t_detection import (
    compute_filtered_derivative,
    detect_t_wave_ecgpuwave_style,
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
    next_r_global_center_idx: Optional[int] = None,
):

    cfg = cfg or ProcessCycleConfig()  # safe default
    
    if verbose or plot:
        print(f"[Cycle {cycle_idx}]: Starting process_cycle()")
        print("=" * 80)

    # Step 1: Basic Input Validation
    if one_cycle.empty or one_cycle["signal_y"].isnull().any():
        if verbose:
            reason = "empty" if one_cycle.empty else "contains NaN values"
            print(f"[Cycle {cycle_idx}]: Input cycle {reason}. Skipping this cycle.")
        return output_dict, previous_r_global_center_idx, previous_p_global_center_idx, None, previous_gauss_features

    if verbose:
        print(f"[Cycle {cycle_idx}]: Input validation passed.")

    # Step 2: Prepare Signal Data
    xs_samples = np.arange(one_cycle["index"].iloc[0], one_cycle["index"].iloc[-1] + 1)
    xs_rel_idxs = np.arange(len(xs_samples))
    sig = one_cycle["signal_y"].to_numpy()

    # Step 3: Detrend Signal
    sig_detrended, trend = detrend_signal(
        xs_rel_idxs, sig, sampling_rate=sampling_rate, window_ms=cfg.detrend_window_ms, cycle=cycle_idx, plot=plot
    )

    output_dict["cycle_trend"][cycle_idx] = trend

    if verbose:
        print(f"[Cycle {cycle_idx}]: Detrending complete and trend saved.")

    # ============================================================================================
    # CASE 1: Use previous Gaussian  features if available
    # ============================================================================================
    if previous_gauss_features is not None and isinstance(previous_gauss_features, dict) and len(previous_gauss_features) > 0:
        if verbose:
            print(f"[Cycle {cycle_idx}]: CASE 1: Previous labeled Gaussian  features found. Using for bounds.")

        bound_factor = cfg.bound_factor 
        peak_labels = list(previous_gauss_features.keys())
        feature_list = [previous_gauss_features[label] for label in peak_labels]
        
        # Determine if using skewed Gaussian
        use_skewed = cfg.use_skewed_gaussian
        params_per_peak = 4 if use_skewed else 3

        if len(feature_list) == 0:
            if verbose:
                print(f"[Cycle {cycle_idx}]: No valid previous  features. Reverting to full estimation.")
            return process_cycle(
                one_cycle,
                output_dict,
                sampling_rate,
                cycle_idx,
                previous_r_global_center_idx,
                previous_p_global_center_idx,
                None,
                verbose=verbose,
                plot=plot,
                cfg=cfg,
            )

        # Build bounds based on symmetric vs skewed
        if use_skewed:
            bounds_list = []
            for feat in feature_list:
                if len(feat) == 4:
                    center, height, std, alpha = feat
                else:
                    center, height, std = feat
                    alpha = 0.0  # default symmetric
                bounds_list.append(calc_bounds_skewed(center, height, std, alpha, bound_factor, cfg.skew_bounds))
        else:
            bounds_list = [calc_bounds(center, height, std, bound_factor) for center, height, std in feature_list]
        
        lower_bounds, upper_bounds = zip(*bounds_list)
        bounds = (np.array(lower_bounds).flatten(), np.array(upper_bounds).flatten())

        if verbose:
            print(f"[Cycle {cycle_idx}]: Running Curve Fit with {len(feature_list)} peaks ({'skewed' if use_skewed else 'symmetric'})")

        # Build guess array
        if use_skewed:
            guess_list = []
            for feat in feature_list:
                if len(feat) == 4:
                    guess_list.extend(feat)
                else:
                    guess_list.extend(list(feat) + [0.0])  # add alpha=0
            guess = np.array(guess_list)
        else:
            guess = np.array(feature_list).flatten()
        
        # Clamp guess within bounds
        epsilon = 1e-8  # small number to avoid edge
        guess = np.clip(guess, bounds[0] + epsilon, bounds[1] - epsilon)

        # Select fitting function
        fit_func = skewed_gaussian_function if use_skewed else gaussian_function

        try:
            gaussian_features_fit, _ = curve_fit(
                fit_func, xs_rel_idxs, sig_detrended, p0=guess, bounds=bounds, method="trf", maxfev=cfg.maxfev
            )
            fitting_success = True
            if verbose:
                print(f"[Cycle {cycle_idx}]: Curvefit succeeded.")
        except (ValueError, RuntimeError) as e:
            if verbose:
                print(f"[Cycle {cycle_idx}]: Error -  Gaussian fitting failed: {e}")
            gaussian_features_fit = np.full((len(feature_list) * params_per_peak,), np.nan)
            fitting_success = False

        if fitting_success:
            new_gaussian_features_reshape = gaussian_features_fit.reshape(-1, params_per_peak)
            if use_skewed:
                new_gauss_center_idxs = new_gaussian_features_reshape[:, 0]
                new_gauss_heights = new_gaussian_features_reshape[:, 1]
                new_gauss_stdevs = new_gaussian_features_reshape[:, 2]
                new_gauss_alphas = new_gaussian_features_reshape[:, 3]
                previous_gauss_features = {
                    comp: [center, height, std, alpha]
                    for comp, center, height, std, alpha in zip(
                        peak_labels, new_gauss_center_idxs, new_gauss_heights, new_gauss_stdevs, new_gauss_alphas
                    )
                }
            else:
                # When not using skewed, only extract first 3 columns (center, height, std)
                # Handle case where reshape might have more columns than expected
                if new_gaussian_features_reshape.shape[1] >= 3:
                    new_gauss_center_idxs = new_gaussian_features_reshape[:, 0]
                    new_gauss_heights = new_gaussian_features_reshape[:, 1]
                    new_gauss_stdevs = new_gaussian_features_reshape[:, 2]
                else:
                    # Fallback if shape is unexpected
                    new_gauss_center_idxs = new_gaussian_features_reshape[:, 0] if new_gaussian_features_reshape.shape[1] >= 1 else np.array([])
                    new_gauss_heights = new_gaussian_features_reshape[:, 1] if new_gaussian_features_reshape.shape[1] >= 2 else np.array([])
                    new_gauss_stdevs = new_gaussian_features_reshape[:, 2] if new_gaussian_features_reshape.shape[1] >= 3 else np.array([])
                new_gauss_alphas = None
                previous_gauss_features = {
                    comp: [center, height, std]
                    for comp, center, height, std in zip(
                        peak_labels, new_gauss_center_idxs, new_gauss_heights, new_gauss_stdevs
                    )
                }
            if verbose:
                print(f"[Cycle {cycle_idx}]: Updated previous_gauss_features: {list(previous_gauss_features.keys())}")
            gaussian_features_to_use = gaussian_features_fit
        else:
            previous_gauss_features = None
            new_gauss_center_idxs = np.array([])
            new_gauss_heights = np.array([])
            new_gauss_stdevs = np.array([])
            new_gauss_alphas = None
 
            gaussian_features_to_use = np.full((len(feature_list) * params_per_peak,), np.nan)

        if isinstance(gaussian_features_to_use, np.ndarray) and gaussian_features_to_use.ndim == 3:
            gaussian_features_to_use = gaussian_features_to_use.flatten()

        if verbose:
            print(f"[Cycle {cycle_idx}]: Generating fitted signal...")
        fit = fit_func(xs_rel_idxs, *gaussian_features_to_use)
        
        # if plot:
        #     plot_fit(xs_rel_idxs, sig_detrended, fit)
        
        if verbose:
            print(f"[Cycle {cycle_idx}]: Fit generation complete.")

        # Normalize stdevs to a 1-D float array once
        stdevs_arr = None
        if new_gauss_stdevs is not None:
            stdevs_arr = np.atleast_1d(np.asarray(new_gauss_stdevs, dtype=float))
        
        gauss_idxs = {}
        for i, comp in enumerate(peak_labels):
            if i >= len(new_gauss_center_idxs):
                if verbose:
                    print(f"[Cycle {cycle_idx}]: Component {comp} skipped — no Gaussian center available")
                continue
        
            gauss_center = new_gauss_center_idxs[i]
            center_val = new_gauss_center_idxs[i]
            gauss_center_idx = int(np.round(center_val)) if np.isfinite(center_val) else None

            # For timing accuracy: use original detected peak position instead of Gaussian-refined center
            # This avoids timing bias from Gaussian fitting while keeping Gaussian params for morphology
            # For P-waves especially, this improves timing accuracy
            # Note: original_peak_indices only available in full estimation path, not fast path with previous_gauss_features
            if 'original_peak_indices' in locals() and comp in original_peak_indices:
                center_idx = original_peak_indices[comp]  # Use original peak for timing
            else:
                # Fallback: use Gaussian center (fast path or component not in original_peak_indices)
                center_idx = gauss_center_idx

            global_center_idx = (
                int(xs_samples[center_idx]) if center_idx is not None and center_idx < len(xs_samples) else None
            )
        
            gauss_height = float(new_gauss_heights[i])
        
            # --- per-peak σ and FWHM ---
            SAMPLE_TO_MS = 1000.0 / sampling_rate
            
            # --- per-peak σ and FWHM ---
            gauss_stdev_samples = None
            gauss_fwhm_samples = None
            gauss_stdev_ms = None
            gauss_fwhm_ms = None
            
            if stdevs_arr is not None and i < stdevs_arr.size:
                s = stdevs_arr[i]  # s in samples
                if np.isfinite(s) and s > 0:
                    gauss_stdev_samples = float(s)
                    gauss_fwhm_samples = float(2.0 * np.sqrt(2.0 * np.log(2.0)) * s)
                    # Convert to ms
                    gauss_stdev_ms = gauss_stdev_samples * SAMPLE_TO_MS
                    gauss_fwhm_ms = gauss_fwhm_samples * SAMPLE_TO_MS
            
            # Store both samples and ms
            gauss_idxs[comp] = {
                "global_center_idx": global_center_idx,
                "center_idx": center_idx,
                "gauss_center": gauss_center,
                "gauss_height": gauss_height,
                "gauss_stdev_samples": gauss_stdev_samples,
                "gauss_fwhm_samples": gauss_fwhm_samples,
                "gauss_stdev_ms": gauss_stdev_ms,
                "gauss_fwhm_ms": gauss_fwhm_ms,
            }


        # Identify missing expected components
        expected_components = ["P", "Q", "R", "S", "T"]
        missing_components = [comp for comp in expected_components if comp not in gauss_idxs]
        if verbose and missing_components:
            print(f"[Cycle {cycle_idx}]: Missing components after fit: {missing_components}")

        # Directly assign gauss_idxs as peak_data (structure now matches)
        peak_data = gauss_idxs.copy()
        
        # Add precomputed T/P waves to peak_data if they weren't in Gaussian fit
        # This ensures precomputed peaks are preserved even if Gaussian fitting failed
        if precomputed_peaks is not None and cycle_idx in precomputed_peaks:
            cycle_start_global = int(one_cycle["signal_x"].iloc[0]) if not one_cycle.empty else 0
            
            # Handle T-wave
            if "T" not in peak_data:
                t_annotation = precomputed_peaks[cycle_idx].get('T')
                if t_annotation is not None:
                    # Convert global index to cycle-relative for center_idx
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
                        # Add onset/offset if available
                        if t_annotation.onset_idx is not None:
                            peak_data["T"]["le_idx"] = t_annotation.onset_idx - cycle_start_global
                        if t_annotation.offset_idx is not None:
                            peak_data["T"]["ri_idx"] = t_annotation.offset_idx - cycle_start_global
            
            # Handle P-wave
            if "P" not in peak_data:
                p_annotation = precomputed_peaks[cycle_idx].get('P')
                if p_annotation is not None:
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
                        if p_annotation.onset_idx is not None:
                            peak_data["P"]["le_idx"] = p_annotation.onset_idx - cycle_start_global
                        if p_annotation.offset_idx is not None:
                            peak_data["P"]["ri_idx"] = p_annotation.offset_idx - cycle_start_global

        # if plot:
        #     plot_labeled_peaks(
        #         xs=xs_rel_idxs,
        #         signal=sig_detrended,
        #         peak_data=peak_data,
        #         show=True  # or False if you want to control plt.show() outside
        #     )

    else:
        # ============================================================================================
        # ============================================================================================
        # ============================================================================================
        # ===========================================================================================
        # ============================================================================================
        # === CASE 2: No usable previous_gauss_features — start fresh ===
        # ============================================================================================

        if verbose:
            print(f"[Cycle {cycle_idx}]: CASE 2: No previous Gaussian  features found. Starting from scratch.")

        # ------------------------------------------------------
        # Step 1: R Peak Detection and Initial Checks
        # ------------------------------------------------------
        if len(sig_detrended) == 0 or np.isnan(sig_detrended).any():
            if verbose:
                print(f"[Cycle {cycle_idx}]: sig_detrended is invalid (empty or NaNs). Skipping this cycle.")
            return output_dict, previous_r_global_center_idx, previous_p_global_center_idx, None, None

        r_center_idx = int(np.argmax(sig_detrended))
        r_height = sig_detrended[r_center_idx]

        # ------------------------------------------------------
        # Step 2: R Peak Stdev and Dynamic Offset
        # ------------------------------------------------------
        if verbose:
            print(f"[Cycle {cycle_idx}]: Finding half-height indices for R peak...")

        r_guess = {"R": (r_center_idx, r_height)}
        r_std_dict = compute_gauss_std(sig_detrended, r_guess)
        r_std = r_std_dict.get("R")

        if r_center_idx is None:
            if verbose:
                print(f"[Cycle {cycle_idx}]: Error -  R peak not detected — skipping cycle.")
            return output_dict, previous_r_global_center_idx, previous_p_global_center_idx, None, previous_gauss_features

        if r_std is None:
            if verbose:
                print(f"[Cycle {cycle_idx}]: Could not estimate std for R peak.")

        # Dynamic offset (in samples)
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

        # ------------------------------------------------------
        # Step 3: Peak Detection - Q, P, S, T
        # ------------------------------------------------------

        # Clamp s_end to signal length
        if s_end is None or s_end >= len(sig_detrended):
            s_end = len(sig_detrended) - 100
            if verbose:
                print(f"[Cycle {cycle_idx}]: Adjusted S max index to within signal bounds: {s_end}")

        # --- Q Peak ---
        # Automatic detection: Skip Q/S detection if sampling rate < 300 Hz
        # At lower sampling rates, Q and S peaks may not be reliably detectable
        # due to insufficient temporal resolution
        MIN_SAMPLING_RATE_FOR_QS = 300.0
        
        if sampling_rate < MIN_SAMPLING_RATE_FOR_QS:
            if verbose:
                print(f"[Cycle {cycle_idx}]: Skipping Q peak detection (sampling rate {sampling_rate:.1f} Hz < {MIN_SAMPLING_RATE_FOR_QS} Hz)")
            q_center_idx, q_height = None, None
        else:
            # Ensure Q search is before R (fix temporal order violations)
            if q_start is not None and r_center_idx is not None:
                if q_start >= r_center_idx:
                    # q_start is after R, which is invalid - search before R instead
                    q_start = max(0, r_center_idx - int(round(0.1 * sampling_rate)))  # 100ms before R
                    if verbose:
                        print(f"[Cycle {cycle_idx}]: Adjusted q_start from invalid position to {q_start} (before R at {r_center_idx})")
            
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

            if q_center_idx is None and verbose:
                print(f"[Cycle {cycle_idx}]: Q peak rejected — not included in fit.")

        # --- P Peak ---
        # Check if precomputed peaks are available
        p_center_idx, p_height = None, None
        p_onset_idx, p_offset_idx = None, None
        
        if precomputed_peaks is not None and cycle_idx in precomputed_peaks:
            p_annotation = precomputed_peaks[cycle_idx].get('P')
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
                else:
                    # Peak outside cycle boundaries, skip precomputed detection
                    p_center_idx = None
        
        # If precomputed peaks didn't provide a P-wave, use standard detection
        if p_center_idx is None:
            p_peak_end_idx = q_center_idx if q_center_idx is not None else r_center_idx        
            if p_peak_end_idx is None or p_peak_end_idx < 2:
                if verbose:
                    print(f"[Cycle {cycle_idx}]: Cannot search for P peak — missing Q/R bounds.")
                p_center_idx, p_height = None, None
            else:
                pre_qrs_len = int(p_peak_end_idx)
        
            # Narrow P-wave search window: search in the last 120ms before QRS (reduced from full pre-QRS)
            # This prevents detecting noise or artifacts far from the QRS complex
            max_p_window_ms = cfg.shape_max_window_ms.get("P", 120)
            max_p_window_samples = int(round(max_p_window_ms * sampling_rate / 1000.0))
            start_idx = max(0, pre_qrs_len - max_p_window_samples)
        
            if pre_qrs_len - start_idx < 2:
                if verbose:
                    print(f"[Cycle {cycle_idx}]: P window too small (start={start_idx}, end={pre_qrs_len}).")
                p_center_idx, p_height = None, None
            else:
                # Apply band-pass filter for P-wave detection
                # This enhances P-wave visibility while preserving morphology
                # Step 3: Add fallback to unfiltered signal if band-pass fails
                p_center_idx, p_height = None, None
                
                if cfg.pwave_use_bandpass:
                    # Filter a larger segment to avoid edge artifacts, then extract the original window
                    # Edge artifacts occur when filtering short segments, so we pad the segment
                    filter_padding_ms = 50  # Extra samples on each side to avoid edge artifacts
                    filter_padding_samples = int(round(filter_padding_ms * sampling_rate / 1000.0))
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
                        edge_ignore_start = filter_padding_samples
                        edge_ignore_end = len(p_search_filtered_large) - filter_padding_samples
                        if edge_ignore_end > edge_ignore_start:
                            p_search_filtered = p_search_filtered_large[edge_ignore_start:edge_ignore_end]
                        else:
                            p_search_filtered = p_search_filtered_large
                        
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
                        
                        # If found, check SNR gate
                        # Step 5: Determine polarity from detected P-wave
                        if p_center_idx is not None:
                            # Store original detected position BEFORE SNR gate (for accurate timing)
                            p_center_idx_detected = p_center_idx
                            
                            seg = sig_detrended[start_idx:pre_qrs_len]
                            # Determine expected polarity from detected P-wave
                            p_detected_value = seg[p_center_idx - start_idx] if (p_center_idx - start_idx) < len(seg) else seg[0]
                            expected_polarity = "positive" if p_detected_value >= 0 else "negative"
                            
                            keep, rel_idx, p_h = gate_by_local_mad(
                                seg, sampling_rate,
                                comp="P",
                                cand_rel_idx=p_center_idx - start_idx,
                                expected_polarity=expected_polarity,  # Use detected polarity
                                cfg=cfg,
                                baseline_mode="rolling",
                            )
                            if keep:
                                # Use original detected position for timing (SNR gate only validates, doesn't refine position for P)
                                p_center_idx = p_center_idx_detected
                                p_height = p_h
                                
                                # Refine peak position in unfiltered signal to avoid filter phase shifts
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
                            else:
                                p_center_idx, p_height = None, None
                
                # Fallback: if band-pass failed or not enabled, try unfiltered signal
                if p_center_idx is None:
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
                        
                        keep, rel_idx, p_h = gate_by_local_mad(
                            seg, sampling_rate,
                            comp="P",
                            cand_rel_idx=p_center_idx - start_idx,
                            expected_polarity=expected_polarity,  # Use detected polarity
                            cfg=cfg,
                            baseline_mode="rolling",
                        )
                        if keep:
                            # Use original detected position for timing (SNR gate only validates, doesn't refine position for P)
                            p_center_idx = p_center_idx_detected
                            p_height = p_h
                            # For unfiltered detection, peak is already in unfiltered signal (no refinement needed)
                        else:
                            p_center_idx, p_height = None, None
                
        
        if p_center_idx is None and verbose:
            print(f"[Cycle {cycle_idx}]: P peak rejected — not included in fit.")

     
        # --- S Peak ---
        # Automatic detection: Skip S detection if sampling rate < 300 Hz
        if sampling_rate < MIN_SAMPLING_RATE_FOR_QS:
            if verbose:
                print(f"[Cycle {cycle_idx}]: Skipping S peak detection (sampling rate {sampling_rate:.1f} Hz < {MIN_SAMPLING_RATE_FOR_QS} Hz)")
            s_center_idx, s_height = None, None
        elif r_center_idx is None or s_end is None or s_end <= r_center_idx:
            if verbose:
                print(f"[Cycle {cycle_idx}]: Cannot search for S peak — invalid R or S window.")
            s_center_idx, s_height = None, None
        else:
            s_center_idx, s_height, _ = find_peaks(
                sig_detrended,
                xs_rel_idxs,
                r_center_idx,
                s_end,
                mode="min",
                verbose=verbose,
                label="S",
                cycle_idx=cycle_idx,
            )

        if s_center_idx is None and verbose and sampling_rate >= MIN_SAMPLING_RATE_FOR_QS:
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
            # ECGPUWAVE searches T waves starting from R peak, not S end
            # Start search 100ms after R peak (ECGPUWAVE's bwind=100ms)
            # This is more accurate than starting from S end
            if r_center_idx is None:
                if verbose:
                    print(f"[Cycle {cycle_idx}]: Cannot search for T — missing R peak.")
            else:
                n = len(sig_detrended)
            
                # ECGPUWAVE starts T search 100ms after R peak (bwind=100ms)
                # This avoids detecting peaks in the ST segment or QRS tail
                ecgpuwave_t_start_offset_ms = 100.0  # ECGPUWAVE's bwind parameter
                t_start_idx = r_center_idx + int(round(ecgpuwave_t_start_offset_ms * sampling_rate / 1000.0))
                t_start_idx = max(0, min(t_start_idx, n - 2))
                
                # ECGPUWAVE end window (ewind) depends on RR interval:
                # - If RR > 900ms: ewind=800ms
                # - If RR > 800ms: ewind=600ms  
                # - If RR > 600ms: ewind=450ms
                # - Otherwise: ewind=450ms
                # For simplicity, use 450ms as default (covers most cases)
                # But also check if next R is available to limit the window
                ecgpuwave_t_end_offset_ms = 450.0  # ECGPUWAVE's default ewind
                t_end_from_r = r_center_idx + int(round(ecgpuwave_t_end_offset_ms * sampling_rate / 1000.0))
                
                # If we have next R peak info, limit window to 210ms before next R (ECGPUWAVE's limit)
                # Otherwise use the fixed window
                # NOTE: Temporarily disabled next R limiting to debug - it may be causing issues
                # if next_r_global_center_idx is not None:
                #     # Convert next R to cycle-relative if needed
                #     cycle_start_global = int(one_cycle["index"].iloc[0]) if not one_cycle.empty else 0
                #     next_r_cycle_rel = next_r_global_center_idx - cycle_start_global
                #     if next_r_cycle_rel > t_start_idx:  # Only limit if next R is after search start
                #         t_end_idx = min(n - 1, next_r_cycle_rel - int(round(210.0 * sampling_rate / 1000.0)))
                #     else:
                #         t_end_idx = min(n - 1, t_end_from_r)
                # else:
                t_end_idx = min(n - 1, t_end_from_r)
                
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
                    # ECGPUWAVE-style T-wave detection: filtered derivative + zero-crossing
                    if verbose:
                        print(f"[Cycle {cycle_idx}]: Using ECGPUWAVE-style T wave detection")
                    
                    # Compute filtered derivative (like ECGPUWAVE's dbuf)
                    # Use the full detrended signal for better derivative computation
                    derivative = compute_filtered_derivative(
                        sig_detrended,
                        sampling_rate,
                        lowpass_cutoff=40.0,  # ECGPUWAVE uses 40 Hz low-pass
                    )
                    
                    # Detect T wave using ECGPUWAVE method
                    # s_end_idx should be cycle-relative (not global)
                    s_end_for_t = s_center_idx if s_center_idx is not None else None
                    
                    t_peak_idx, t_start_boundary, t_end_boundary, t_peak_amplitude, morphology = (
                        detect_t_wave_ecgpuwave_style(
                            signal=sig_detrended,
                            derivative=derivative,
                            search_start=t_start_idx,
                            search_end=t_end_idx,
                            s_end_idx=s_end_for_t,
                            sampling_rate=sampling_rate,
                            verbose=verbose,
                        )
                    )
                    
                    if t_peak_idx is not None:
                        # T wave detected successfully
                        t_center_idx = t_peak_idx
                        t_height = t_peak_amplitude
                        
                        if verbose:
                            print(f"[Cycle {cycle_idx}]: T wave detected via ECGPUWAVE method: "
                                  f"peak={t_center_idx}, amplitude={t_height:.4f}, "
                                  f"morphology={morphology}")
                    else:
                        # No T wave detected
                        t_center_idx, t_height = None, None
                        if verbose:
                            print(f"[Cycle {cycle_idx}]: No T wave detected via ECGPUWAVE method")
                        
        if t_center_idx is None and verbose:
            print(f"[Cycle {cycle_idx}]: T peak rejected — not included in fit.")


        # ------------------------------------------------------
        # Step 4: Sanity Checks - Distance and Prominence
        # ------------------------------------------------------
        # Package detections
        peaks = {
            "P": (p_center_idx, p_height),
            "Q": (q_center_idx, q_height),
            "S": (s_center_idx, s_height),
            "T": (t_center_idx, t_height),
        }
        
        # Validate peak
        validated = validate_peaks(
            peaks=peaks,
            r_center_idx=r_center_idx,
            r_height=r_height,
            sampling_rate=sampling_rate,
            verbose=verbose,
            cycle_idx=cycle_idx,
            cfg=cfg,
        )
        
        # Build final guess dict (add R, filter invalid, keep verbose logging)
        components = {**validated, "R": (r_center_idx, r_height)}
        guess_idxs = {}
        # Store original peak indices before Gaussian fitting (for accurate timing after fit)
        original_peak_indices = {}
        for label, (center, height) in components.items():
            if center is None or height is None or not np.isfinite(height):
                if verbose:
                    print(f"[Cycle {cycle_idx}]: Excluding {label} from Gaussian fit due to missing or invalid values.")
                continue
            guess_idxs[label] = (int(center), float(height))
            original_peak_indices[label] = int(center)  # Store original peak for timing accuracy

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
            std_guess = std_dict.get(comp)
        
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


        # Build guess array for curve fitting
        guess_list = list(guess_dict.values())
        guess = np.array(guess_list)

        if verbose:
            print(f"[Cycle {cycle_idx}]: Initial Gaussian guess shape: {guess.shape} ({'skewed' if use_skewed else 'symmetric'})")
            print(f"[Cycle {cycle_idx}]: Components in guess_dict: {list(guess_dict.keys())}")

        # Determine valid components (keys)
        valid_components = list(guess_dict.keys())

        if verbose:
            print(f"[Cycle {cycle_idx}]: Valid components for fitting: {valid_components}")

        # Filter valid guesses
        valid_guess_list = [guess_dict[comp] for comp in valid_components if comp in guess_dict]

        if not valid_guess_list:
            if verbose:
                print(f"[Cycle {cycle_idx}]: No valid guesses found for curve_fit.")
            valid_guess = np.empty((0, params_per_peak))
            fitting_success = False
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
        fit = fit_func(xs_rel_idxs, *gaussian_features_to_use)
       
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

    if "peak_data" not in locals() or not peak_data:
        if verbose:
            print(f"[Cycle {cycle_idx}]: peak_data missing or empty — skipping shape feature extraction.")
        return output_dict, previous_r_global_center_idx, previous_p_global_center_idx, sig_detrended, None

    # ==============================================================================
    # EXTRACT SHAPE FEATURES
    # ==============================================================================
    component_labels = list(peak_data.keys())

    def extract_feature_array(peak_data, component_labels, key):
        return np.array([peak_data[comp][key] for comp in component_labels])

    gauss_center = extract_feature_array(peak_data, component_labels, "gauss_center")
    gauss_height = extract_feature_array(peak_data, component_labels, "gauss_height")
    gauss_stdev_samples = extract_feature_array(peak_data, component_labels, "gauss_stdev_samples")

    # Extract R height for dynamic thresholding
    r_height = peak_data["R"]["gauss_height"] if "R" in peak_data else np.nan

    if verbose:
        print(f"[Cycle {cycle_idx}]: Extracting shape features...")
        print(f"[Cycle {cycle_idx}]: Components to process: {component_labels}")

    # Compute shape features (asymmetric, threshold-based)
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
        )

            
        valid_components = shape["valid_components"]
        shape_features_array = shape.get("array", np.empty((0, 7)))  # optional, for existing code
    
        if len(valid_components) == 0:
            if verbose:
                print(f"[Cycle {cycle_idx}]: No valid components after shape extraction.")
            return output_dict, previous_r_global_center_idx, previous_p_global_center_idx, sig_detrended, peak_data
    
    except Exception as e:
        print(f"[ERROR] [Cycle {cycle_idx}]:  extract_shape_features() failed: {e}")
        shape = {"valid_components": [], "per_component": {}, "global_metrics": {}, "array": np.empty((0, 7))}
        valid_components = []
    
    shape_feature_keys = ["duration_ms", "ri_idx", "le_idx", "rise_ms", "decay_ms", "rdsm", "sharpness"]
    
    # Add shape features to each component in peak_data
    for comp in valid_components:
        comp_dict = shape["per_component"].get(comp, {})
        for key in shape_feature_keys:
            peak_data[comp][key] = comp_dict.get(key, np.nan)
        peak_data[comp]["voltage_integral_uv_ms"] = comp_dict.get("voltage_integral_uv_ms", np.nan)
    
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

    r_global_center_idx = (
        int(xs_samples[r_center_idx]) if r_center_idx is not None and r_center_idx < len(xs_samples) else None
    )
    p_global_center_idx = (
        int(xs_samples[p_center_idx]) if p_center_idx is not None and p_center_idx < len(xs_samples) else None
    )

    # Assign values
    for comp, val in zip(["R_global_center_idx", "P_global_center_idx"], [r_global_center_idx, p_global_center_idx]):
        output_dict[comp][cycle_idx] = val

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
        
        # If peak ordering is violated, invalidate only the problematic peaks
        if validation_errors.get('peak_ordering'):
            problematic_components = validation_errors.get('problematic_components', set())
            if problematic_components:
                if verbose:
                    print(f"[Cycle {cycle_idx}]: Invalidating problematic peaks due to ordering violations: {problematic_components}")
                # Only invalidate the specific peaks that have ordering issues
                # Don't invalidate R (R is critical and should be preserved if detected)
                # Don't invalidate peaks that are correctly ordered (e.g., T if only P has issues)
                for comp in problematic_components:
                    if comp != "R" and comp in peak_data:  # Never invalidate R
                        # Set center indices and related values to NaN in output_dict
                        for key_suffix in ["_global_center_idx", "_global_le_idx", "_global_ri_idx", 
                                           "_center_voltage", "_le_voltage", "_ri_voltage"]:
                            key = f"{comp}{key_suffix}"
                            if key in output_dict:
                                if verbose:
                                    print(f"[Cycle {cycle_idx}]: Setting {key} to NaN due to ordering violation")
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
    
    # Update prevs for next cycle
    previous_r_global_center_idx = r_global_center_idx
    previous_p_global_center_idx = p_global_center_idx

    return (output_dict,
            previous_r_global_center_idx,
            previous_p_global_center_idx,
            sig_detrended,
            previous_gauss_features)
