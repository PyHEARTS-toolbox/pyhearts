from __future__ import annotations

##PyHEARTS IMPORTS
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# Custom imports for PyHEARTS
from pyhearts.config import ProcessCycleConfig
from pyhearts.feature import calc_intervals, interval_ms, extract_shape_features
from pyhearts.fitmetrics import calc_r_squared, calc_rmse
from pyhearts.plts import plot_fit, plot_labeled_peaks, plot_rise_decay
from .bounds import calc_bounds
from .detrend import detrend_signal
from .gaussian import compute_gauss_std, gaussian_function
from .peaks import find_peaks
from .validation import validate_peaks
from .waveletoffset import calc_wavelet_dynamic_offset
from .snrgate import gate_by_local_mad


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

        if len(feature_list) == 0:
            if verbose:
                print(f"[Cycle {cycle_idx}]: No valid previous  features. Reverting to full estimation.")
            return process_cycle(
                one_cycle,
                output_dict,
                sampling_rate,
                cycle_idx,
                filtered_r_peaks,
                previous_r_global_center_idx,
                previous_p_global_center_idx,
                None,
                verbose=verbose,
                plot=plot,
            )

        bounds = [calc_bounds(center, height, std, bound_factor) for center, height, std in feature_list]
        lower_bounds, upper_bounds = zip(*bounds)
        bounds = (np.array(lower_bounds).flatten(), np.array(upper_bounds).flatten())

        if verbose:
            print(f"[Cycle {cycle_idx}]: Running Curve Fit with {len(feature_list)} peaks")

        # Clamp guess within bounds
        epsilon = 1e-8  # small number to avoid edge
        guess = np.array(feature_list).flatten()
        guess = np.clip(guess, bounds[0] + epsilon, bounds[1] - epsilon)

        try:
            gaussian_features_fit, _ = curve_fit(
                gaussian_function, xs_rel_idxs, sig_detrended, p0=guess, bounds=bounds, method="trf", maxfev=cfg.maxfev
            )
            fitting_success = True
            if verbose:
                print(f"[Cycle {cycle_idx}]: Curvefit succeeded.")
        except (ValueError, RuntimeError) as e:
            if verbose:
                print(f"[Cycle {cycle_idx}]: Error -  Gaussian fitting failed: {e}")
            gaussian_features_fit = np.full((len(feature_list) * 3,), np.nan)
            fitting_success = False

        if fitting_success:
            new_gaussian_features_reshape = gaussian_features_fit.reshape(-1, 3)
            new_gauss_center_idxs, new_gauss_heights, new_gauss_stdevs = new_gaussian_features_reshape.T

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
 
            gaussian_features_to_use = np.full((len(feature_list) * 3,), np.nan)

        if isinstance(gaussian_features_to_use, np.ndarray) and gaussian_features_to_use.ndim == 3:
            gaussian_features_to_use = gaussian_features_to_use.flatten()

        if verbose:
            print(f"[Cycle {cycle_idx}]: Generating fitted signal...")
        fit = gaussian_function(xs_rel_idxs, *gaussian_features_to_use)
        
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
            center_idx = int(np.round(center_val)) if np.isfinite(center_val) else None

            # Detecting peak center voltage value within detected peak window 
            from math import sqrt, log

            # s = per-peak sigma in samples (already computed as new_gauss_stdevs[i])
            s = float(stdevs_arr[i]) if (stdevs_arr is not None and i < stdevs_arr.size) else float("nan")
            
            if center_idx is not None and np.isfinite(s) and s > 0:
                half = max(1, int(round(s * sqrt(2.0 * log(2.0)))))  # half-FWHM in samples
                left  = max(center_idx - half, 0)
                right = min(center_idx + half + 1, len(sig_detrended))
                window = sig_detrended[left:right]
            
                if comp in ("P", "R", "T"):                      # positive waves
                    local_ext_idx = int(np.argmax(window))
                elif comp in ("Q", "S"):                         # negative waves
                    local_ext_idx = int(np.argmin(window))
                else:                                            # fallback: strongest magnitude
                    local_ext_idx = int(np.argmax(np.abs(window)))
            
                corrected_center = left + local_ext_idx
                if corrected_center != center_idx and verbose:
                    print(f"[Cycle {cycle_idx}]: {comp} center adjusted from {center_idx} to {corrected_center}")
                center_idx = corrected_center
            else: 
                if verbose:
                    print(f"[Cycle {cycle_idx}]: {comp} σ invalid/missing; keeping center at {center_idx} (no refinement)")

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
            return None  # or return {}, [], whatever your pipeline expects

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
        p_peak_end_idx = q_center_idx if q_center_idx is not None else r_center_idx        
        if p_peak_end_idx is None or p_peak_end_idx < 2:
            if verbose:
                print(f"[Cycle {cycle_idx}]: Cannot search for P peak — missing Q/R bounds.")
            p_center_idx, p_height = None, None
        else:
            pre_qrs_len = int(p_peak_end_idx)
        
            #
            start_idx = min(pre_qrs_len - 2, int(round(10.0 * sampling_rate / 1000.0)))
        
            if pre_qrs_len - start_idx < 2:
                if verbose:
                    print(f"[Cycle {cycle_idx}]: P window too small (start={start_idx}, end={pre_qrs_len}).")
                p_center_idx, p_height = None, None
            else:
                # Candidate by extremum before QRS (assume P positive)
                p_center_idx, p_height, _ = find_peaks(
                    signal=sig_detrended,
                    xs=xs_rel_idxs,
                    start_idx=start_idx,
                    end_idx=pre_qrs_len,
                    mode="max",
                    verbose=verbose,
                    label="P",
                    cycle_idx=cycle_idx,
                )

                # --- P SNR gate ---
                if p_center_idx is not None:
                    seg = sig_detrended[start_idx:pre_qrs_len]
                    keep, rel_idx, p_h = gate_by_local_mad(
                        seg, sampling_rate,
                        comp="P",
                        cand_rel_idx=p_center_idx - start_idx,  # we already localized P
                        expected_polarity="positive",
                        cfg=cfg,
                        baseline_mode="rolling",
                    )
                    if keep:
                        p_center_idx = start_idx + rel_idx
                        p_height = p_h
                    else:
                        p_center_idx, p_height = None, None
                


        
        if p_center_idx is None and verbose:
            print(f"[Cycle {cycle_idx}]: P peak rejected — not included in fit.")

     
        # --- S Peak ---
        if r_center_idx is None or s_end is None or s_end <= r_center_idx:
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

        if s_center_idx is None and verbose:
            print(f"[Cycle {cycle_idx}]: S peak rejected — not included in fit.")

        # # --- T Peak ---
        t_center_idx, t_height = None, None
        
        t_region_start = s_center_idx if s_center_idx is not None else r_center_idx
        if t_region_start is None or t_region_start >= len(sig_detrended) - 2:
            if verbose:
                print(f"[Cycle {cycle_idx}]: Cannot search for T — missing S/R bounds.")
        else:
            n = len(sig_detrended)
        
            # 1) Lightweight guards
            gap = int(round(cfg.postQRS_refractory_window_ms * sampling_rate / 1000.0))
            
            # wavelet-derived guard (from calc_wavelet_dynamic_offset), then cap via config
            post_qrs_guard = int(offset_samples) if "offset_samples" in locals() else 0
            cap_samples    = int(round(cfg.wavelet_guard_cap_ms * sampling_rate / 1000.0))
            post_qrs_guard = max(0, min(post_qrs_guard, cap_samples))
            
            t_start_idx = min(max(int(t_region_start) + 1 + max(gap, post_qrs_guard), 0), n - 2)
            t_end_idx   = max(
                t_start_idx + 2,
                n - int(round((cfg.detrend_window_ms / 2.0) * sampling_rate / 1000.0)),
            )

            if t_end_idx - t_start_idx < 3:
                if verbose:
                    print(f"[Cycle {cycle_idx}]: T window too small (start={t_start_idx}, end={t_end_idx}).")
            else:
                # T gate
                seg = sig_detrended[t_start_idx:t_end_idx]
                keep, rel_idx, t_h = gate_by_local_mad(
                    seg, sampling_rate,
                    comp="T",
                    expected_polarity="positive",
                    cfg=cfg,
                    baseline_mode="median",
                )
                if keep:
                    t_center_idx = t_start_idx + rel_idx
                    t_height = t_h
                else:
                    t_center_idx, t_height = None, None
                        
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
        for label, (center, height) in components.items():
            if center is None or height is None or not np.isfinite(height):
                if verbose:
                    print(f"[Cycle {cycle_idx}]: Excluding {label} from Gaussian fit due to missing or invalid values.")
                continue
            guess_idxs[label] = (int(center), float(height))

        # ------------------------------------------------------
        # Step 5: Compute Gaussian Guess features
        # ------------------------------------------------------
        if verbose:
            print(f"[Cycle {cycle_idx}]: Computing initial Gaussian featureeter guesses...")

        # Estimate standard deviations
        std_dict = compute_gauss_std(sig_detrended, guess_idxs)

        guess_dict = {}
        for comp, (center, height) in guess_idxs.items():
            std_guess = std_dict.get(comp)
        
            if std_guess is not None:
                # Optional: enforce only a numerical stability floor
                std_guess = max(std_guess, 0.5)
        
                if verbose:
                    print(f"[Cycle {cycle_idx}]: Using {comp} std guess: {std_guess:.2f} samples (no clamping)")
        
                guess_dict[comp] = [int(round(center)), float(height), std_guess]
            else:
                if verbose:
                    print(f"[Cycle {cycle_idx}]: Skipping {comp}. No std estimate available.")


        # Build guess array for curve fitting
        guess_list = list(guess_dict.values())
        guess = np.array(guess_list)

        if verbose:
            print(f"[Cycle {cycle_idx}]: Initial Gaussian guess shape: {guess.shape}")
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
            valid_guess = np.empty((0, 3))
            fitting_success = False
        else:
            valid_guess = np.array(valid_guess_list)
            if verbose:
                print(f"[Cycle {cycle_idx}]: Valid Gaussian guesses prepared: shape={valid_guess.shape}")

            # Compute curve_fit bounds
            if verbose:
                print(f"[Cycle {cycle_idx}]: Calculating bounds for Gaussian components...")

            bound_factor = cfg.bound_factor 
            valid_gaus_bounds = [calc_bounds(center, height, std, bound_factor) for center, height, std in valid_guess]
            lower_bounds, upper_bounds = zip(*valid_gaus_bounds)
            bounds = (np.array(lower_bounds).flatten(), np.array(upper_bounds).flatten())

            if verbose:
                print(f"[Cycle {cycle_idx}]: Bounds computed")

            # Perform curve fitting
            p0 = valid_guess.flatten()
            
            if verbose:
                print(f"[Cycle {cycle_idx}]: Preparing to run curve_fit...")
            try:
                gaussian_features_fit, _ = curve_fit(
                    gaussian_function,
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
            gaussian_features_reshape = gaussian_features_fit.reshape(-1, 3)
            gauss_center_idxs, gauss_heights, gauss_stdevs = gaussian_features_reshape.T
        
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
            gaussian_features_to_use = np.full((len(p0),), np.nan)

        # Ensure flat array for use in Gaussian function
        if isinstance(gaussian_features_to_use, np.ndarray) and gaussian_features_to_use.ndim == 3:
            gaussian_features_to_use = gaussian_features_to_use.flatten()

        if verbose:
            print(f"[Cycle {cycle_idx}]: Generating fitted signal...")
        fit = gaussian_function(xs_rel_idxs, *gaussian_features_to_use)
       
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
            center_idx = int(np.round(c_val)) if np.isfinite(c_val) else None
            corrected_center_idx = center_idx
        
            # refine local extremum within ±10 samples if possible
            if center_idx is not None and 10 < center_idx < len(sig_detrended) - 10:
                window = sig_detrended[center_idx - 10 : center_idx + 11]
                if comp in ("P", "R", "T"):
                    local_ext_idx = int(np.argmax(window))
                elif comp in ("Q", "S"):
                    local_ext_idx = int(np.argmin(window))
                else:
                    local_ext_idx = 10
                corrected_center_idx = center_idx - 10 + local_ext_idx
                if corrected_center_idx != center_idx and verbose:
                    print(f"[Cycle {cycle_idx}]: {comp} center adjusted {center_idx} → {corrected_center_idx}")
        
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

    # Update prevs for next cycle
    previous_r_global_center_idx = r_global_center_idx
    previous_p_global_center_idx = p_global_center_idx

    return (output_dict,
            previous_r_global_center_idx,
            previous_p_global_center_idx,
            sig_detrended,
            previous_gauss_features)
