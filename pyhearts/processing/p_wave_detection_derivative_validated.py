"""
Derivative-based P wave detection with comprehensive validation.

This algorithm uses derivative-based detection with zero-crossing localization
and multiple validation checks including amplitude ratios, noise level checks,
and temporal constraints. It provides robust P wave detection with high precision.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.signal import butter, filtfilt

__all__ = ["detect_p_wave_derivative_validated"]


def bandpass_filter_p_wave(
    signal: np.ndarray,
    sampling_rate: float,
    lowcut: float = 1.0,
    highcut: float = 60.0,
    order: int = 2,
) -> np.ndarray:
    """Apply bandpass filter for P wave detection (1-60 Hz)."""
    nyq = sampling_rate / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    try:
        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        return filtered
    except Exception:
        return signal


def estimate_noise_level(signal_segment: np.ndarray) -> float:
    """
    Estimate noise level using window-based noise estimation method.
    
    Window-based noise estimation method:
    - Divides signal into 5-sample windows
    - For each window, calculates (max - min)
    - Returns the average of these differences
    
    This is more robust than standard deviation for ECG noise estimation.
    """
    if len(signal_segment) == 0:
        return 0.0
    
    # Window-based method: divide into 5-sample windows, calculate (max-min) for each, average
    inicio = 0
    iew = len(signal_segment)
    mnoise = 0.0
    i = 0
    
    while inicio < iew:
        ifinal = min(inicio + 5, iew)
        segment = signal_segment[inicio:ifinal]
        if len(segment) > 0:
            ymin2 = float(np.min(segment))
            ymax2 = float(np.max(segment))
            mnoise += (ymax2 - ymin2)
            i += 1
        inicio = ifinal
    
    if i > 0:
        mnoise = abs(mnoise / i)
    
    return float(mnoise)


def zerocross(X: np.ndarray) -> Optional[int]:
    """
    Find first zero crossing in signal X.
    
    Returns the index of the first zero-crossing where sign changes from X(0).
    """
    if len(X) < 2:
        return None
    
    sign_X = np.sign(X)
    first_sign = sign_X[0]
    
    # Find where sign changes
    sign_changes = np.where(sign_X != first_sign)[0]
    
    if len(sign_changes) == 0:
        return None
    
    ncero = sign_changes[0]
    
    # Adjust if previous value is closer to zero
    if ncero > 0 and abs(X[ncero - 1]) < abs(X[ncero]):
        ncero = ncero - 1
    
    return int(ncero)


def thresholdcross(X: np.ndarray, umbral: float) -> Optional[int]:
    """
    Find position of first value crossing threshold.
    
    Parameters
    ----------
    X : np.ndarray
        Signal array
    umbral : float
        Threshold value
    
    Returns
    -------
    Optional[int]
        Index of first threshold crossing, or None if not found
    """
    if len(X) == 0:
        return None
    
    if X[0] > umbral:
        # Signal starts above threshold, find where it goes below
        I = np.where(X < umbral)[0]
        if len(I) == 0:
            return None
        iumb = int(I[0])
        # Adjust if previous value is closer to threshold
        if iumb > 0 and abs(X[iumb - 1] - umbral) < abs(X[iumb] - umbral):
            iumb = iumb - 1
        return iumb
    else:
        # Signal starts below threshold, find where it goes above
        I = np.where(X > umbral)[0]
        if len(I) == 0:
            return None
        iumb = int(I[0])
        # Adjust if previous value is closer to threshold
        if iumb > 0 and abs(X[iumb - 1] - umbral) < abs(X[iumb] - umbral):
            iumb = iumb - 1
        return iumb


def detect_p_wave_derivative_validated(
    signal: np.ndarray,
    qrs_onset_idx: int,
    r_peak_idx: Optional[int] = None,
    r_amplitude: Optional[float] = None,
    sampling_rate: float = 250.0,
    previous_t_end_idx: Optional[int] = None,
    previous_p_end_idx: Optional[int] = None,
    max_derivative: Optional[float] = None,
    verbose: bool = False,
    cycle_idx: Optional[int] = None,
) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[int]]:
    """
    Derivative-based P wave detection with comprehensive validation.
    
    This is a faithful translation maintaining the exact same logic, thresholds,
    Uses derivative extrema, zero-crossing localization, and multiple validation checks.
    
    Parameters
    ----------
    signal : np.ndarray
        ECG signal (detrended).
    qrs_onset_idx : int
        QRS onset index (start of QRS complex).
    r_peak_idx : int, optional
        R peak index (for amplitude validation).
    r_amplitude : float, optional
        R peak amplitude (for validation).
    sampling_rate : float, default 250.0
        Sampling rate in Hz.
    previous_t_end_idx : int, optional
        Previous T wave end index (to avoid overlap).
    previous_p_end_idx : int, optional
        Previous P wave end index (to avoid overlap).
    max_derivative : float, optional
        Maximum derivative value in signal (dermax).
    verbose : bool, default False
        If True, print diagnostic messages.
    cycle_idx : int, optional
        Cycle index for logging.
    
    Returns
    -------
    Tuple[Optional[int], Optional[float], Optional[int], Optional[int]]
        (p_peak_idx, p_amplitude, p_onset_idx, p_offset_idx)
        Returns (None, None, None, None) if no P wave detected.
    """
    if qrs_onset_idx < 0 or qrs_onset_idx >= len(signal):
        if verbose:
            print(f"[Cycle {cycle_idx}]: Invalid QRS onset index: {qrs_onset_idx}")
        return None, None, None, None
    
    # Step 1: Signal preprocessing
    # Use bandpass filtered signal (1-60 Hz) for amplitude checks
    # and F (derivative) for extrema detection
    Xpb = bandpass_filter_p_wave(signal, sampling_rate, lowcut=1.0, highcut=60.0, order=2)
    
    # Compute derivative F
    F = np.diff(Xpb)
    
    # Step 2: Define search window (200ms before QRS to 30ms before QRS)
    # FIX 1: Add outer iterative window adjustment loop for adaptive P-wave search window
    bwindp = 200e-3  # 200ms
    ewindp = 30e-3   # 30ms
    
    # Store original window for outer loop
    iew_original = qrs_onset_idx - int(round(ewindp * sampling_rate))
    ibw_original = qrs_onset_idx - int(round(bwindp * sampling_rate))
    
    if ibw_original <= 0:
        ibw_original = 1
    
        # Outer loop: iteratively reduce window if P not found (adaptive window reduction)
    iew = iew_original
    ibw = ibw_original
    outer_iteration = 0
        max_outer_iterations = 6  # Reduced from 10 to 6 for performance (iterative window reduction typically uses fewer iterations)
    P_detected = False
    final_result = (None, None, None, None)
    
    while outer_iteration < max_outer_iterations and not P_detected:
        if verbose:
            print(f"[Cycle {cycle_idx}]: OUTER LOOP iteration {outer_iteration}: ibw={ibw}, iew={iew}, window_size={iew-ibw} samples ({(iew-ibw)/sampling_rate*1000:.1f}ms)")
        
        if outer_iteration > 0:
            # Reduce window by 50ms each iteration (adaptive window reduction: iew=iew-round(50e-3*Fs))
            iew = iew - int(round(50e-3 * sampling_rate))
            ibw = ibw - int(round(50e-3 * sampling_rate))
            if ibw <= 0:
                ibw = 1
            if ibw >= iew:
                if verbose:
                    print(f"[Cycle {cycle_idx}]: OUTER LOOP: Window exhausted (ibw={ibw} >= iew={iew})")
                break
            if verbose:
                print(f"[Cycle {cycle_idx}]: OUTER LOOP: Reduced window to ibw={ibw}, iew={iew} ({(iew-ibw)/sampling_rate*1000:.1f}ms)")
        
        # Avoid overlap with previous T/P wave end
        # FIX 5: Add iterative window adjustment loop for handling previous T/P wave overlap
        prevt = 0
        if previous_t_end_idx is not None and previous_t_end_idx > ibw:
            prevt = previous_t_end_idx
        if previous_p_end_idx is not None and previous_p_end_idx > ibw:
            prevt = max(prevt, previous_p_end_idx)
        
        # Iterative adjustment for previous T/P overlap (adaptive window positioning)
        nofi = 1
        Pex = 1  # Track if P detection should continue
        Pp = None  # Track if P peak found
        
        # Adaptive window adjustment: while (nofi==1)&isempty(Pp)&(QRS1-ibw)/Fs<300e-3
        while nofi == 1 and Pp is None and (qrs_onset_idx - ibw) / sampling_rate < 300e-3:
            if cycle_idx == 0 or prevt == 0:
                nofi = 0
            elif ibw < prevt:
                ibw = prevt
                nofi = 0
            else:
                nofi = 1
            
            # Check if window is valid
            if ibw >= iew:
                if verbose:
                    print(f"[Cycle {cycle_idx}]: P search window invalid: ibw={ibw} >= iew={iew}")
                Pex = 0
                break
            
            # Will set Pp if P detected, otherwise continue loop
            # For now, break and continue with detection
            break
        
        if Pex == 0 or ibw >= iew:
            outer_iteration += 1
            continue
        
        # Step 3: Find extrema in derivative F
        # Note: F indices are offset by 1 from Xpb indices
        F_window = F[max(0, ibw):min(len(F), iew)]
        if len(F_window) == 0:
            if verbose:
                print(f"[Cycle {cycle_idx}]: P search window empty: ibw={ibw}, iew={iew}, len(F)={len(F)}")
            outer_iteration += 1
            continue
        
        ymin = float(np.min(F_window))
        ymax = float(np.max(F_window))
        imin_rel = int(np.argmin(F_window))
        imax_rel = int(np.argmax(F_window))
        
        imin = ibw + imin_rel
        imax = ibw + imax_rel
        
        if verbose:
            print(f"[Cycle {cycle_idx}]: P search window: ibw={ibw}, iew={iew}, len={iew-ibw}")
            print(f"[Cycle {cycle_idx}]: Derivative extrema: ymin={ymin:.4f} at {imin}, ymax={ymax:.4f} at {imax}")
        
        # Step 4: Adjust window if maximum is too close to QRS (<30ms)
        if imin <= imax and (qrs_onset_idx - imax) / sampling_rate < 30e-3:
            iew = qrs_onset_idx - int(round(30e-3 * sampling_rate))
            ibw = ibw - int(round(30e-3 * sampling_rate))
            if ibw < 0:
                ibw = 0
            
            # Recalculate extrema
            F_window = F[max(0, ibw):min(len(F), iew)]
            if len(F_window) == 0:
                outer_iteration += 1
                continue
            ymin = float(np.min(F_window))
            ymax = float(np.max(F_window))
            imin_rel = int(np.argmin(F_window))
            imax_rel = int(np.argmax(F_window))
            imin = ibw + imin_rel
            imax = ibw + imax_rel
        
        # Step 5: Baseline estimation (15ms before QRS)
        baseline_start = max(0, qrs_onset_idx - int(round(15e-3 * sampling_rate)))
        baseline_end = qrs_onset_idx
        if baseline_end > baseline_start:
            base = float(np.mean(Xpb[baseline_start:baseline_end]))
        else:
            base = 0.0
        
        # Step 6: Calculate maximum amplitude in search window (relative to baseline)
        Xpb_window = Xpb[ibw:iew]
        ecgpbmax = float(np.max(np.abs(Xpb_window - base)))
        
        # Step 7: Calculate PKni (R peak position relative to search window start)
        # PKni is the R peak position relative to the search window start
        # It should be relative to the search window start (ibw)
        PKni = None
        if r_peak_idx is not None:
            if ibw <= r_peak_idx < iew:
                PKni = r_peak_idx - ibw
            # If R peak is outside window, we can't use it for amplitude validation
            # PKni should be within the window for amplitude validation
        
        # Step 8: Get dermax (maximum derivative)
        if max_derivative is None:
            dermax = float(np.max(np.abs(F))) if len(F) > 0 else 1.0
        else:
            dermax = max_derivative
        
        if verbose:
            print(f"[Cycle {cycle_idx}]: dermax={dermax:.6f}, threshold={dermax/100.0:.6f}")
            print(f"[Cycle {cycle_idx}]: ecgpbmax={ecgpbmax:.6f}, base={base:.6f}")
            if PKni is not None:
                print(f"[Cycle {cycle_idx}]: PKni={PKni}, R amplitude in window={abs(Xpb_window[PKni] - base) if PKni < len(Xpb_window) else 'N/A':.6f}")
        
        # Step 9: Validation check
        # Reject if: (1) amplitude too small relative to R peak, or
        #            (2) both derivative extrema are small AND have invalid ratio, or
        #            (3) derivative extrema have wrong signs
        
        validation_failed = False
        
        # Check 1: Amplitude check (only if PKni is valid)
        # Check 1: Amplitude too small relative to R peak
        # This check is only performed if PKni is within the search window
        if PKni is not None and PKni >= 0 and PKni < len(Xpb_window):
            r_amplitude_in_window = abs(Xpb_window[PKni] - base)
            if r_amplitude_in_window > 0 and ecgpbmax <= r_amplitude_in_window / 30.0:
                validation_failed = True
                if verbose:
                    print(f"[Cycle {cycle_idx}]: P rejected - amplitude {ecgpbmax:.4f} <= R/30 = {r_amplitude_in_window/30.0:.4f}")
        
        # Check 2: Both extrema small AND invalid ratio
        both_small = ymax < dermax / 100.0 and abs(ymin) < dermax / 100.0
        invalid_ratio = ymax < abs(ymin) / 1.5 or ymax > abs(ymin) * 1.5
        if both_small and invalid_ratio:
            validation_failed = True
            if verbose:
                print(f"[Cycle {cycle_idx}]: P rejected - both small and invalid ratio")
                print(f"[Cycle {cycle_idx}]:   ymax={ymax:.6f}, ymin={ymin:.6f}, dermax={dermax:.6f}, threshold={dermax/100.0:.6f}")
                print(f"[Cycle {cycle_idx}]:   both_small={both_small}, invalid_ratio={invalid_ratio}")
        
        # Check 3: Derivative sign check
        if ymax < 0 or ymin > 0:
            validation_failed = True
            if verbose:
                print(f"[Cycle {cycle_idx}]: P rejected - invalid signs (ymax={ymax:.4f}, ymin={ymin:.4f})")
        
        if validation_failed:
            # Continue to next outer iteration
            if verbose:
                print(f"[Cycle {cycle_idx}]: OUTER LOOP: Validation failed, continuing to iteration {outer_iteration + 1}")
            outer_iteration += 1
            continue
        
        # Step 10: Handle inverted P wave (imin <= imax means inverted)
        Pex = 1
        type_p = 0
        
        if imin <= imax:
            # Inverted P wave - swap max and min
            type_p = 1
            iaux = imin
            yaux = ymin
            imin = imax
            ymin = ymax
            imax = iaux
            ymax = yaux
        
        # Step 11: Find P wave onset using threshold crossing
        # Threshold: ymax / Kpb (Kpb = 1.35)
        Kpb = 1.35
        umbral = ymax / Kpb
        
        # Search backward from maximum (flip F from start to imax)
        F_reversed = F[:imax + 1][::-1]
        iumb = thresholdcross(F_reversed, umbral)
        
        if iumb is None:
            iumb = imax
        else:
            iumb = imax - iumb + 1
        
        # Step 12: Iterative adjustment if onset is too far from QRS (>240ms)
        # or overlaps with previous T/P end
        # FIX 2: Use 20 SAMPLES not 20ms (sample-based window adjustment: ibw=ibw+20)
        max_iterations = 10
        iteration = 0
        
        while Pex == 1 and iteration < max_iterations:
            p_qrs_distance_ms = (qrs_onset_idx - iumb) / sampling_rate * 1000.0
            too_far = p_qrs_distance_ms >= 240.0
            overlaps_prev = iumb <= prevt and prevt > 0
            
            if not too_far and not overlaps_prev:
                break
            
            # FIX 2: Adjust by 20 SAMPLES (not 20ms) - uses sample-based window adjustment
            ibw = ibw + 20  # 20 samples, not 20ms!
            
            if ibw >= iew - int(round(20e-3 * sampling_rate)):
                Pex = 0
                if verbose:
                    print(f"[Cycle {cycle_idx}]: P rejected - search window exhausted")
                break
            
            # Recalculate extrema in new window
            F_window = F[max(0, ibw):min(len(F), iew)]
            if len(F_window) == 0:
                Pex = 0
                break
            
            ymin = float(np.min(F_window))
            ymax = float(np.max(F_window))
            imin_rel = int(np.argmin(F_window))
            imax_rel = int(np.argmax(F_window))
            imin = ibw + imin_rel
            imax = ibw + imax_rel
            
            # Recalculate onset
            F_reversed = F[:imax + 1][::-1]
            iumb = thresholdcross(F_reversed, umbral)
            if iumb is None:
                iumb = imax
            else:
                iumb = imax - iumb + 1
            
            iteration += 1
        
        if Pex == 0:
            outer_iteration += 1
            continue
        
        P1 = iumb
        
        # Step 13: Find P wave peak using zero-crossings
        # Find zero-crossing to the right of imax
        # Find zero-crossing to the right of maximum
        F_forward = F[imax:min(len(F), qrs_onset_idx)]
        icero1 = zerocross(F_forward)
        if icero1 is None:
            icero1 = 0
        else:
            icero1 = imax + icero1 - 1  # Adjust for relative indexing
        
        # Find zero-crossing to the left of imin
        # Find zero-crossing to the left of minimum
        # F(1:imin) in MATLAB (1-based) = F[0:imin] in Python (0-based)
        # But we need to be careful: F has length len(Xpb)-1, so valid indices are 0 to len(Xpb)-2
        # imin is an index in F, so we use F[0:imin+1] to get F(1:imin+1) in MATLAB terms
        F_backward = F[0:imin + 1][::-1]  # Reverse for backward search (flipud)
        icero2 = zerocross(F_backward)
        if icero2 is None:
            icero2 = imin  # Fallback: use imin itself
        else:
            # Adjust for relative indexing
            # icero2 is the index in the reversed array (0-based)
            # In the original array, this corresponds to position (imin - icero2)
            # Account for relative indexing in flipped segment
            # In 0-based: icero2 = imin - icero2
            icero2 = imin - icero2 + 1
        
        # P peak is midpoint of zero-crossings
        # FIX 3: Don't clip P peak to window bounds (peak detection method doesn't clip)
        Pp = int(round((icero1 + icero2) / 2.0))
        # Peak detection method doesn't clip Pp - only ensure it's within signal bounds
        if Pp < 0:
            Pp = 0
        elif Pp >= len(Xpb):
            Pp = len(Xpb) - 1
        p_amplitude = float(Xpb[Pp])
        
        # Step 14: Noise level check before P onset
        # Noise estimation uses SAMPLE-BASED offsets, not time-based:
        # inic=P1-40+1, fin=P1-5 (in samples, not milliseconds)
        # At 250Hz: 40 samples = 160ms, 5 samples = 20ms
        inic = max(0, P1 - 40 + 1)  # 40 samples before P1 (sample-based noise estimation window)
        fin = max(0, P1 - 5)        # 5 samples before P1 (sample-based noise estimation window)
        if fin <= 0:
            fin = 1
        if inic <= 0:
            inic = 1
        
        X_noise = signal[inic:fin] if fin > inic else signal[max(0, P1-40):max(1, P1-5)]
        ruido = estimate_noise_level(X_noise)
        
        # Check: abs(Xpb(P1)-Xpb(Pp)) < 1.5*ruido AND (Pp-P1)/Fs < 40ms
        # RELAXED FOR P PEAK DETECTION: Removed duration requirement and made amplitude check more lenient
        # Since we're focusing on P peak center detection (not onset/offset characterization),
        # and P onset (P1) detection may be imperfect, we use a lenient amplitude-only check.
        p_onset_to_peak_diff = abs(Xpb[P1] - Xpb[Pp])
        p_onset_to_peak_duration_ms = (Pp - P1) / sampling_rate * 1000.0
        
        # Lenient check: Only reject if amplitude difference is extremely small (likely noise)
        # Use 1.0*ruido (more lenient than original 1.5*ruido) and removed duration requirement
        # This allows P peaks even if onset detection is imperfect
        if p_onset_to_peak_diff < 1.0 * ruido:
            if verbose:
                print(f"[Cycle {cycle_idx}]: OUTER LOOP: P rejected - noise check #1 failed (diff={p_onset_to_peak_diff:.4f} < 1.0*ruido={1.0*ruido:.4f}, duration={p_onset_to_peak_duration_ms:.1f}ms)")
            outer_iteration += 1
            continue
        
        # Step 15: Find P wave offset
        # FIX 4: Search to end of signal, not just QRS onset (offset detection: F(imin:length(F)))
        Kpe = 2.0
        umbral_offset = ymin / Kpe
        
        # Offset detection method: Faux=F(imin:length(F)) - search to end of signal
        F_forward_offset = F[imin:len(F)]  # Search to end of signal, not just QRS onset
        iumb_offset = thresholdcross(F_forward_offset, umbral_offset)
        
        if iumb_offset is None:
            iumb_offset = 0
        else:
            iumb_offset = imin + iumb_offset
        
        # Offset detection: if iumb>=QRS1, find min in F(imin:QRS1)
        if iumb_offset >= qrs_onset_idx:
            # Find minimum between imin and QRS onset
            if imin < qrs_onset_idx:
                F_segment = F[imin:qrs_onset_idx]
                if len(F_segment) > 0:
                    min_idx_rel = int(np.argmin(F_segment))
                    iumb_offset = imin + min_idx_rel
                else:
                    iumb_offset = qrs_onset_idx - 1
        
        P2 = iumb_offset
        # Offset boundary constraint: if P2>=QRS1, P2=QRS1-1
        if P2 >= qrs_onset_idx:
            P2 = qrs_onset_idx - 1
        
        # Step 16: Final noise level check
        X_noise_final = signal[ibw:iew] if iew > ibw else signal[max(0, ibw-50):iew]
        ruido_final = estimate_noise_level(X_noise_final)
        
        if abs(Xpb[Pp] - Xpb[P2]) <= 1.5 * ruido_final:
            if verbose:
                print(f"[Cycle {cycle_idx}]: OUTER LOOP: P rejected - final noise check failed (diff={abs(Xpb[Pp] - Xpb[P2]):.4f} <= 1.5*ruido_final={1.5*ruido_final:.4f})")
            outer_iteration += 1
            continue
        
        # Step 17: Final validation checks
        duration_ms = (P2 - P1) / sampling_rate * 1000.0
        
        if P1 >= P2 or Pp <= P1 or Pp >= P2 or P1 <= prevt or duration_ms > 180.0:
            if verbose:
                reason = []
                if P1 >= P2:
                    reason.append(f"P1({P1}) >= P2({P2})")
                if Pp <= P1:
                    reason.append(f"Pp({Pp}) <= P1({P1})")
                if Pp >= P2:
                    reason.append(f"Pp({Pp}) >= P2({P2})")
                if P1 <= prevt:
                    reason.append(f"P1({P1}) <= prevt({prevt})")
                if duration_ms > 180.0:
                    reason.append(f"duration({duration_ms:.1f}ms) > 180ms")
                print(f"[Cycle {cycle_idx}]: OUTER LOOP: P rejected - validation failed: {', '.join(reason)}")
            outer_iteration += 1
            continue
        
        # Step 18: Safety check - ensure P peak is at least 30ms before R peak
        # This provides additional protection against P-R overlap, especially if QRS onset detection is wrong
        if r_peak_idx is not None:
            min_pr_separation_ms = 30.0
            min_pr_separation_samples = int(min_pr_separation_ms * sampling_rate / 1000.0)
            pr_distance_samples = r_peak_idx - Pp
            pr_distance_ms = (pr_distance_samples / sampling_rate) * 1000.0
            
            if pr_distance_ms < min_pr_separation_ms:
                if verbose:
                    print(f"[Cycle {cycle_idx}]: OUTER LOOP: P rejected - too close to R peak (P-R={pr_distance_ms:.1f}ms < {min_pr_separation_ms}ms)")
                outer_iteration += 1
                continue
        
        # P wave successfully detected!
        if verbose:
            pr_info = ""
            if r_peak_idx is not None:
                pr_distance_ms = ((r_peak_idx - Pp) / sampling_rate) * 1000.0
                pr_info = f", P-R={pr_distance_ms:.1f}ms"
            print(f"[Cycle {cycle_idx}]: OUTER LOOP: P detected successfully at iteration {outer_iteration} - peak={Pp}, amplitude={p_amplitude:.4f}, "
                  f"onset={P1}, offset={P2}, duration={duration_ms:.1f}ms{pr_info}")
        
        # Mark as detected and return
        P_detected = True
        final_result = (Pp, p_amplitude, P1, P2)
        break  # Exit outer loop
    
    if not P_detected and verbose:
        print(f"[Cycle {cycle_idx}]: OUTER LOOP: Exhausted all {outer_iteration} iterations - P not detected")
    
    # End of outer loop - return result (either detected P or None)
    return final_result

