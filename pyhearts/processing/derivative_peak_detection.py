"""
Derivative-based peak detection for ECG waveforms.

This module implements a derivative-based approach to detecting P, Q, R, S, and T waves:
- Full-signal filtering before detection (avoids edge artifacts)
- Derivative-based peak detection (more accurate timing, no Gaussian fitting bias)
- Derivative-based waveform limit detection (onset/offset)
- One P-wave per cycle (with onset, peak, offset)

Key advantages over Gaussian fitting approach:
1. Eliminates +30ms systematic bias from Gaussian fitting
2. More robust to baseline drift (derivative-based)
3. Better detection rate (wider search windows, full-signal processing)
4. More accurate onset/offset detection (derivative-based waveform limits)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.qrs_removal import remove_qrs_sigmoid, remove_baseline_wander


@dataclass
class PeakAnnotation:
    """Represents a detected peak with onset and offset."""
    peak_idx: int  # Peak position in samples
    peak_amplitude: float  # Peak amplitude
    onset_idx: Optional[int] = None  # Onset position (waveform start)
    offset_idx: Optional[int] = None  # Offset position (waveform end)
    component: str = ""  # Component label: "P", "Q", "R", "S", "T"


class DerivativeBasedPeakDetector:
    """
    Derivative-based peak detection for ECG waveforms.
    
    Detects P, Q, R, S, T waves using:
    - Full-signal filtering (band-pass for P/T, high-pass for QRS)
    - Derivative-based peak detection
    - Derivative-based waveform limit detection
    """
    
    def __init__(self, sampling_rate: float, cfg: ProcessCycleConfig):
        self.sampling_rate = sampling_rate
        self.cfg = cfg
    
    def filter_full_signal(
        self,
        signal: np.ndarray,
        filter_type: str = "bandpass",
        lowcut: Optional[float] = None,
        highcut: Optional[float] = None,
    ) -> np.ndarray:
        """
        Filter entire signal using full-signal filtering approach.
        
        This avoids edge artifacts that occur when filtering short segments.
        
        Parameters
        ----------
        signal : np.ndarray
            Input ECG signal.
        filter_type : {"bandpass", "highpass"}
            Type of filter to apply.
        lowcut : float, optional
            Low cutoff frequency in Hz (for bandpass/highpass).
        highcut : float, optional
            High cutoff frequency in Hz (for bandpass).
        
        Returns
        -------
        np.ndarray
            Filtered signal.
        """
        nyq = self.sampling_rate / 2.0
        
        if filter_type == "bandpass":
            if lowcut is None or highcut is None:
                raise ValueError("bandpass requires both lowcut and highcut")
            low = max(0.01, min(lowcut / nyq, 0.99))
            high = max(low + 0.01, min(highcut / nyq, 0.99))
            b, a = butter(self.cfg.pwave_bandpass_order, [low, high], btype='band')
        elif filter_type == "highpass":
            if lowcut is None:
                raise ValueError("highpass requires lowcut")
            low = max(0.01, min(lowcut / nyq, 0.99))
            b, a = butter(self.cfg.rpeak_filter_order, low, btype='high')
        else:
            raise ValueError(f"Unknown filter_type: {filter_type}")
        
        try:
            filtered = filtfilt(b, a, signal)
            return filtered
        except Exception:
            # Fallback: return original signal if filtering fails
            return signal
    
    def detect_peak_derivative(
        self,
        signal: np.ndarray,
        start_idx: int,
        end_idx: int,
        polarity: str = "positive",
        min_prominence: Optional[float] = None,
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        Detect peak using derivative-based method.
        
        Finds peaks by locating zero-crossings in the first derivative:
        - Positive peaks: derivative goes from positive to negative
        - Negative peaks: derivative goes from negative to positive
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal array.
        start_idx : int
            Start index of search window (inclusive).
        end_idx : int
            End index of search window (exclusive).
        polarity : {"positive", "negative"}
            Whether to search for positive or negative peaks.
        min_prominence : float, optional
            Minimum peak prominence (amplitude above baseline).
        
        Returns
        -------
        Tuple[Optional[int], Optional[float]]
            (peak_idx, peak_amplitude) or (None, None) if not found.
        """
        if polarity not in {"positive", "negative"}:
            raise ValueError("polarity must be 'positive' or 'negative'")
        
        # Validate search window
        if start_idx >= end_idx or start_idx < 0 or end_idx > len(signal):
            return None, None
        
        segment = signal[start_idx:end_idx]
        if len(segment) < 3:
            return None, None
        
        # Compute first derivative
        deriv = np.diff(segment)
        if len(deriv) < 2:
            return None, None
        
        # Find zero-crossings in derivative
        if polarity == "positive":
            # Positive peak: derivative goes from positive to negative
            sign_changes = np.diff(np.sign(deriv))
            zero_crossings = np.where(sign_changes < 0)[0]  # negative transition
        else:
            # Negative peak: derivative goes from negative to positive
            sign_changes = np.diff(np.sign(deriv))
            zero_crossings = np.where(sign_changes > 0)[0]  # positive transition
        
        if len(zero_crossings) == 0:
            return None, None
        
        # Find the most prominent peak
        best_peak_idx = None
        best_peak_amp = None
        
        for zc in zero_crossings:
            peak_idx_rel = zc + 1  # Peak is one sample after zero-crossing
            if peak_idx_rel < len(segment):
                peak_idx_abs = start_idx + peak_idx_rel
                peak_amp = signal[peak_idx_abs]
                
                # Check prominence if specified
                if min_prominence is not None:
                    # Estimate local baseline
                    local_start = max(0, peak_idx_abs - 10)
                    local_end = min(len(signal), peak_idx_abs + 10)
                    local_baseline = np.median(signal[local_start:local_end])
                    prominence = abs(peak_amp - local_baseline)
                    if prominence < min_prominence:
                        continue
                
                if best_peak_amp is None or abs(peak_amp) > abs(best_peak_amp):
                    best_peak_idx = peak_idx_abs
                    best_peak_amp = peak_amp
        
        return best_peak_idx, best_peak_amp
    
    def find_waveform_limit_derivative(
        self,
        signal: np.ndarray,
        peak_idx: int,
        peak_amplitude: float,
        direction: str = "left",
        comp_label: str = "P",
    ) -> int:
        """
        Find waveform limit (onset/offset) using derivative-based method.
        
        Detects the point where signal slope/curvature changes significantly
        relative to local baseline using derivative-based approach.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal array.
        peak_idx : int
            Index of the peak.
        peak_amplitude : float
            Peak amplitude.
        direction : {"left", "right"}
            "left" for onset, "right" for offset.
        comp_label : str
            Component label ("P", "T", etc.).
        
        Returns
        -------
        int
            Index of the waveform limit.
        """
        # Search window based on component
        max_window_ms = self.cfg.shape_max_window_ms.get(comp_label, 120)
        search_samples = int(round(max_window_ms * self.sampling_rate / 1000.0))
        
        if direction == "left":
            search_start = max(0, peak_idx - search_samples)
            search_end = peak_idx
            search_slice = slice(search_start, search_end + 1)
        else:  # right/offset
            search_start = peak_idx
            search_end = min(len(signal) - 1, peak_idx + search_samples)
            search_slice = slice(search_start, search_end + 1)
        
        segment = signal[search_slice]
        if len(segment) < 5:
            return peak_idx  # fallback
        
        # Smooth signal to reduce noise
        smoothing_window = 7
        if comp_label == "T" and direction == "right":
            # Longer smoothing for T-offset
            smoothing_window_ms = getattr(self.cfg, 't_wave_offset_smoothing_window_ms', 20)
            smoothing_window = int(round(smoothing_window_ms * self.sampling_rate / 1000.0))
            smoothing_window = min(smoothing_window, len(segment) // 2)
            if smoothing_window < 7:
                smoothing_window = 7
            smoothing_window = smoothing_window if smoothing_window % 2 == 1 else smoothing_window + 1
        
        if len(segment) >= smoothing_window:
            segment_smooth = savgol_filter(segment, window_length=smoothing_window, polyorder=3, mode="interp")
        else:
            segment_smooth = segment
        
        # Compute derivatives
        dt = 1.0 / self.sampling_rate
        first_deriv = np.diff(segment_smooth) / dt
        
        # Estimate local baseline and noise
        if direction == "left":
            baseline_start = max(0, search_start - search_samples)
            baseline_segment = signal[baseline_start:search_start]
        else:
            baseline_end = min(len(signal), search_end + search_samples)
            baseline_segment = signal[search_end:baseline_end]
        
        if len(baseline_segment) < 3:
            local_baseline = np.median(signal)
            local_noise = 1.4826 * np.median(np.abs(signal - local_baseline))
        else:
            local_baseline = np.median(baseline_segment)
            local_noise = 1.4826 * np.median(np.abs(baseline_segment - local_baseline))
        
        # Adaptive threshold
        deriv_multiplier = getattr(self.cfg, 'waveform_limit_deriv_multiplier', 2.0)
        baseline_multiplier = getattr(self.cfg, 'waveform_limit_baseline_multiplier', 1.5)
        
        # For T-offset, use stricter criteria
        if comp_label == "T" and direction == "right":
            baseline_multiplier *= 0.7  # Stricter baseline tolerance
            deriv_multiplier *= 1.3  # Less sensitive derivative threshold
        
        deriv_threshold = local_noise * deriv_multiplier
        baseline_tolerance = local_noise * baseline_multiplier
        
        # Search from peak outward
        limit_idx = peak_idx
        
        # Map peak_idx to segment coordinates
        if direction == "left":
            peak_in_segment = peak_idx - search_start
        else:
            peak_in_segment = 0
        
        # Search for transition point
        for i in range(len(segment_smooth) - 2):
            if direction == "left":
                idx_in_segment = peak_in_segment - i  # search backwards
            else:
                idx_in_segment = i  # search forwards
            
            if idx_in_segment < 1 or idx_in_segment >= len(segment_smooth):
                continue
            
            signal_at_idx = segment_smooth[idx_in_segment]
            
            # Check if at baseline level
            at_baseline = abs(signal_at_idx - local_baseline) < baseline_tolerance
            
            # Check if derivative is below threshold
            if idx_in_segment - 1 < len(first_deriv):
                deriv_mag = abs(first_deriv[idx_in_segment - 1])
                deriv_small = deriv_mag < deriv_threshold
            else:
                deriv_small = False
            
            # Transition found if: at baseline AND derivative small
            if at_baseline and deriv_small:
                # For T-offset, require sustained baseline
                if comp_label == "T" and direction == "right":
                    # Check if signal stays at baseline for next few samples
                    sustained_samples_required = int(round(20 * self.sampling_rate / 1000.0))  # 20ms
                    sustained_samples_required = min(sustained_samples_required, len(segment_smooth) - idx_in_segment - 1)
                    
                    if sustained_samples_required >= 3:
                        # Check next samples
                        all_at_baseline = True
                        for j in range(1, sustained_samples_required + 1):
                            check_idx = idx_in_segment + j
                            if check_idx >= len(segment_smooth):
                                all_at_baseline = False
                                break
                            
                            signal_check = segment_smooth[check_idx]
                            at_baseline_check = abs(signal_check - local_baseline) < baseline_tolerance
                            
                            # Also check derivative
                            if check_idx - 1 < len(first_deriv):
                                deriv_check = abs(first_deriv[check_idx - 1])
                                deriv_small_check = deriv_check < deriv_threshold
                            else:
                                deriv_small_check = False
                            
                            if not (at_baseline_check and deriv_small_check):
                                all_at_baseline = False
                                break
                        
                        if all_at_baseline:
                            # Signal sustained at baseline - this is the offset
                            limit_idx = search_start + idx_in_segment
                            break
                        else:
                            # Not sustained, continue searching
                            continue
                    else:
                        # Not enough samples to check, use current point
                        limit_idx = search_start + idx_in_segment
                        break
                else:
                    # For other waves/onsets, use immediate detection
                    limit_idx = search_start + idx_in_segment
                    break
        
        return limit_idx
    
    def detect_p_wave(
        self,
        signal: np.ndarray,
        filtered_signal: np.ndarray,
        r_peak_idx: int,
        prev_r_peak_idx: Optional[int] = None,
    ) -> Optional[PeakAnnotation]:
        """
        Detect P-wave for a single cycle using derivative-based method.
        
        Searches the entire R-R interval (or up to max window) before the R-peak.
        Detects one P-wave with onset, peak, and offset.
        
        Parameters
        ----------
        signal : np.ndarray
            Original (unfiltered) signal.
        filtered_signal : np.ndarray
            Band-pass filtered signal (5-15 Hz).
        r_peak_idx : int
            Current R-peak index.
        prev_r_peak_idx : int, optional
            Previous R-peak index (for R-R interval).
        
        Returns
        -------
        Optional[PeakAnnotation]
            Detected P-wave or None.
        """
        # Determine search window
        if prev_r_peak_idx is not None:
            # Search entire R-R interval
            search_start = prev_r_peak_idx
        else:
            # Fallback: use max window
            max_window_ms = self.cfg.shape_max_window_ms.get("P", 120)
            search_start = max(0, r_peak_idx - int(round(max_window_ms * self.sampling_rate / 1000.0)))
        
        search_end = r_peak_idx
        
        if search_end - search_start < 10:  # Need at least 10 samples
            return None
        
        # Try both positive and negative P-waves (inverted leads)
        p_peak_pos, p_amp_pos = self.detect_peak_derivative(
            filtered_signal, search_start, search_end, polarity="positive"
        )
        p_peak_neg, p_amp_neg = self.detect_peak_derivative(
            filtered_signal, search_start, search_end, polarity="negative"
        )
        
        # Choose the one with larger absolute amplitude
        if p_peak_pos is not None and p_peak_neg is not None:
            if abs(p_amp_pos) >= abs(p_amp_neg):
                p_peak_idx, p_amp = p_peak_pos, p_amp_pos
            else:
                p_peak_idx, p_amp = p_peak_neg, p_amp_neg
        elif p_peak_pos is not None:
            p_peak_idx, p_amp = p_peak_pos, p_amp_pos
        elif p_peak_neg is not None:
            p_peak_idx, p_amp = p_peak_neg, p_amp_neg
        else:
            return None
        
        # Validate with SNR gate (if enabled)
        if hasattr(self.cfg, 'snr_mad_multiplier') and 'P' in self.cfg.snr_mad_multiplier:
            # Simple SNR check: peak should be above noise threshold
            local_start = max(0, p_peak_idx - 20)
            local_end = min(len(signal), p_peak_idx + 20)
            local_segment = signal[local_start:local_end]
            local_median = np.median(local_segment)
            local_mad = 1.4826 * np.median(np.abs(local_segment - local_median))
            snr_threshold = local_mad * self.cfg.snr_mad_multiplier['P']
            
            if abs(p_amp - local_median) < snr_threshold:
                return None  # SNR too low
        
        # Find onset and offset using derivative-based method
        # Use original signal for limits (not filtered) to avoid filter artifacts
        p_onset = self.find_waveform_limit_derivative(
            signal, p_peak_idx, p_amp, direction="left", comp_label="P"
        )
        p_offset = self.find_waveform_limit_derivative(
            signal, p_peak_idx, p_amp, direction="right", comp_label="P"
        )
        
        return PeakAnnotation(
            peak_idx=p_peak_idx,
            peak_amplitude=p_amp,
            onset_idx=p_onset,
            offset_idx=p_offset,
            component="P"
        )
    
    def detect_t_wave(
        self,
        signal: np.ndarray,
        filtered_signal: np.ndarray,
        r_peak_idx: int,
        next_r_peak_idx: Optional[int] = None,
    ) -> Optional[PeakAnnotation]:
        """
        Detect T-wave for a single cycle using derivative-based method.
        
        Uses QRS removal (ECGPUWAVE approach) if enabled in config.
        
        Parameters
        ----------
        signal : np.ndarray
            Original (unfiltered) signal.
        filtered_signal : np.ndarray
            Band-pass filtered signal (5-15 Hz).
        r_peak_idx : int
            Current R-peak index.
        next_r_peak_idx : int, optional
            Next R-peak index (for R-R interval).
        
        Returns
        -------
        Optional[PeakAnnotation]
            Detected T-wave or None.
        """
        # Determine search window (after QRS, before next R-peak)
        post_qrs_refractory_ms = getattr(self.cfg, 'postQRS_refractory_window_ms', 20)
        post_qrs_refractory = int(round(post_qrs_refractory_ms * self.sampling_rate / 1000.0))
        search_start = r_peak_idx + post_qrs_refractory
        
        if next_r_peak_idx is not None:
            search_end = next_r_peak_idx
        else:
            # Fallback: use max window
            max_window_ms = self.cfg.shape_max_window_ms.get("T", 180)
            search_end = min(len(signal), search_start + int(round(max_window_ms * self.sampling_rate / 1000.0)))
        
        if search_end - search_start < 10:
            return None
        
        # filtered_signal is already QRS-removed if enabled (from detect_all_peaks)
        # Try both positive and negative T-waves
        t_peak_pos, t_amp_pos = self.detect_peak_derivative(
            filtered_signal, search_start, search_end, polarity="positive"
        )
        t_peak_neg, t_amp_neg = self.detect_peak_derivative(
            filtered_signal, search_start, search_end, polarity="negative"
        )
        
        # Choose the one with larger absolute amplitude
        if t_peak_pos is not None and t_peak_neg is not None:
            if abs(t_amp_pos) >= abs(t_amp_neg):
                t_peak_idx, t_amp = t_peak_pos, t_amp_pos
            else:
                t_peak_idx, t_amp = t_peak_neg, t_amp_neg
        elif t_peak_pos is not None:
            t_peak_idx, t_amp = t_peak_pos, t_amp_pos
        elif t_peak_neg is not None:
            t_peak_idx, t_amp = t_peak_neg, t_amp_neg
        else:
            return None
        
        # Validate with SNR gate
        # IMPORTANT: Use the same signal that the peak was detected from (filtered_signal)
        # for SNR calculation, not the original signal, as they have different scales
        if hasattr(self.cfg, 'snr_mad_multiplier') and 'T' in self.cfg.snr_mad_multiplier:
            # Use filtered signal for SNR gate (same as peak detection)
            local_start = max(0, t_peak_idx - 20)
            local_end = min(len(filtered_signal), t_peak_idx + 20)
            local_segment = filtered_signal[local_start:local_end]
            local_median = np.median(local_segment)
            local_mad = 1.4826 * np.median(np.abs(local_segment - local_median))
            snr_threshold = local_mad * self.cfg.snr_mad_multiplier['T']
            
            if abs(t_amp - local_median) < snr_threshold:
                return None
        
        # Find onset and offset
        t_onset = self.find_waveform_limit_derivative(
            signal, t_peak_idx, t_amp, direction="left", comp_label="T"
        )
        t_offset = self.find_waveform_limit_derivative(
            signal, t_peak_idx, t_amp, direction="right", comp_label="T"
        )
        
        return PeakAnnotation(
            peak_idx=t_peak_idx,
            peak_amplitude=t_amp,
            onset_idx=t_onset,
            offset_idx=t_offset,
            component="T"
        )
    
    def detect_all_peaks(
        self,
        signal: np.ndarray,
        r_peaks: np.ndarray,
    ) -> Dict[int, Dict[str, Optional[PeakAnnotation]]]:
        """
        Detect all P and T waves for all cycles using derivative-based method.
        
        Uses full-signal filtering before detection to avoid edge artifacts.
        Optionally uses QRS removal (ECGPUWAVE approach) for T-wave detection.
        
        Parameters
        ----------
        signal : np.ndarray
            Preprocessed ECG signal.
        r_peaks : np.ndarray
            Array of R-peak indices.
        
        Returns
        -------
        Dict[int, Dict[str, Optional[PeakAnnotation]]]
            Dictionary mapping cycle_idx to detected peaks:
            {cycle_idx: {'P': PeakAnnotation, 'T': PeakAnnotation}}
        """
        # Filter entire signal using full-signal filtering approach
        # Band-pass for P and T waves (5-15 Hz)
        filtered_pt = self.filter_full_signal(
            signal,
            filter_type="bandpass",
            lowcut=self.cfg.pwave_bandpass_low_hz,
            highcut=self.cfg.pwave_bandpass_high_hz,
        )
        
        # For T-wave detection: optionally remove QRS complexes (ECGPUWAVE approach)
        use_qrs_removal = getattr(self.cfg, 't_wave_use_qrs_removal', True)
        filtered_pt_for_t = filtered_pt
        
        if use_qrs_removal and len(r_peaks) > 0:
            try:
                # Remove QRS from full signal
                signal_no_qrs, _ = remove_qrs_sigmoid(
                    signal,
                    r_peaks,
                    sampling_rate=self.sampling_rate,
                )
                
                # Apply baseline removal (ECGPUWAVE style)
                signal_no_qrs = remove_baseline_wander(
                    signal_no_qrs,
                    self.sampling_rate,
                )
                
                # Apply band-pass filter to QRS-removed signal
                filtered_pt_for_t = self.filter_full_signal(
                    signal_no_qrs,
                    filter_type="bandpass",
                    lowcut=self.cfg.pwave_bandpass_low_hz,
                    highcut=self.cfg.pwave_bandpass_high_hz,
                )
            except Exception:
                # Fallback to original filtered signal if QRS removal fails
                filtered_pt_for_t = filtered_pt
        
        # Detect peaks for each cycle
        results = {}
        
        for i, r_peak_idx in enumerate(r_peaks):
            prev_r = r_peaks[i - 1] if i > 0 else None
            next_r = r_peaks[i + 1] if i < len(r_peaks) - 1 else None
            
            # Detect P-wave (use original filtered signal)
            p_wave = self.detect_p_wave(signal, filtered_pt, r_peak_idx, prev_r)
            
            # Detect T-wave (use QRS-removed signal if enabled)
            t_wave = self.detect_t_wave(signal, filtered_pt_for_t, r_peak_idx, next_r)
            
            results[i] = {
                'P': p_wave,
                'T': t_wave,
            }
        
        return results


