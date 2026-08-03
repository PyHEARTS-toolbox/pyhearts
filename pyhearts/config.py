# pyhearts/config.py
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple, Dict, List


@dataclass(frozen=True)
class ProcessCycleConfig:
    """
    Central, typed configuration for process_cycle and helpers.
    Keep algorithmic behavior here; do not scatter magic numbers in code.
    Defaults are species-agnostic. Production human: :meth:`for_human_unified`.
    Legacy human baseline: :meth:`for_human`. Mouse: :meth:`for_mouse`.
    
    Key tuning notes (based on QTDB benchmark Dec 2024):
    - rpeak_prominence_multiplier: 2.5σ balances precision/recall better than 3.0σ
    - threshold_fraction: 0.15 captures more of P/T wave boundaries vs 0.30
    - epoch_corr_thresh: 0.70 retains more beats with morphological variation
    """
    #  ----  R-peak detection  ----
    #
    # BREAKING CHANGE NOTE:
    # Historically PyHEARTS exposed multiple R-peak detection algorithms via
    # `rpeak_method` (e.g., pan_tompkins, bandpass_energy). Those alternate paths
    # have been removed to reduce maintenance surface area. PyHEARTS now uses
    # the unified `pyhearts.processing.r_peak_detection`.
    rpeak_prominence_multiplier: float = 2.5          # σ multiplier (lowered from 3.0 for better recall)
    rpeak_min_refrac_ms: float = 100.0                # first-pass refractory
    rpeak_rr_frac_second_pass: float = 0.50           # second-pass refractory = k * median RR (lowered for sensitivity)
    rpeak_bpm_bounds: Tuple[float, float] = (40.0, 900.0)  # clamp for RR estimate
    
    # ---- R-peak detection algorithm ----
    # Production uses derivative + Phase A only.
    r_detection_method: str = "derivative"

    # ---- R-peak preprocessing (bandpass filter for noise robustness) ----
    rpeak_preprocess: bool = True                     # enable filtering before peak detection
    rpeak_highpass_hz: float = 0.5                    # highpass cutoff (removes baseline wander)
    rpeak_lowpass_hz: float = 40.0                    # lowpass cutoff (removes HF noise/EMG)
    rpeak_filter_order: int = 2                       # Butterworth filter order
    rpeak_notch_hz: Optional[float] = None            # power line notch (50/60 Hz), None=disabled

    # ----- R-peak Phase A (post-detection polish) -----
    r_post_detection_refrac_enabled: bool = False
    r_post_detection_refrac_ms: float = 300.0
    # PRP spacing: ~one candidate per beat before kurtosis/TRP
    r_prp_spacing_enabled: bool = True
    r_prp_min_interval_ms: float = 400.0
    r_prp_spacing_in_detection: bool = True
    r_kurtosis_rejection: bool = True
    r_kurtosis_window_ms: float = 80.0
    r_kurtosis_fraction_of_median: float = 0.75
    r_kurtosis_reference_mode: str = "sharpest_peak"
    r_kurtosis_fraction_of_sharpest: float = 0.50
    r_kurtosis_apply_only_if_oversegmented: bool = True
    r_kurtosis_oversegmented_peaks_per_min: float = 85.0
    r_miss_beat_rr_factor: float = 1.5
    r_trp_on_input_signal: bool = True

    # ----- Epoching thresholds (used in epoch_ecg) -----
    # Lowered from 0.80→0.70 to retain more beats with morphological variation
    epoch_corr_thresh: float = 0.70      # min correlation to keep an epoch (0–1)
    epoch_var_thresh: float  = 6.0       # max multiple of global variance allowed (raised from 5.0)

    # # ----- Epoch window policy  -----
    pre_r_window: Optional[int] = None # Optional, default if rr interval / 2

    # ----- Per-cycle processing (process_cycle) -----
    # epoch_ecg already applies a linear detrend per beat; a second detrend here shifts fiducials.
    apply_cycle_detrend: bool = False
    # Use global r_peak_detection indices as the R fiducial (refined locally), not argmax(|signal|).
    anchor_r_to_global_detection: bool = True
    r_anchor_refine_half_window_ms: float = 30.0
    r_anchor_refine_mode: str = "extremum"
    # Parabolic interpolation for R/P/T peaks; global_center_idx stored as float sample index.
    use_subsample_peak_refinement: bool = True

    # ----- QRS onset (P search boundary) -----
    qrs_onset_search_window_ms: float = 120.0
    qrs_onset_fallback_offset_ms: float = 80.0
    qrs_onset_max_before_r_ms: float = 200.0
    qrs_onset_min_before_r_ms: float = 30.0
    p_use_unified_qrs_onset: bool = True

    # ----- T-wave apex localization (derivative detector) -----
    t_wave_region_expansion_ms: float = 80.0
    t_wave_region_min_fraction: float = 0.0
    t_wave_refine_apex_on_original_ms: float = 40.0

    # ----- T RT plausibility gate (post per-cycle detection) -----
    t_rt_plausibility_gate: bool = True
    t_rt_bounds_ms: Tuple[float, float] = (120.0, 500.0)
    t_rt_plausibility_mad_multiplier: float = 3.0
    t_rt_plausibility_min_fence_ms: float = 40.0
    t_morphology_dominance_check: bool = True
    t_dominance_min_fraction: float = 0.30
    t_dominance_max_offset_ms: float = 25.0

    # ----- P–R interval validation (per-cycle) -----
    p_pr_interval_validation: bool = True
    p_pr_interval_bounds_ms: Tuple[float, float] = (80.0, 220.0)

    # ----- Record-level fiducial smoothing (step 5 / Tier A polish) -----
    record_fiducial_smoothing: bool = True
    record_smooth_min_beats: int = 5
    record_smooth_max_deviation_mad: float = 2.5
    record_smooth_strength: float = 0.85
    record_smooth_p: bool = True
    record_smooth_t: bool = True
    record_smooth_default_rp_ms: float = 120.0
    record_smooth_default_rt_ms: float = 280.0
    record_smooth_refine_on_epoch_ms: float = 0.0
    lock_r_global_to_detection: bool = False

    # ----- Record-level delineation (Tier B1: median-beat template) -----
    record_delineation: bool = False
    record_delineation_min_beats: int = 8
    record_delineation_refine_ms: float = 15.0
    record_delineation_rr_scale_pt: bool = True
    record_delineation_rr_scale_bounds: Tuple[float, float] = (0.85, 1.25)
    record_delineation_replace_p: bool = True
    record_delineation_overwrite_existing_p: bool = False
    record_delineation_replace_t: bool = True
    record_delineation_overwrite_existing_t: bool = False
    record_delineation_replace_t_if_outlier: bool = True
    record_delineation_t_outlier_mad: float = 2.5
    record_delineation_replace_r: bool = False
    record_delineation_p_search_before_r_ms: float = 200.0
    record_delineation_p_search_end_before_r_ms: float = 40.0
    record_delineation_t_search_after_r_ms: float = 100.0
    record_delineation_t_search_end_ms: float = 450.0
    record_delineation_refresh_features: bool = True
    record_delineation_refresh_shape: bool = True
    # Sprint 1: map P/T on every R (record-T + template fallback when search misses)
    record_delineation_map_all_beats: bool = False
    record_delineation_map_p_even_if_finite: bool = False
    record_delineation_map_t_even_if_finite: bool = False
    record_delineation_template_fallback: bool = True
    # Rahul-first: record-T seed before process_cycle (skip post-pass overwrite).
    record_delineation_before_cycles: bool = False
    # Phase 1: project S→Q template windows; per-cycle P/T search inside bounds only.
    record_template_prior_windows: bool = False
    record_template_prior_apply_p: bool = False
    # Phase 2A: rescue per-cycle T near morphology-aware template landmark.
    record_template_prior_rescue: bool = False
    record_template_prior_rescue_dispute_ms: float = 60.0
    record_template_prior_rescue_radius_ms: float = 80.0
    record_template_prior_rescue_min_prominence_frac: float = 0.06
    # Phase 2C: morphology-specific candidate scoring / rescue routing.
    record_template_prior_morphology_routing: bool = True
    record_template_prior_biphasic_rescue_lobe: str = "negative"
    # Phase 2D: cluster S→Q segments (0 or 1 = record template only).
    record_template_prior_cluster_k: int = 0
    # Uncertainty windows: landmark ± sigma_mult × per-record T timing σ (ms).
    record_template_prior_uncertainty_windows: bool = False
    record_template_prior_sigma_mult: float = 2.0
    record_template_prior_min_half_width_ms: float = 40.0
    record_template_prior_default_sigma_ms: float = 28.0
    # Landmark ensemble: score diverse hypotheses; template is soft prior only.
    record_template_prior_landmark_ensemble: bool = False
    record_template_prior_ensemble_timing_sigma_ms: float = 45.0
    record_template_prior_ensemble_template_sigma_ms: float = 55.0
    # Independent evidence features (ablation on/off; weights learned via LOO later).
    record_template_prior_evidence_local_shape: bool = False
    record_template_prior_evidence_neighborhood: bool = False
    record_template_prior_evidence_stability: bool = False
    record_template_prior_evidence_neighborhood_tolerance_ms: float = 50.0
    record_template_prior_evidence_neighborhood_sigma_ms: float = 45.0
    record_template_prior_evidence_neighborhood_max_penalty: float = 0.15
    # Learned candidate ranker (L2 logistic on interpretable evidence; ranks, not detects).
    record_template_prior_learned_ranker: bool = False
    record_template_prior_ranker_model_path: Optional[str] = None
    record_template_prior_ranker_commit_threshold: float = 0.5
    record_template_prior_ranker_correct_tol_ms: float = 20.0
    record_template_prior_ranker_C: float = 1.0
    record_template_prior_ranker_withhold_low_confidence: bool = True
    # Optional inverted-T post-selection rescue: negative_peak → DZC when
    # winner voltage percentile is extreme and DZC has better neighborhood consistency.
    record_inverted_dzc_rescue: bool = False
    record_inverted_dzc_rescue_volt_percentile_min: float = 95.0
    record_inverted_dzc_rescue_match_ms: float = 12.0
    # Never uses per-cycle derivative P/T fallback.
    record_fiducial_miss_policy: str = "template_fallback_then_nan"
    # Sprint 2: adaptive local refine after record-T/template guess
    record_delineation_refine_always: bool = False
    record_delineation_refine_adaptive: bool = False
    record_delineation_refine_rt_frac: float = 0.05
    record_delineation_refine_ms_max: float = 50.0
    record_refine_p_operator: str = "derivative_apex"
    record_refine_t_operator: str = "derivative_apex"
    record_refine_t_lowpass_hz: float = 40.0
    # Record local refine signal: record-T guess unchanged; only apex search substrate.
    # "delineation" = median-baseline delineation trace (SG optional via p_t_search_savgol).
    # "epoch" = per-cycle ``signal_y`` from epoching (analyzed input segment).
    # "clinical" = ``clinical_ecg`` slice when provided (export: raw WFDB crop).
    record_delineation_refine_signal: str = "delineation"
    # Step A diagnostic: stash record-T guess / refine / S–Q anchors per cycle (export only).
    record_delineation_t_timing_audit: bool = False
    # Sprint 3: clinical second-pass on raw/epoch trace
    record_clinical_verify: bool = False
    clinical_verify_signal: str = "clinical"
    clinical_verify_p_window_ms: float = 40.0
    clinical_verify_t_window_ms: float = 0.0
    clinical_verify_t_adaptive: bool = True
    clinical_verify_p_operator: str = "derivative_apex"
    clinical_verify_t_operator: str = "derivative_zc"
    clinical_verify_refresh_features: bool = True
    shape_at_timing_index_signal: str = "delineation"

    # ----- Phase B: P/T delineation signal & record-T template -----
    delineation_baseline_method: str = "linear_epoch"
    delineation_median_baseline_windows_s: Tuple[float, float] = (1.0, 2.0)
    # Optional light bandpass on record delineation signal (optional; off for the record-level T path).
    delineation_bandpass: bool = False
    delineation_bandpass_low_hz: float = 5.0
    delineation_bandpass_high_hz: float = 20.0
    delineation_bandpass_order: int = 4
    p_t_search_savgol: bool = True
    p_t_savgol_window_ms: float = 25.0
    p_t_savgol_polyorder: int = 3
    record_template_anchor: str = "r_centered"
    record_template_max_duration_s: float = 60.0
    record_template_aggregate: str = "median"  # "median" | "mean" (Rahul uses mean)
    record_delineation_t_search: bool = True  # S→Q anchored w1/w2 + template thresholds
    # "derivative": per-cycle P/T in process_cycle; "record_only": record record-T pass only
    p_t_detection_method: str = "derivative"
    p_t_threshold_mode: str = "fixed"
    record_template_t_r_amplitude_ratio: float = 0.5
    record_template_fixed_p_mv: float = 0.1
    # record-T template Tⱼ: dominant early positive peak vs isoelectric fallback
    record_template_t_landmark_sq_frac: Tuple[float, float] = (0.15, 0.40)
    # Rising-edge fallback only: scan onset from this S→Q fraction (shared peak gate stays at 15%).
    record_template_t_rising_edge_lo_frac: float = 0.08
    record_template_t_landmark_min_peak_frac: float = 0.20
    # Min prominence (fraction of T-region ref) for early positive Tⱼ; flat ST humps fail → isoelectric.
    record_template_t_landmark_min_prominence_frac: float = 0.10
    # Allow inverted (negative) T apex in the landmark search window when it dominates positive ST.
    record_template_t_landmark_inverted_peak: bool = True
    # T-threshold normalization: max positive excursion in this S→Q fraction (not global |deflection|).
    record_template_t_amplitude_norm_sq_frac: Tuple[float, float] = (0.15, 0.80)
    # Optional fixed S→Q window for inverted_t vs normal morphology (None = t_j→mid(T,P)).
    record_template_t_morphology_sq_frac: Optional[Tuple[float, float]] = None
    # Biphasic +−: template classify (morphology tag + pos/neg landmarks on median beat)
    record_template_detect_biphasic_positive_negative: bool = False
    # Ablation only: override Tⱼ / record-T search / projection (off in production guardrail path)
    record_biphasic_pm_lobe_search: bool = False
    record_t_biphasic_pm_early_margin_ms: float = 35.0
    record_t_biphasic_pm_sep_before_neg_ms: float = 50.0
    record_t_biphasic_pm_late_cap_ms: float = 120.0
    # Guardrail only: if T is >margin_ms before template positive apex, clamp to apex (no late shift)
    record_biphasic_pm_early_t_guardrail: bool = False
    record_biphasic_pm_early_guardrail_margin_ms: float = 10.0
    record_biphasic_pm_early_guardrail_max_uplift_ms: float = 80.0
    record_biphasic_pm_early_guardrail_beat_order_check: bool = False
    record_biphasic_pm_early_guardrail_min_prominence_frac: float = 0.0
    # Ablation: post-positive-peak downslope dz vs template-guided apex (off by default)
    record_t_post_apex_dz_preference: bool = False
    record_t_post_apex_dz_min_late_ms: float = 10.0
    record_t_post_apex_dz_max_late_ms: float = 80.0
    record_t_post_apex_dz_rt_tolerance_ms: float = 45.0
    record_t_post_apex_dz_pos_early_ms: float = 8.0
    record_t_post_apex_dz_min_beat_frac: float = 0.20
    record_t_post_apex_dz_max_beat_frac: Optional[float] = 0.38
    # Soft toggles (experimental; off by default — keep only if QTDB P/T improves)
    # Full-window record-T multi-candidate with RR-scaled RT prior (not landmark-local).
    soft_t_rt_multicand: bool = False
    soft_t_rt_prior_frac_rr: float = 0.32
    soft_t_rt_prior_sigma_ms: float = 80.0
    soft_t_rt_timing_weight: float = 1.0
    soft_t_rt_late_preference: float = 0.25
    soft_t_rt_use_mid_tp_window: bool = True  # open w1 to mid(T,P) so late apex is reachable
    # After record-T merge: restore Gaussian T on miss; if both finite and Gaussian is
    # later by ≥ this many ms, prefer Gaussian (LUDB near-misses are systematically early).
    record_t_fallback_gaussian_on_miss: bool = False
    record_t_prefer_later_gaussian_ms: float = 0.0  # 0 = disabled
    record_qs_search_window_ms: float = 150.0
    record_s_search_after_r_ms: float = 200.0  # S trough can lag R in wide QRS
    record_q_search_before_r_ms: float = 150.0
    record_t_max_rt_ms: float = 550.0  # physiological cap on T search / placement from R
    # record-T P search: R-anchored PR window (more robust than S→Q fraction w2)
    record_t_p_r_anchor: bool = True
    record_t_p_r_anchor_mode: str = "current_r"  # current_r (q1c) | next_r (Rahul S→Q)
    record_t_p_pr_max_ms: float = 250.0  # earliest P: R_anchor − this (max PR interval)
    record_t_p_pr_min_ms: float = 80.0  # latest P: R_anchor − this (min PR interval)
    record_t_p_template_guided: bool = True  # fallback / reconcile via template PR center
    record_t_p_template_guided_half_window_ms: float = 80.0  # ± search around projected P
    record_t_p_template_guided_reconcile_ms: float = 50.0  # re-pick if threshold apex farther than this
    record_t_p_template_guided_distance_penalty: float = 0.002  # per-sample score penalty vs projected P
    record_t_p_pr_center_ms: float = 180.0  # projected P before R (current_r); q1c typical PR
    # Mode 1 per-beat T: cap w1 at template Tⱼ (+ margin) and template-guided apex in w1
    record_t_w1_end_mode: str = "template_tj_margin"  # mid_tp | template_tj_margin
    record_t_w1_post_tj_frac: float = 0.08  # w1 hi = t_j + frac*(mid(T,P)-t_j)
    # When formula w1_hi (S→Q fraction) is below min, extend to min capped at p_j − margin (0 = off)
    record_t_w1_hi_min_sq_frac: float = 0.0
    record_t_w1_hi_pj_margin_sq_frac: float = 0.15
    record_t_apex_mode: str = "mode1"  # mode1 = template-guided primary; threshold = amplitude path
    record_t_project_from: str = "landmark"  # landmark | delineated | blend
    record_t_landmark_blend_frac: float = 0.35  # blend: land + frac*(offset-land)
    record_t_mode1_max_dist_ms: float = 22.0  # mode1: ignore extrema farther than this
    record_t_template_guided: bool = True
    record_t_template_guided_half_window_ms: float = 35.0
    record_t_template_guided_reconcile_ms: float = 40.0
    record_t_template_guided_distance_penalty: float = 0.006
    # plateau_apex + inverted_t: forward argmin trough search from projected t_j (ms)
    record_t_plateau_apex_forward_ms: float = 40.0
    record_t_mode1_min_amp_frac: float = 0.20  # nearest-apex pool: |amp| >= frac * window max
    # early_peak landmark: narrow w1 / delineation so late-lobe threshold search cannot dominate
    record_t_early_peak_w1_post_tj_frac: float = 0.03
    record_t_early_peak_delineation_post_tj_frac: float = 0.04
    record_t_early_peak_rising_edge_frac: float = 0.35  # onset threshold vs window peak
    record_t_early_peak_max_late_from_center_ms: float = 15.0  # cap record-T pick vs projected Tⱼ
    # When record-T guess exceeds this distance from per-cycle T, keep per-cycle (requires t_wave_use_record_prior)
    record_t_per_cycle_guardrail_ms: float = 20.0
    # Sprint 4: QRS-energy wavelet coarse P/T priors (record-T/template expected offsets)
    record_wavelet_pt_prior: bool = False
    record_wavelet_r_std_ms: float = 20.0
    record_wavelet_t_after_s_ms: float = 20.0
    record_wavelet_p_before_q_ms: float = 20.0
    record_t_use_savgol: bool = True  # False = record-T apex search on raw delineation segment
    record_delineation_fill_missing_t: bool = False  # post-pass record-T/template for NaN T
    clinical_verify_t_conditional: bool = False  # T clinical only if RT disputed vs record prior
    clinical_verify_t_dispute_mad_mult: float = 2.5
    clinical_verify_t_dispute_min_ms: float = 50.0

    # ----- Per-record routing (record delineation vs per-cycle T) -----
    # Optional path to a JSON routing table:
    # { "records": [ { "record": "sel104", "route": "per_cycle_t" }, ... ] }
    record_t_routing_table: Optional[str] = None

    # ----- Wavelet-based dynamic offsets / R context -----
    wavelet_base_offset_ms: int = 1      # min dynamic offset (ms) around QRS
    wavelet_max_offset_ms: int = 60      # max dynamic offset (ms) around QRS
    wavelet_name: str = "db6"            # wavelet used for QRS-energy guided offsets
    wavelet_k_multiplier: float = 1.75   # k·σ on R to define local R-bounds
    wavelet_detail_level: int = 3        # preferred wavelet detail level
    wavelet_peak_height_sigma: float = 1.2
    
    # ----- P-wave specific band-pass filtering -----
    pwave_use_bandpass: bool = True      # enable band-pass filter for P-wave detection
    pwave_bandpass_low_hz: float = 5.0   # low cutoff frequency (Hz) for P-wave enhancement
    pwave_bandpass_high_hz: float = 15.0 # high cutoff frequency (Hz) for P-wave enhancement
    pwave_bandpass_order: int = 4        # filter order for P-wave band-pass
    
    # ----- Experimental P-peak detection improvements (for testing) -----
    p_use_training_phase: bool = False   # Enable training-phase adaptive thresholds
    p_use_training_as_primary: bool = False  # Use training thresholds as PRIMARY validation (vs secondary check)
    p_safety_margin_ms: float = 60.0     # Safety margin before Q/R peak (adjustable, default 60ms)
    p_use_derivative_validated_method: bool = False  # Use derivative-validated P wave detection (derivative-based with comprehensive validation)
    p_enable_distance_validation: bool = False  # Enable distance-based validation (P-R, P-Q distances). Disabled by default to support abnormal cycles
    p_enable_morphology_validation: bool = False  # Enable morphology-based validation (duration, sharpness). Disabled by default to support abnormal cycles
    
    # ---- Amplitude ratios to avoid noise ---
    # Increased P wave minimum ratio from 0.02 to 0.03 to reduce false positives (low precision issue)
    # Typical P waves are 5-15% of R peak amplitude, so 3% minimum is still lenient but filters very small deflections
    amp_min_ratio: Dict[str, float] = field(
        default_factory=lambda: {"P": 0.03, "T": 0.02, "Q": 0.02, "S": 0.02}  # T: 2% of R for improved recall
    )
    
    # ---- SNR gate (P/T only) ----
    # Increased P wave SNR threshold from 1.0 to 1.5 to reduce false positives (low precision issue)
    # Many false positives had very low prominence (0.0025-0.0475 mV), indicating baseline noise
    # Higher threshold (1.5× MAD) will filter out these low-prominence noise artifacts
    snr_mad_multiplier: dict[str, float] = field(
        default_factory=lambda: {"P": 1.5, "T": 1.5}  # |peak| ≥ k × MAD (increased P from 1.0 to 1.5 for precision)
    )
    snr_exclusion_ms: dict[str, int] = field(
        default_factory=lambda: {"P": 0, "T": 15}     # 0 ⇒ use half-FWHM policy; else ms
    )
    snr_apply_savgol: dict[str, bool] = field(
        default_factory=lambda: {"P": False, "T": True}
    )
    savgol_window_pts: int = 7
    savgol_polyorder: int = 3
    wavelet_guard_cap_ms: int = 100  # cap for post-QRS wavelet guard used in T search (reduced from 120)
    
     # ----- Curve-fit -----
    bound_factor: float = 0.20           # bounds scale around (center, height, std)
    maxfev: int = 2500                   # scipy curve_fit eval cap
    detrend_window_ms: int = 150         # baseline detrend window (ms)
    postQRS_refractory_window_ms: int = 20    # small fixed refractory after QRS to avoid S tail (~20 ms in humans)
    
    # ----- Physiological interval limits (RR/PP) -----
    # Defaults span human brady to mouse tachy; presets will narrow these.
    rr_bounds_ms: Tuple[int, int] = (60, 1800)          # 1000–33 bpm
    pp_bounds_ms: Optional[Tuple[int, int]] = None      # None → reuse rr_bounds_ms

    #  ---- Search window policy for bounds (physiologic caps by wave) ---- 
    shape_search_scale: float = 2.0
    shape_max_window_ms: Dict[str, int] = field(
        default_factory=lambda: {"P": 1200, "Q": 40, "R": 60, "S": 40, "T": 180}  # P: 1200ms for full R-R interval search (was 250ms)
    )

    # ----- Shape feature thresholds -----
    # threshold_fraction: Lowered from 0.30→0.20 based on QTDB benchmark
    # PR interval was -74ms biased (P onset detected late)
    # QT interval was -47ms biased (T offset detected early)
    # Using 20% of peak height captures more wave morphology while staying robust
    # Note: 0.15 was too aggressive and caused issues with validation
    threshold_fraction: float = 0.20     # fraction of (peak-to-baseline) for width crossings
    duration_min_ms: int = 20             # minimum valid duration for humans
    
    # ----- Waveform limit locator (derivative-based detection) -----
    use_derivative_based_limits: bool = True      # enable derivative-based waveform limit detection
    waveform_limit_deriv_multiplier: float = 1.5  # multiplier for derivative threshold (lower = more sensitive)
    waveform_limit_baseline_multiplier: float = 2.0  # multiplier for baseline proximity check
    local_baseline_window_fraction: float = 0.3  # fraction of search window for baseline estimation
    p_wave_deriv_sensitivity_multiplier: float = 0.7  # P-wave specific sensitivity boost (lower = more sensitive)
    t_wave_offset_smoothing_window_ms: int = 50  # longer smoothing for T-offset detection
    detect_u_wave: bool = True                    # detect U-waves to avoid including in T-offset

    # ----- T-wave detection (search window + QRS removal) -----
    t_wave_use_qrs_removal: bool = True
    t_wave_search_start_ms: float = 100.0       # minimum delay after R before T search
    t_wave_search_qrs_end_margin_ms: float = 20.0
    t_wave_search_rr_frac: float = 0.55         # max search extent as fraction of cycle RR
    t_wave_search_max_ms: float = 600.0       # absolute cap after R (long QT)
    t_wave_search_end_margin_ms: float = 40.0  # margin before cycle end
    t_wave_search_min_window_ms: float = 100.0
    t_wave_qrs_pre_rr_frac: float = 1.0 / 3.0   # QRS removal: start before R
    t_wave_qrs_post_ms: float = 80.0            # QRS removal: minimum extent after R

    # ----- T-wave phase 2: record prior + secondary detector / fusion -----
    t_wave_use_record_prior: bool = True
    t_wave_use_secondary_detector: bool = True
    t_wave_prior_min_beats: int = 5
    t_wave_prior_window_ms: float = 80.0
    t_wave_prior_min_window_ms: float = 40.0
    t_wave_prior_max_deviation_mad: float = 3.0
    t_wave_prior_default_rt_ms: float = 280.0
    t_wave_prior_amp_min_ratio: float = 0.01
    t_wave_fusion_prefer_prior_on_tie: bool = True

    # ----- Lite mode (fast iteration: peaks only, no Gaussian/shape/ST) -----
    lite_mode: bool = False

    # ----- Sharpness (derivative-based; minimal public knobs) -----
    sharp_stat: str = "p95"              # {"mean","median","p95"}
    sharp_amp_norm: str = "p2p"          # {"p2p","rms","mad"}

    # Pairwise differences
    shape_diff_mode: str = "signed"      # {"signed","absolute"}
    shape_interdeflection_pairs: List[Tuple[str, str]] = field(
        default_factory=lambda: [("R", "S"), ("R", "P"), ("T", "R")]
    )

    # ----- Repro tag -----
    version: str = field(default="v1", compare=False)

    # -------- Validation --------
    def __post_init__(self):
        # epoching thresholds
        if not (0.0 <= self.epoch_corr_thresh <= 1.0): raise ValueError("epoch_corr_thresh in [0,1]")
        if self.epoch_var_thresh <= 0: raise ValueError("epoch_var_thresh > 0")
        if self.r_anchor_refine_half_window_ms <= 0:
            raise ValueError("r_anchor_refine_half_window_ms > 0")
        if self.r_anchor_refine_mode not in ("derivative", "extremum"):
            raise ValueError("r_anchor_refine_mode must be 'derivative' or 'extremum'")
        if self.r_post_detection_refrac_ms <= 0:
            raise ValueError("r_post_detection_refrac_ms > 0")
        if self.r_prp_min_interval_ms <= 0:
            raise ValueError("r_prp_min_interval_ms > 0")
        if self.r_kurtosis_window_ms <= 0:
            raise ValueError("r_kurtosis_window_ms > 0")
        if not (0.0 < self.r_kurtosis_fraction_of_median <= 1.0):
            raise ValueError("r_kurtosis_fraction_of_median in (0, 1]")
        if self.r_kurtosis_reference_mode not in (
            "sharpest_peak",
            "local_rr_neighbor",
            "upper_median",
        ):
            raise ValueError(
                "r_kurtosis_reference_mode must be sharpest_peak, local_rr_neighbor, or upper_median"
            )
        if not (0.0 < self.r_kurtosis_fraction_of_sharpest <= 1.0):
            raise ValueError("r_kurtosis_fraction_of_sharpest in (0, 1]")
        if self.r_kurtosis_oversegmented_peaks_per_min <= 0:
            raise ValueError("r_kurtosis_oversegmented_peaks_per_min > 0")
        if self.r_miss_beat_rr_factor <= 1.0:
            raise ValueError("r_miss_beat_rr_factor > 1.0")
        if self.r_detection_method != "derivative":
            raise ValueError("r_detection_method must be 'derivative'")
        if self.qrs_onset_search_window_ms <= 0:
            raise ValueError("qrs_onset_search_window_ms > 0")
        if self.qrs_onset_fallback_offset_ms <= 0:
            raise ValueError("qrs_onset_fallback_offset_ms > 0")
        if self.qrs_onset_max_before_r_ms <= self.qrs_onset_min_before_r_ms:
            raise ValueError("qrs_onset_max_before_r_ms > qrs_onset_min_before_r_ms")
        if self.t_wave_region_expansion_ms < 0:
            raise ValueError("t_wave_region_expansion_ms >= 0")
        if not (0.0 <= self.t_wave_region_min_fraction <= 1.0):
            raise ValueError("t_wave_region_min_fraction in [0, 1]")
        if self.t_wave_refine_apex_on_original_ms <= 0:
            raise ValueError("t_wave_refine_apex_on_original_ms > 0")
        rt_lo, rt_hi = self.t_rt_bounds_ms
        if rt_lo <= 0 or rt_hi <= rt_lo:
            raise ValueError("t_rt_bounds_ms: 0 < lo < hi")
        if self.t_rt_plausibility_mad_multiplier <= 0:
            raise ValueError("t_rt_plausibility_mad_multiplier > 0")
        if self.t_rt_plausibility_min_fence_ms <= 0:
            raise ValueError("t_rt_plausibility_min_fence_ms > 0")
        if not (0.0 < self.t_dominance_min_fraction <= 1.0):
            raise ValueError("t_dominance_min_fraction in (0, 1]")
        if self.t_dominance_max_offset_ms <= 0:
            raise ValueError("t_dominance_max_offset_ms > 0")
        p_lo, p_hi = self.p_pr_interval_bounds_ms
        if p_lo <= 0 or p_hi <= p_lo:
            raise ValueError("p_pr_interval_bounds_ms: 0 < lo < hi")
        if self.record_smooth_min_beats < 2:
            raise ValueError("record_smooth_min_beats >= 2")
        if self.record_smooth_max_deviation_mad <= 0:
            raise ValueError("record_smooth_max_deviation_mad > 0")
        if not (0.0 <= self.record_smooth_strength <= 1.0):
            raise ValueError("record_smooth_strength in [0, 1]")
        if self.record_smooth_default_rp_ms <= 0:
            raise ValueError("record_smooth_default_rp_ms > 0")
        if self.record_smooth_default_rt_ms <= 0:
            raise ValueError("record_smooth_default_rt_ms > 0")
        if self.record_smooth_refine_on_epoch_ms < 0:
            raise ValueError("record_smooth_refine_on_epoch_ms >= 0")
        if self.record_delineation_min_beats < 3:
            raise ValueError("record_delineation_min_beats >= 3")
        if self.record_delineation_refine_ms <= 0:
            raise ValueError("record_delineation_refine_ms > 0")
        if self.record_delineation_refine_rt_frac < 0:
            raise ValueError("record_delineation_refine_rt_frac >= 0")
        if self.record_delineation_refine_ms_max < self.record_delineation_refine_ms:
            raise ValueError("record_delineation_refine_ms_max >= record_delineation_refine_ms")
        for op_name, op_val in (
            ("record_refine_p_operator", self.record_refine_p_operator),
            ("record_refine_t_operator", self.record_refine_t_operator),
        ):
            if op_val not in (
                "derivative_apex",
                "derivative_zc",
                "argmax",
                "argmin",
            ):
                raise ValueError(
                    f"{op_name} must be derivative_apex, derivative_zc, argmax, or argmin"
                )
        if self.record_refine_t_lowpass_hz <= 0:
            raise ValueError("record_refine_t_lowpass_hz > 0")
        if self.record_delineation_refine_signal not in (
            "delineation",
            "epoch",
            "clinical",
        ):
            raise ValueError(
                "record_delineation_refine_signal must be "
                "'delineation', 'epoch', or 'clinical'"
            )
        if self.clinical_verify_signal not in ("clinical", "epoch"):
            raise ValueError("clinical_verify_signal must be 'clinical' or 'epoch'")
        if self.clinical_verify_p_window_ms <= 0:
            raise ValueError("clinical_verify_p_window_ms > 0")
        if self.clinical_verify_t_window_ms < 0:
            raise ValueError("clinical_verify_t_window_ms >= 0")
        for op_name, op_val in (
            ("clinical_verify_p_operator", self.clinical_verify_p_operator),
            ("clinical_verify_t_operator", self.clinical_verify_t_operator),
        ):
            if op_val not in (
                "derivative_apex",
                "derivative_zc",
                "argmax",
                "argmin",
            ):
                raise ValueError(
                    f"{op_name} must be derivative_apex, derivative_zc, argmax, or argmin"
                )
        if self.shape_at_timing_index_signal not in ("delineation", "clinical"):
            raise ValueError("shape_at_timing_index_signal must be 'delineation' or 'clinical'")
        lo_rs, hi_rs = self.record_delineation_rr_scale_bounds
        if not (0.5 <= lo_rs <= hi_rs <= 2.0):
            raise ValueError("record_delineation_rr_scale_bounds in [0.5,2]")
        if self.record_delineation_t_outlier_mad <= 0:
            raise ValueError("record_delineation_t_outlier_mad > 0")
        if self.delineation_baseline_method not in ("linear_epoch", "median_record"):
            raise ValueError("delineation_baseline_method must be 'linear_epoch' or 'median_record'")
        w1, w2 = self.delineation_median_baseline_windows_s
        if w1 <= 0 or w2 <= 0:
            raise ValueError("delineation_median_baseline_windows_s entries > 0")
        if self.p_t_savgol_window_ms <= 0:
            raise ValueError("p_t_savgol_window_ms > 0")
        if self.p_t_savgol_polyorder < 1:
            raise ValueError("p_t_savgol_polyorder >= 1")
        if self.delineation_bandpass:
            lo, hi = self.delineation_bandpass_low_hz, self.delineation_bandpass_high_hz
            if lo <= 0 or hi <= 0 or lo >= hi:
                raise ValueError("delineation_bandpass requires 0 < low_hz < high_hz")
            if self.delineation_bandpass_order < 1:
                raise ValueError("delineation_bandpass_order >= 1")
        if self.record_template_anchor not in ("r_centered", "s_to_q"):
            raise ValueError("record_template_anchor must be 'r_centered' or 's_to_q'")
        if self.record_template_max_duration_s <= 0:
            raise ValueError("record_template_max_duration_s > 0")
        if self.record_template_aggregate not in ("median", "mean"):
            raise ValueError("record_template_aggregate must be 'median' or 'mean'")
        if self.p_t_detection_method not in ("derivative", "record_only"):
            raise ValueError("p_t_detection_method must be 'derivative' or 'record_only'")
        if self.p_t_threshold_mode not in ("fixed", "template"):
            raise ValueError("p_t_threshold_mode must be 'fixed' or 'template'")
        if self.p_t_detection_method == "record_only" and not self.record_delineation:
            raise ValueError("p_t_detection_method='record_only' requires record_delineation=True")
        if self.record_fiducial_miss_policy not in (
            "template_fallback_then_nan",
            "nan_only",
        ):
            raise ValueError(
                "record_fiducial_miss_policy must be 'template_fallback_then_nan' or 'nan_only'"
            )
        if self.record_delineation_before_cycles and not self.record_delineation:
            raise ValueError("record_delineation_before_cycles requires record_delineation=True")
        if self.record_delineation_before_cycles and self.p_t_detection_method != "record_only":
            raise ValueError(
                "record_delineation_before_cycles requires p_t_detection_method='record_only'"
            )
        if not (0.0 < self.record_template_t_r_amplitude_ratio <= 2.0):
            raise ValueError("record_template_t_r_amplitude_ratio in (0, 2]")
        if self.record_template_fixed_p_mv <= 0:
            raise ValueError("record_template_fixed_p_mv > 0")
        t_lo, t_hi = self.record_template_t_landmark_sq_frac
        if not (0.0 <= t_lo < t_hi <= 1.0):
            raise ValueError("record_template_t_landmark_sq_frac must satisfy 0 <= lo < hi <= 1")
        if not (0.0 <= self.record_template_t_rising_edge_lo_frac < t_lo):
            raise ValueError(
                "record_template_t_rising_edge_lo_frac must satisfy 0 <= lo < record_template_t_landmark_sq_frac[0]"
            )
        if not (0.0 < self.record_template_t_landmark_min_peak_frac <= 1.0):
            raise ValueError("record_template_t_landmark_min_peak_frac in (0, 1]")
        if not (0.0 <= self.record_template_t_landmark_min_prominence_frac <= 1.0):
            raise ValueError("record_template_t_landmark_min_prominence_frac in [0, 1]")
        if self.record_t_w1_end_mode not in ("mid_tp", "template_tj_margin"):
            raise ValueError("record_t_w1_end_mode must be mid_tp or template_tj_margin")
        if not (0.0 <= self.record_t_w1_post_tj_frac <= 1.0):
            raise ValueError("record_t_w1_post_tj_frac in [0, 1]")
        if not (0.0 <= self.record_t_w1_hi_min_sq_frac <= 1.0):
            raise ValueError("record_t_w1_hi_min_sq_frac in [0, 1]")
        if not (0.0 <= self.record_t_w1_hi_pj_margin_sq_frac <= 1.0):
            raise ValueError("record_t_w1_hi_pj_margin_sq_frac in [0, 1]")
        if self.record_t_apex_mode not in ("mode1", "threshold"):
            raise ValueError("record_t_apex_mode must be mode1 or threshold")
        if self.record_t_project_from not in ("landmark", "delineated", "blend"):
            raise ValueError("record_t_project_from must be landmark, delineated, or blend")
        if not (0.0 <= self.record_t_landmark_blend_frac <= 1.0):
            raise ValueError("record_t_landmark_blend_frac in [0, 1]")
        if self.record_t_mode1_max_dist_ms < 0:
            raise ValueError("record_t_mode1_max_dist_ms must be >= 0")
        if self.record_t_template_guided_half_window_ms <= 0:
            raise ValueError("record_t_template_guided_half_window_ms > 0")
        if self.record_t_template_guided_reconcile_ms <= 0:
            raise ValueError("record_t_template_guided_reconcile_ms > 0")
        if self.record_t_template_guided_distance_penalty < 0:
            raise ValueError("record_t_template_guided_distance_penalty >= 0")
        if self.record_t_plateau_apex_forward_ms <= 0:
            raise ValueError("record_t_plateau_apex_forward_ms > 0")
        an_lo, an_hi = self.record_template_t_amplitude_norm_sq_frac
        if not (0.0 <= an_lo < an_hi <= 1.0):
            raise ValueError("record_template_t_amplitude_norm_sq_frac must satisfy 0 <= lo < hi <= 1")
        if self.record_template_t_morphology_sq_frac is not None:
            m_lo, m_hi = self.record_template_t_morphology_sq_frac
            if not (0.0 <= m_lo < m_hi <= 1.0):
                raise ValueError(
                    "record_template_t_morphology_sq_frac must satisfy 0 <= lo < hi <= 1"
                )
        if self.record_t_max_rt_ms <= 0:
            raise ValueError("record_t_max_rt_ms > 0")
        if self.record_t_prefer_later_gaussian_ms < 0:
            raise ValueError("record_t_prefer_later_gaussian_ms >= 0")
        if self.record_t_p_pr_min_ms <= 0:
            raise ValueError("record_t_p_pr_min_ms > 0")
        if self.record_t_p_pr_max_ms <= self.record_t_p_pr_min_ms:
            raise ValueError("record_t_p_pr_max_ms > record_t_p_pr_min_ms")
        if self.record_t_p_r_anchor_mode not in ("current_r", "next_r"):
            raise ValueError("record_t_p_r_anchor_mode must be 'current_r' or 'next_r'")
        if self.record_t_p_template_guided_half_window_ms <= 0:
            raise ValueError("record_t_p_template_guided_half_window_ms > 0")
        if self.record_t_p_template_guided_reconcile_ms <= 0:
            raise ValueError("record_t_p_template_guided_reconcile_ms > 0")
        if self.record_t_p_template_guided_distance_penalty < 0:
            raise ValueError("record_t_p_template_guided_distance_penalty >= 0")
        if not (self.record_t_p_pr_min_ms <= self.record_t_p_pr_center_ms <= self.record_t_p_pr_max_ms):
            raise ValueError("record_t_p_pr_center_ms must lie within [pr_min, pr_max]")
        if self.record_qs_search_window_ms <= 0:
            raise ValueError("record_qs_search_window_ms > 0")
        if self.record_s_search_after_r_ms <= 0:
            raise ValueError("record_s_search_after_r_ms > 0")
        if self.record_q_search_before_r_ms <= 0:
            raise ValueError("record_q_search_before_r_ms > 0")
        for ms_name, ms_val in (
            ("record_delineation_p_search_before_r_ms", self.record_delineation_p_search_before_r_ms),
            ("record_delineation_p_search_end_before_r_ms", self.record_delineation_p_search_end_before_r_ms),
            ("record_delineation_t_search_after_r_ms", self.record_delineation_t_search_after_r_ms),
            ("record_delineation_t_search_end_ms", self.record_delineation_t_search_end_ms),
        ):
            if ms_val <= 0:
                raise ValueError(f"{ms_name} > 0")

        # amplitude ratios
        for k, v in self.amp_min_ratio.items():
            if k not in {"P", "Q", "R", "S", "T"} or not (0.0 <= v < 1.0):
                raise ValueError("amp_min_ratio keys ∈ {P,Q,R,S,T} and values in [0,1)")

        # SNR gate dicts
        for d, name in [(self.snr_mad_multiplier,"snr_mad_multiplier")]:
            for k, v in d.items():
                if k not in {"P","T"} or not (v > 0):
                    raise ValueError(f"{name} keys ∈ {{P,T}}, values > 0")
        for d, name in [(self.snr_exclusion_ms,"snr_exclusion_ms")]:
            for k, v in d.items():
                if k not in {"P","T"} or v < 0:
                    raise ValueError(f"{name} keys ∈ {{P,T}}, values ≥ 0")
        for d, name in [(self.snr_apply_savgol,"snr_apply_savgol")]:
            for k, v in d.items():
                if k not in {"P","T"} or not isinstance(v, bool):
                    raise ValueError(f"{name} keys ∈ {{P,T}}, bool values")
        if self.savgol_window_pts < 3 or self.savgol_window_pts % 2 == 0:
            raise ValueError("savgol_window_pts must be odd and ≥3")
        if not (1 <= self.savgol_polyorder < self.savgol_window_pts):
            raise ValueError("savgol_polyorder must be ≥1 and < savgol_window_pts")

        # curve-fit / preprocessing
        if not (0.0 < self.bound_factor < 1.0): raise ValueError("bound_factor in (0,1)")
        if self.maxfev <= 0: raise ValueError("maxfev > 0")
        if self.detrend_window_ms <= 0: raise ValueError("detrend_window_ms > 0")
        if self.postQRS_refractory_window_ms <= 0: raise ValueError ("postQRS_refractory_window_ms > 0")
        
        # wavelet offsets / R context
        if self.wavelet_base_offset_ms <= 0 or self.wavelet_max_offset_ms <= 0:
            raise ValueError("wavelet offsets > 0")
        if self.wavelet_base_offset_ms >= self.wavelet_max_offset_ms:
            raise ValueError("wavelet_base_offset_ms < wavelet_max_offset_ms")
        if self.wavelet_detail_level <= 0: raise ValueError("wavelet_detail_level >= 1")
        if self.wavelet_peak_height_sigma <= 0: raise ValueError("wavelet_peak_height_sigma > 0")
        if self.wavelet_k_multiplier <= 0: raise ValueError("wavelet_k_multiplier > 0")

        if self.wavelet_guard_cap_ms <= 0:
            raise ValueError("wavelet_guard_cap_ms must be > 0")

        # physiologic limits
        lo_rr, hi_rr = self.rr_bounds_ms
        if not (lo_rr > 0 and hi_rr > 0 and lo_rr < hi_rr): raise ValueError("rr_bounds_ms lo < hi, >0")
        if self.pp_bounds_ms is not None:
            lo_pp, hi_pp = self.pp_bounds_ms
            if not (lo_pp > 0 and hi_pp > 0 and lo_pp < hi_pp):
                raise ValueError("pp_bounds_ms lo < hi, >0")

        # shape feature thresholds
        if not (0.0 < self.threshold_fraction < 1.0): raise ValueError("threshold_fraction in (0,1)")
        if self.duration_min_ms <= 0: raise ValueError("duration_min_ms > 0")
        if self.shape_search_scale <= 0: raise ValueError("shape_search_scale > 0")
        if not isinstance(self.shape_max_window_ms, dict) or not all(
            isinstance(k, str) and isinstance(v, (int, float)) and v > 0
            for k, v in self.shape_max_window_ms.items()
        ):
            raise ValueError("shape_max_window_ms: dict[str] -> positive number")
        if self.shape_diff_mode not in {"signed", "absolute"}:
            raise ValueError("shape_diff_mode ∈ {'signed','absolute'}")

        # sharpness
        if self.sharp_stat not in {"mean", "median", "p95"}:
            raise ValueError("sharp_stat ∈ {'mean','median','p95'}")
        if self.sharp_amp_norm not in {"p2p", "rms", "mad"}:
            raise ValueError("sharp_amp_norm ∈ {'p2p','rms','mad'}")
        
        # r-peak
        lo_bpm, hi_bpm = self.rpeak_bpm_bounds
        if not (0 < lo_bpm < hi_bpm):
            raise ValueError("rpeak_bpm_bounds require 0 < low < high")
        if self.rpeak_prominence_multiplier <= 0:
            raise ValueError("rpeak_prominence_multiplier > 0")
        if self.rpeak_min_refrac_ms <= 0:
            raise ValueError("rpeak_min_refrac_ms > 0")
        if not (0.0 < self.rpeak_rr_frac_second_pass < 1.0):
            raise ValueError("rpeak_rr_frac_second_pass in (0,1)")
        
        # P-wave band-pass filter parameters
        if self.pwave_bandpass_low_hz <= 0 or self.pwave_bandpass_high_hz <= 0:
            raise ValueError("pwave_bandpass frequencies must be > 0")
        if self.pwave_bandpass_low_hz >= self.pwave_bandpass_high_hz:
            raise ValueError("pwave_bandpass_low_hz must be < pwave_bandpass_high_hz")
        if self.pwave_bandpass_order < 1:
            raise ValueError("pwave_bandpass_order must be >= 1")
        
        # waveform limit locator parameters
        if self.waveform_limit_deriv_multiplier <= 0:
            raise ValueError("waveform_limit_deriv_multiplier > 0")
        if self.waveform_limit_baseline_multiplier <= 0:
            raise ValueError("waveform_limit_baseline_multiplier > 0")
        if not (0.0 < self.local_baseline_window_fraction < 1.0):
            raise ValueError("local_baseline_window_fraction in (0,1)")
        if self.p_wave_deriv_sensitivity_multiplier <= 0:
            raise ValueError("p_wave_deriv_sensitivity_multiplier > 0")
        if self.t_wave_offset_smoothing_window_ms <= 0:
            raise ValueError("t_wave_offset_smoothing_window_ms > 0")
        if self.t_wave_search_start_ms < 0:
            raise ValueError("t_wave_search_start_ms >= 0")
        if self.t_wave_search_qrs_end_margin_ms < 0:
            raise ValueError("t_wave_search_qrs_end_margin_ms >= 0")
        if not (0.0 < self.t_wave_search_rr_frac <= 1.0):
            raise ValueError("t_wave_search_rr_frac in (0, 1]")
        if self.t_wave_search_max_ms <= 0:
            raise ValueError("t_wave_search_max_ms > 0")
        if self.t_wave_search_end_margin_ms < 0:
            raise ValueError("t_wave_search_end_margin_ms >= 0")
        if self.t_wave_search_min_window_ms <= 0:
            raise ValueError("t_wave_search_min_window_ms > 0")
        if not (0.0 < self.t_wave_qrs_pre_rr_frac < 1.0):
            raise ValueError("t_wave_qrs_pre_rr_frac in (0, 1)")
        if self.t_wave_qrs_post_ms <= 0:
            raise ValueError("t_wave_qrs_post_ms > 0")


    # -------- Presets --------
    @classmethod
    def for_mouse(cls) -> "ProcessCycleConfig":
        """Preset tuned for mouse physiology."""
        return replace(
            cls(),
            detrend_window_ms=100,
            postQRS_refractory_window_ms = 10,    # small fixed refractory after QRS to avoid S tail
            amp_min_ratio={"P": 0.03, "T": 0.04, "Q": 0.025, "S": 0.025},  # lead II, capture non-ideal
            snr_mad_multiplier={"P": 2.0, "T": 2.0},
            snr_exclusion_ms={"P": 0, "T": 20},
            snr_apply_savgol={"P": False, "T": True},
            rr_bounds_ms=(80, 250),                # ~750–240 bpm
            shape_max_window_ms={"P": 35, "Q": 12, "R": 18, "S": 12, "T": 60},
            duration_min_ms=2, 
            # R-peak knobs can stay at defaults unless you want to tighten:
            rpeak_bpm_bounds=(300.0, 1000.0), rpeak_min_refrac_ms=67.0, # 900 bpm theoretical ceiling
            # Mouse QRS complexes can contain higher-frequency content; a 40Hz low-pass
            # may overly smooth at high sampling rates (e.g., 10kHz) and reduce detectability.
            rpeak_lowpass_hz=150.0,
            # Typical acquisition has mains interference; enable notch by default.
            rpeak_notch_hz=50.0,
            version="v1-mouse",
        )

    @classmethod
    def for_human(cls) -> "ProcessCycleConfig":
        """
        Human morphology preset without record-level record-T (per-cycle P/T only).

        The public analyzer uses :meth:`for_human_unified` for its record-T config.
        This preset remains useful as a morphology-only baseline.

        R detection (Phase A):
        - R-peak bandpass 5–20 Hz
        - 400 ms post-detection refractory (PRP spacing)
        - Softer kurtosis rejection (0.55 × median sharp-peak kurtosis)
        - TRP ±30 ms on raw signal; gap-fill RR factor 1.5
        - Slightly looser epoch screening to retain marginal morphologies

        P/T (unchanged from v2.1): per-cycle detection, no record delineation,
        PR interval validation, T RT gate + morphology dominance.
        """
        return replace(
            cls(),
            detrend_window_ms=200,
            postQRS_refractory_window_ms=20,
            amp_min_ratio={"P": 0.020, "T": 0.020, "Q": 0.015, "S": 0.015},
            snr_mad_multiplier={"P": 0.5, "T": 0.8},
            t_wave_use_qrs_removal=True,
            snr_exclusion_ms={"P": 0, "T": 10},
            snr_apply_savgol={"P": False, "T": True},
            rr_bounds_ms=(300, 1800),
            shape_max_window_ms={"P": 200, "Q": 60, "R": 80, "S": 60, "T": 220},
            duration_min_ms=20,
            threshold_fraction=0.15,
            epoch_corr_thresh=0.65,
            epoch_var_thresh=7.0,
            rpeak_prominence_multiplier=2.5,
            rpeak_bpm_bounds=(30.0, 240.0),
            rpeak_min_refrac_ms=120.0,
            rpeak_highpass_hz=5.0,
            rpeak_lowpass_hz=20.0,
            use_derivative_based_limits=True,
            waveform_limit_deriv_multiplier=1.5,
            waveform_limit_baseline_multiplier=2.0,
            local_baseline_window_fraction=0.3,
            p_wave_deriv_sensitivity_multiplier=0.7,
            t_wave_offset_smoothing_window_ms=50,
            detect_u_wave=True,
            pwave_use_bandpass=True,
            pwave_bandpass_low_hz=1.0,
            pwave_bandpass_high_hz=60.0,
            pwave_bandpass_order=2,
            p_use_derivative_validated_method=True,
            p_enable_distance_validation=False,
            p_enable_morphology_validation=False,
            record_delineation=False,
            record_delineation_replace_p=False,
            record_delineation_replace_t=False,
            record_fiducial_smoothing=False,
            delineation_baseline_method="linear_epoch",
            p_t_search_savgol=True,
            record_template_anchor="r_centered",
            p_t_threshold_mode="fixed",
            r_post_detection_refrac_enabled=True,
            r_post_detection_refrac_ms=400.0,
            r_prp_spacing_enabled=True,
            r_prp_min_interval_ms=400.0,
            r_prp_spacing_in_detection=True,
            r_kurtosis_reference_mode="sharpest_peak",
            r_kurtosis_fraction_of_sharpest=0.50,
            r_kurtosis_apply_only_if_oversegmented=True,
            r_kurtosis_oversegmented_peaks_per_min=85.0,
            r_kurtosis_fraction_of_median=0.55,
            r_miss_beat_rr_factor=1.5,
            r_trp_on_input_signal=True,
            r_anchor_refine_half_window_ms=30.0,
            t_rt_plausibility_gate=True,
            t_morphology_dominance_check=True,
            p_pr_interval_validation=True,
            version="v2.3-human-r-prp-kurtosis",
        )

    @classmethod
    def _for_human_unified_base(cls) -> "ProcessCycleConfig":
        """
        Production human_unified baseline.

        Derivative R + per-cycle P + record record-level T with threshold apex on delineated
        trace, w1 floor (40% S→Q), fixed-window morphology classification,
        template-guided reconcile, and derivative_apex local refine.
        """
        return replace(
            cls.for_human(),
            delineation_baseline_method="median_record",
            delineation_median_baseline_windows_s=(1.0, 2.0),
            p_t_detection_method="derivative",
            p_use_derivative_validated_method=True,
            p_t_search_savgol=False,
            record_t_use_savgol=False,
            delineation_bandpass=False,
            record_delineation=True,
            record_template_anchor="s_to_q",
            record_template_aggregate="mean",
            record_template_max_duration_s=60.0,
            p_t_threshold_mode="template",
            record_delineation_t_search=True,
            record_delineation_replace_p=True,
            record_delineation_replace_t=True,
            record_delineation_overwrite_existing_p=False,
            record_delineation_overwrite_existing_t=True,
            record_template_t_amplitude_norm_sq_frac=(0.15, 0.80),
            record_template_t_morphology_sq_frac=(0.20, 0.60),
            record_t_max_rt_ms=550.0,
            record_t_w1_end_mode="template_tj_margin",
            record_t_w1_post_tj_frac=0.15,
            record_t_w1_hi_min_sq_frac=0.40,
            record_t_w1_hi_pj_margin_sq_frac=0.15,
            record_t_apex_mode="threshold",
            record_t_project_from="delineated",
            record_t_template_guided=True,
            record_t_template_guided_half_window_ms=60.0,
            record_t_template_guided_reconcile_ms=40.0,
            record_t_template_guided_distance_penalty=0.002,
            record_refine_t_operator="derivative_apex",
            record_template_t_landmark_inverted_peak=True,
            record_template_t_landmark_min_prominence_frac=0.10,
            t_wave_use_record_prior=False,
            t_wave_use_secondary_detector=False,
            record_delineation_refresh_features=True,
            record_delineation_refresh_shape=True,
            record_fiducial_smoothing=True,
            record_qs_search_window_ms=150.0,
            record_s_search_after_r_ms=200.0,
            record_q_search_before_r_ms=150.0,
            record_t_fallback_gaussian_on_miss=True,
            record_t_prefer_later_gaussian_ms=20.0,
            version="human-unified",
        )

    @classmethod
    def for_human_unified(cls) -> "ProcessCycleConfig":
        """
        Production human preset (default): signed-polarity record-T pipeline.

        Archived experimental fill-missing-T variant: :meth:`for_human_unified_v33a`.
        """
        return cls._for_human_unified_base()

    @classmethod
    def for_human_unified_v321(cls) -> "ProcessCycleConfig":
        """Deprecated alias for :meth:`for_human_unified`."""
        return cls._for_human_unified_base()

    @classmethod
    def for_human_unified_template_prior_phase1(cls) -> "ProcessCycleConfig":
        """
        Phase 1 experiment: S→Q template windows constrain per-cycle T search.

        Skips record record-T overwrite; per-cycle derivative T/P detection runs inside
        template-projected w1/w2 bounds (see ``template_prior_windows``).
        """
        return replace(
            cls.for_human_unified(),
            record_template_prior_windows=True,
            record_delineation_overwrite_existing_t=False,
            record_delineation_replace_t=False,
            record_delineation_replace_p=False,
            record_delineation_map_all_beats=False,
            record_fiducial_smoothing=False,
            version="human-unified-template-prior-phase1",
        )

    @classmethod
    def for_human_unified_template_prior_phase1_uncertainty(cls) -> "ProcessCycleConfig":
        """Phase 1 + landmark-centered uncertainty windows (prediction ± 2σ)."""
        return replace(
            cls.for_human_unified_template_prior_phase1(),
            record_template_prior_uncertainty_windows=True,
            version="human-unified-template-prior-phase1-uncertainty",
        )

    @classmethod
    def for_human_unified_template_prior_ensemble(cls) -> "ProcessCycleConfig":
        """
        Template-prior + landmark ensemble scoring + uncertainty windows.

        Template shifts probability via soft prior; per-beat ensemble picks landmark.
        """
        return replace(
            cls.for_human_unified_template_prior_phase2(),
            record_template_prior_landmark_ensemble=True,
            record_template_prior_uncertainty_windows=True,
            record_template_prior_cluster_k=0,
            # LOO ablation: neighborhood +1.6 pp holdout; shape/stability need learned weights.
            record_template_prior_evidence_neighborhood=True,
            version="human-unified-template-prior-ensemble",
        )

    @classmethod
    def for_human_unified_template_prior_ensemble_inverted_dzc(cls) -> "ProcessCycleConfig":
        """Ensemble + narrow inverted_t negative_peak→DZC post-selection rescue.

        Evidence features match the base ensemble preset; only the rescue flag differs.
        """
        return replace(
            cls.for_human_unified_template_prior_ensemble(),
            record_inverted_dzc_rescue=True,
            record_inverted_dzc_rescue_volt_percentile_min=95.0,
            version="human-unified-template-prior-ensemble-inv-dzc",
        )

    @classmethod
    def for_human_unified_template_prior_ranked(cls) -> "ProcessCycleConfig":
        """
        Ensemble evidence + L2 logistic candidate ranker (model path set at runtime).

        Candidate generation unchanged; hand-tuned ensemble argmax replaced by
        P(correct | evidence) with calibrated commit threshold.
        """
        return replace(
            cls.for_human_unified_template_prior_ensemble(),
            record_template_prior_learned_ranker=True,
            record_template_prior_rescue=False,
            record_template_prior_evidence_local_shape=True,
            record_template_prior_evidence_stability=True,
            version="human-unified-template-prior-ranked",
        )

    @classmethod
    def for_human_unified_template_prior_phase2(cls) -> "ProcessCycleConfig":
        """
        Phase 2: template-prior windows + morphology-routed rescue + optional clustering.

        Extends Phase 1 with candidate rescue near template landmarks and k=2 S→Q
        cluster templates for window projection (priors only, no record-T overwrite).
        """
        return replace(
            cls.for_human_unified_template_prior_phase1(),
            record_template_prior_rescue=True,
            record_template_prior_rescue_dispute_ms=60.0,
            record_template_prior_rescue_radius_ms=80.0,
            record_template_prior_morphology_routing=True,
            record_template_prior_cluster_k=2,
            record_template_detect_biphasic_positive_negative=True,
            record_template_t_morphology_sq_frac=(0.20, 0.60),
            version="human-unified-template-prior-phase2",
        )

    @classmethod
    def for_human_unified_biphasic_positive_negative_lobe_search(cls) -> "ProcessCycleConfig":
        """
        Production + biphasic +− template classify and positive-lobe-only record-level T search.

        Ablation preset: does not change routing, R-centered windows, or other morphologies.
        """
        return replace(
            cls._for_human_unified_base(),
            record_template_detect_biphasic_positive_negative=True,
            record_biphasic_pm_lobe_search=True,
            version="human-unified-biphasic-pm-lobe",
        )

    @classmethod
    def for_human_unified_biphasic_pm_early_guardrail(cls) -> "ProcessCycleConfig":
        """
        Production + biphasic +− classify-only early-T guardrail (no record-T override / lobe search).

        If ``T < first_positive_peak - margin``, clamp ``T`` to the template positive apex.
        Never shifts T later.
        """
        return replace(
            cls._for_human_unified_base(),
            record_template_detect_biphasic_positive_negative=True,
            record_biphasic_pm_lobe_search=False,
            record_biphasic_pm_early_t_guardrail=True,
            record_biphasic_pm_early_guardrail_margin_ms=10.0,
            record_biphasic_pm_early_guardrail_max_uplift_ms=80.0,
            record_biphasic_pm_early_guardrail_beat_order_check=False,
            record_biphasic_pm_early_guardrail_min_prominence_frac=0.0,
            version="human-unified-biphasic-pm-guardrail",
        )

    @classmethod
    def for_human_unified_post_apex_dz_preference(cls) -> "ProcessCycleConfig":
        """Ablation: post_apex_dz vs positive_peak record-T compare."""
        return replace(
            cls._for_human_unified_base(),
            record_t_post_apex_dz_preference=True,
            version="human-unified-post-apex-dz-preference",
        )

    @classmethod
    def for_human_unified_biphasic_pm_early_guardrail_uplift120(cls) -> "ProcessCycleConfig":
        """Guardrail ablation: 120 ms max uplift + beat-level pos-before-neg checks."""
        return replace(
            cls.for_human_unified_biphasic_pm_early_guardrail(),
            record_biphasic_pm_early_guardrail_max_uplift_ms=120.0,
            record_biphasic_pm_early_guardrail_beat_order_check=True,
            record_biphasic_pm_early_guardrail_min_prominence_frac=0.12,
            version="human-unified-biphasic-pm-guardrail-u120",
        )

    @classmethod
    def for_human_unified_v321_biphasic_positive_negative_lobe_search(cls) -> "ProcessCycleConfig":
        """Deprecated alias for :meth:`for_human_unified_biphasic_positive_negative_lobe_search`."""
        return cls.for_human_unified_biphasic_positive_negative_lobe_search()

    @classmethod
    def for_human_unified_v321_biphasic_pm_early_guardrail(cls) -> "ProcessCycleConfig":
        """Deprecated alias for :meth:`for_human_unified_biphasic_pm_early_guardrail`."""
        return cls.for_human_unified_biphasic_pm_early_guardrail()

    @classmethod
    def for_human_unified_v321_post_apex_dz_preference(cls) -> "ProcessCycleConfig":
        """Deprecated alias for :meth:`for_human_unified_post_apex_dz_preference`."""
        return cls.for_human_unified_post_apex_dz_preference()

    @classmethod
    def for_human_unified_v321_biphasic_pm_early_guardrail_uplift120(cls) -> "ProcessCycleConfig":
        """Deprecated alias for :meth:`for_human_unified_biphasic_pm_early_guardrail_uplift120`."""
        return cls.for_human_unified_biphasic_pm_early_guardrail_uplift120()

    @classmethod
    def for_human_unified_v33a(cls) -> "ProcessCycleConfig":
        """
        Archived experimental fill-missing-T variant (not the production default).
        """
        return replace(
            cls._for_human_unified_base(),
            record_delineation_fill_missing_t=True,
            record_delineation_template_fallback=True,
            record_clinical_verify=False,
            record_delineation_map_all_beats=False,
            t_wave_use_record_prior=True,
            record_t_per_cycle_guardrail_ms=20.0,
            version="human-unified-fill-only",
        )

    @classmethod
    def for_human_unified_v33a_routed(cls) -> "ProcessCycleConfig":
        """Archived v3.3a + per-record routing table (May 2026 sprint)."""
        cfg = cls.for_human_unified_v33a()
        return replace(
            cfg,
            record_t_routing_table=(
                "benchmark_results/archive/v33a_sprint_20260526/v33a_t_routing_table_20260525.json"
            ),
        )
