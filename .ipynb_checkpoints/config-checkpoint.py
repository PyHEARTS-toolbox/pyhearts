# pyhearts/config.py
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple, Dict, List


@dataclass(frozen=True)
class ProcessCycleConfig:
    """
    Central, typed configuration for process_cycle and helpers.
    Keep algorithmic behavior here; do not scatter magic numbers in code.
    Defaults are species-agnostic; use `for_mouse()` / `for_human()` for presets.
    """
   #  ----  R-peak detection  ---- 
    rpeak_prominence_multiplier: float = 3.0          # σ multiplier
    rpeak_min_refrac_ms: float = 100.0                # first-pass refractory
    rpeak_rr_frac_second_pass: float = 0.55           # second-pass refractory = k * median RR
    rpeak_bpm_bounds: Tuple[float, float] = (40.0, 900.0)  # clamp for RR estimate

    # ----- Epoching thresholds (used in epoch_ecg) -----
    epoch_corr_thresh: float = 0.80      # min correlation to keep an epoch (0–1)
    epoch_var_thresh: float  = 5.0       # max multiple of global variance allowed

    # # ----- Epoch window policy  -----
    pre_r_window: Optional[int] = None # Optional, default if rr interval / 2

    # ----- Wavelet-based dynamic offsets / R context -----
    wavelet_base_offset_ms: int = 1      # min dynamic offset (ms) around QRS
    wavelet_max_offset_ms: int = 60      # max dynamic offset (ms) around QRS
    wavelet_name: str = "db6"            # wavelet used for QRS-energy guided offsets
    wavelet_k_multiplier: float = 1.75   # k·σ on R to define local R-bounds
    wavelet_detail_level: int = 3        # preferred wavelet detail level
    wavelet_peak_height_sigma: float = 1.2

    # ---- Amplitude ratios to avoid noise ---
    amp_min_ratio: Dict[str, float] = field(
        default_factory=lambda: {"P": 0.03, "T": 0.05, "Q": 0.02, "S": 0.02}  # conservative defaults
    )
    
    # ---- SNR gate (P/T only) ----
    snr_mad_multiplier: dict[str, float] = field(
        default_factory=lambda: {"P": 2.0, "T": 2.0}  # |peak| ≥ k × MAD
    )
    snr_exclusion_ms: dict[str, int] = field(
        default_factory=lambda: {"P": 0, "T": 20}     # 0 ⇒ use half-FWHM policy; else ms
    )
    snr_apply_savgol: dict[str, bool] = field(
        default_factory=lambda: {"P": False, "T": True}
    )
    savgol_window_pts: int = 7
    savgol_polyorder: int = 3
    wavelet_guard_cap_ms: int = 120  # cap for post-QRS wavelet guard used in T search
    
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
        default_factory=lambda: {"P": 120, "Q": 40, "R": 60, "S": 40, "T": 180}
    )

    # ----- Shape feature thresholds -----
    threshold_fraction: float = 0.30     # fraction of (peak-to-baseline) for width crossings
    duration_min_ms: int = 20             # minimum valid duration for humans

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
            version="v1-mouse",
        )

    @classmethod
    def for_human(cls) -> "ProcessCycleConfig":
        """Preset tuned for adult human physiology."""
        return replace(
            cls(),
            detrend_window_ms=200,
            postQRS_refractory_window_ms = 20,    # small fixed refractory after QRS to avoid S tail
            amp_min_ratio={"P": 0.03, "T": 0.05, "Q": 0.02, "S": 0.02},  # lead II, capture non-ideal
            snr_mad_multiplier={"P": 1.75, "T": 1.75},   # slightly looser; P/T smaller in mice
            snr_exclusion_ms={"P": 0, "T": 10},          # shorter morphology → narrower exclusion
            snr_apply_savgol={"P": False, "T": True},
            rr_bounds_ms=(300, 1800),              # ~200–33 bpm
            shape_max_window_ms={"P": 160, "Q": 60, "R": 80, "S": 60, "T": 200},
            duration_min_ms=20,
            # Example tighter human bounds if desired:
            rpeak_bpm_bounds=(30.0, 240.0), rpeak_min_refrac_ms=120.0, # 500 bpm theoretical ceiling
            version="v1-human",
        )

#