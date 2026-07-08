"""
Sprint 4: QRS-energy-guided coarse P/T delay priors per beat.

Produces per-cycle expected R→P / R→T sample delays (template-offset units, same
convention as ``MedianBeatTemplate.p_offset_samples`` / ``t_offset_samples``).
Priors feed STPQ/template + outlier fences — not a replacement delineation stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.record_delineation import MedianBeatTemplate, _local_rr_samples
from pyhearts.processing.waveletoffset import calc_wavelet_dynamic_offset


def _ms_to_samples(ms: float, fs: float) -> int:
    return int(round(ms * fs / 1000.0))


def _finite(val) -> bool:
    return val is not None and not (isinstance(val, float) and np.isnan(val))


@dataclass
class RecordWaveletPriors:
    """Per-cycle expected P/T delays (template-offset units), aligned to ``cycle_labels``."""

    p_offset_samples: List[float]
    t_offset_samples: List[float]
    valid: bool = False
    n_beats: int = 0
    stats: Dict[str, int] = field(default_factory=dict)

    def expected_p_offset(self, cycle_idx: int) -> Optional[float]:
        if cycle_idx >= len(self.p_offset_samples):
            return None
        v = self.p_offset_samples[cycle_idx]
        return float(v) if _finite(v) else None

    def expected_t_offset(self, cycle_idx: int) -> Optional[float]:
        if cycle_idx >= len(self.t_offset_samples):
            return None
        v = self.t_offset_samples[cycle_idx]
        return float(v) if _finite(v) else None


def _beat_analysis_window(
    r_det: int,
    local_rr: float,
    n_samples: int,
) -> tuple[int, int]:
    half = max(int(round(local_rr * 0.55)), 8)
    lo = max(0, int(r_det) - half)
    hi = min(n_samples, int(r_det) + half)
    if hi - lo < 8:
        lo = max(0, int(r_det) - 4)
        hi = min(n_samples, int(r_det) + 5)
    return lo, hi


def _prior_for_beat(
    ecg: np.ndarray,
    r_det: int,
    local_rr: float,
    tmpl: MedianBeatTemplate,
    scale: float,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    expected_max_energy: float,
) -> tuple[Optional[float], Optional[float], bool]:
    """
    Coarse P/T template offsets blending median template with wavelet Q/S bounds.
    """
    fs = float(sampling_rate)
    scale = max(float(scale), 1e-6)
    lo, hi = _beat_analysis_window(r_det, local_rr, ecg.size)
    seg = ecg[lo:hi]
    r_rel = int(r_det - lo)
    r_std = max(
        float(_ms_to_samples(cfg.record_wavelet_r_std_ms, fs)),
        0.06 * float(local_rr),
    )

    _, _, _, q_min, s_max = calc_wavelet_dynamic_offset(
        seg,
        fs,
        expected_max_energy,
        r_center_idx=r_rel,
        r_std=r_std,
        cfg=cfg,
    )

    t_unscaled = tmpl.t_offset_samples
    p_unscaled = tmpl.p_offset_samples

    post_s = _ms_to_samples(cfg.record_wavelet_t_after_s_ms, fs)
    pre_q = _ms_to_samples(cfg.record_wavelet_p_before_q_ms, fs)

    if s_max is not None:
        delay_t = float(lo + int(s_max) - r_det) + post_s
        wavelet_t = delay_t / scale
        if t_unscaled is None:
            t_unscaled = wavelet_t
        else:
            t_unscaled = max(float(t_unscaled), wavelet_t)

    if q_min is not None:
        delay_p = float(lo + int(q_min) - r_det) - pre_q
        wavelet_p = delay_p / scale
        if p_unscaled is None:
            p_unscaled = wavelet_p
        else:
            p_unscaled = min(float(p_unscaled), wavelet_p)

    used_wavelet = (s_max is not None) or (q_min is not None)
    return p_unscaled, t_unscaled, used_wavelet


def compute_record_wavelet_pt_priors(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    cycle_labels: np.ndarray,
    tmpl: MedianBeatTemplate,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    expected_max_energy: float,
) -> RecordWaveletPriors:
    """
    Build per-cycle expected P/T template offsets for STPQ/template + outlier fences.

    Parameters
    ----------
    ecg
        Delineation trace (median baseline applied upstream in record delineation).
    expected_max_energy
        Reference QRS wavelet energy from ``epoch_ecg`` (95th percentile).
    """
    n_cycles = len(cycle_labels)
    p_offs: List[float] = [np.nan] * n_cycles
    t_offs: List[float] = [np.nan] * n_cycles
    stats: Dict[str, int] = {
        "n_cycles": n_cycles,
        "wavelet_used": 0,
        "template_only": 0,
    }

    if not cfg.record_wavelet_pt_prior or not tmpl.valid:
        return RecordWaveletPriors(p_offs, t_offs, valid=False, n_beats=0, stats=stats)

    ecg = np.asarray(ecg, dtype=float)
    ref_energy = float(expected_max_energy) if expected_max_energy > 0 else 1.0
    lo_rr_scale, hi_rr_scale = cfg.record_delineation_rr_scale_bounds

    for cycle_idx, cycle_label in enumerate(cycle_labels):
        epoch_i = int(cycle_label)
        if epoch_i < 0 or epoch_i >= len(r_peaks):
            continue
        r_det = int(r_peaks[epoch_i])
        local_rr = _local_rr_samples(
            cycle_idx, r_peaks, cycle_labels, tmpl.median_rr_samples
        )
        scale = 1.0
        if cfg.record_delineation_rr_scale_pt and tmpl.median_rr_samples > 0:
            scale = float(
                np.clip(
                    local_rr / tmpl.median_rr_samples,
                    lo_rr_scale,
                    hi_rr_scale,
                )
            )
        p_prior, t_prior, used = _prior_for_beat(
            ecg,
            r_det,
            local_rr,
            tmpl,
            scale,
            sampling_rate,
            cfg,
            ref_energy,
        )
        if p_prior is not None:
            p_offs[cycle_idx] = p_prior
        if t_prior is not None:
            t_offs[cycle_idx] = t_prior
        if used:
            stats["wavelet_used"] += 1
        else:
            stats["template_only"] += 1

    stats["valid_beats"] = sum(
        1 for i in range(n_cycles) if _finite(p_offs[i]) or _finite(t_offs[i])
    )
    return RecordWaveletPriors(
        p_offs,
        t_offs,
        valid=stats["valid_beats"] > 0,
        n_beats=stats["valid_beats"],
        stats=stats,
    )
