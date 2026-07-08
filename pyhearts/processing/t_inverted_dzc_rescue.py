"""
Optional post-selection rescue: inverted_t negative_peak → DZC.

Rule (shipped counterfactual that passed table gates):
  morphology == inverted_t
  winner_type == negative_peak   # landmark-ensemble argmax
  DZC candidate exists
  committed T voltage percentile > 95   # geometry of export/committed sample
  dzc_neighborhood_consistency > ensemble_pick neighborhood_consistency

Substitutes the highest-evidence DZC (neighborhood − 0.5·stability + ...) among
candidates that beat the ensemble pick on neighborhood_consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as dc_replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.delineation_signal import prepare_record_delineation_signal
from pyhearts.processing.t_candidate_scoring import generate_t_candidates
from pyhearts.processing.t_landmark_ensemble import (
    _neighbor_rts_ms,
    _primary_source,
    filter_ensemble_candidates,
    score_landmark_ensemble,
)
from pyhearts.processing.t_morphology_routing import (
    morphology_rescue_landmark_global,
    normalize_t_morphology_tag,
)
from pyhearts.processing.template_prior_window_diagnostics import _beat_template_correlation


def stt_voltage_percentile(
    ecg: np.ndarray,
    sample: float,
    s_i: int,
    q_next: int,
) -> float:
    """Percentile of landmark voltage within the S→Q ST–T segment (0–100)."""
    if not np.isfinite(sample) or len(ecg) < 3:
        return float("nan")
    i = int(round(float(sample)))
    if i < 0 or i >= len(ecg):
        return float("nan")
    lo = max(int(s_i), 0)
    hi = min(int(q_next), len(ecg))
    if hi <= lo + 2:
        return float("nan")
    stt = ecg[lo:hi].astype(float)
    return float(100.0 * np.mean(stt <= float(ecg[i])))


def _dzc_evidence_score(neighborhood: float, stability: float, timing: float, deriv: float) -> float:
    return (
        float(neighborhood)
        + 0.3 * float(timing)
        + 0.2 * float(deriv)
        - 0.5 * float(stability)
    )


@dataclass
class InvertedDzcRescueDecision:
    cycle_idx: int
    applied: bool
    reason: str
    winner_sample: Optional[float] = None
    winner_type: Optional[str] = None
    rescued_sample: Optional[float] = None
    rescued_source: Optional[str] = None
    winner_voltage_percentile: float = float("nan")
    winner_neighborhood: float = float("nan")
    dzc_neighborhood: float = float("nan")


def try_inverted_dzc_rescue(
    *,
    ecg: np.ndarray,
    r_idx: int,
    s_i: int,
    q_next: int,
    fs: float,
    current_t: float,
    current_source: Optional[str],
    tmpl,
    cfg: ProcessCycleConfig,
    neighbor_t_samples: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> Tuple[Optional[float], InvertedDzcRescueDecision]:
    """
    Return (new_t_sample or None, decision).

    Winner type is the landmark-ensemble argmax (``is_ensemble_pick``).
    Voltage percentile gates the committed / exported sample (geometry winner),
    matching the counterfactual table which overwrote pick voltage with export
    geometry. When gates pass, substitutes the chosen DZC for ``current_t``.
    """
    morph = normalize_t_morphology_tag(
        tmpl.t_morphology if tmpl is not None and getattr(tmpl, "valid", False) else "unknown"
    )
    volt_thr = float(getattr(cfg, "record_inverted_dzc_rescue_volt_percentile_min", 95.0))

    dec = InvertedDzcRescueDecision(
        cycle_idx=-1,
        applied=False,
        reason="init",
        winner_sample=float(current_t) if np.isfinite(current_t) else None,
        winner_type=_primary_source(str(current_source or "")),
    )

    if morph != "inverted_t":
        dec.reason = "morphology_not_inverted"
        return None, dec
    if not np.isfinite(current_t):
        dec.reason = "missing_t"
        return None, dec

    # Voltage gate uses committed T (export geometry), not the ensemble pick sample.
    volt = stt_voltage_percentile(ecg, float(current_t), int(s_i), int(q_next))
    dec.winner_voltage_percentile = volt
    if not np.isfinite(volt) or volt <= volt_thr:
        dec.reason = "voltage_percentile_gate"
        return None, dec

    candidates, ctx = generate_t_candidates(
        ecg,
        int(r_idx),
        int(s_i),
        int(q_next),
        fs,
        tmpl=tmpl,
        cfg=cfg,
        neighbor_t_samples=neighbor_t_samples,
    )
    ensemble = filter_ensemble_candidates(candidates)
    if not ensemble:
        dec.reason = "no_candidates"
        return None, dec

    beat_corr = (
        _beat_template_correlation(ecg, int(s_i), int(q_next), tmpl)
        if tmpl is not None and getattr(tmpl, "valid", False)
        else float("nan")
    )
    template_landmark_rt_ms = None
    if tmpl is not None and getattr(tmpl, "valid", False):
        land_s, _ = morphology_rescue_landmark_global(int(s_i), int(q_next), tmpl, fs, cfg)
        if land_s is not None:
            template_landmark_rt_ms = (float(land_s) - float(r_idx)) * 1000.0 / fs

    # Neighborhood gate needs the evidence feature even if the base preset only
    # enables neighborhood globally (shape/stability may stay off).
    score_cfg = cfg
    if not bool(getattr(cfg, "record_template_prior_evidence_neighborhood", False)):
        score_cfg = dc_replace(cfg, record_template_prior_evidence_neighborhood=True)

    scored = score_landmark_ensemble(
        ensemble,
        ctx,
        ecg=ecg,
        template_landmark_rt_ms=template_landmark_rt_ms,
        beat_template_corr=beat_corr,
        cfg=score_cfg,
        neighbor_rts_ms=_neighbor_rts_ms(int(r_idx), fs, neighbor_t_samples),
    )
    if not scored:
        dec.reason = "no_scored_candidates"
        return None, dec

    # Ensemble pick == winner type for gating (matches counterfactual is_ensemble_pick).
    win_sc = scored[0]
    win_cand = next(
        (c for c in ensemble if int(c.sample_idx) == int(win_sc.sample_idx)),
        None,
    )
    if win_cand is None:
        dec.reason = "missing_winner_candidate"
        return None, dec

    win_primary = _primary_source(win_cand.source)
    dec.winner_type = win_primary
    if win_primary != "negative_peak":
        dec.reason = "winner_not_negative_peak"
        return None, dec

    dzc_cands = [c for c in ensemble if _primary_source(c.source) == "derivative_zero_crossing"]
    if not dzc_cands:
        dec.reason = "no_dzc_candidate"
        return None, dec

    if win_sc.evidence is None:
        dec.reason = "missing_winner_evidence"
        return None, dec
    win_neigh = float(win_sc.evidence.neighborhood_consistency)
    dec.winner_neighborhood = win_neigh
    if not np.isfinite(win_neigh):
        dec.reason = "winner_neighborhood_nan"
        return None, dec

    by_idx = {int(s.sample_idx): s for s in scored}
    eligible: List[Tuple[float, object, object]] = []
    for c in dzc_cands:
        sc = by_idx.get(int(c.sample_idx))
        if sc is None or sc.evidence is None:
            continue
        dn = float(sc.evidence.neighborhood_consistency)
        if not np.isfinite(dn) or not (dn > win_neigh):
            continue
        stab = float(sc.evidence.candidate_stability)
        if not np.isfinite(stab):
            stab = 0.0
        score = _dzc_evidence_score(
            dn,
            stab,
            float(sc.timing_prior_score),
            float(sc.derivative_support),
        )
        eligible.append((score, c, sc))

    if not eligible:
        dec.reason = "no_dzc_with_better_neighborhood"
        return None, dec

    eligible.sort(key=lambda x: x[0], reverse=True)
    best_c = eligible[0][1]
    best_sc = eligible[0][2]
    dec.dzc_neighborhood = float(best_sc.evidence.neighborhood_consistency)
    dec.applied = True
    dec.reason = "substituted"
    dec.rescued_sample = float(best_c.sample_idx)
    dec.rescued_source = _primary_source(best_c.source)
    return float(best_c.sample_idx), dec


def apply_inverted_dzc_rescue_pass(
    output_dict: dict,
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    cycles: List[int],
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    template_prior_by_cycle: dict,
    record_template,
) -> Dict[str, int]:
    """Post-selection pass: substitute T when inverted DZC rescue gates pass."""
    stats = {
        "attempted": 0,
        "eligible_morph": 0,
        "rescued": 0,
        "gated": 0,
        "no_prior": 0,
    }
    if not getattr(cfg, "record_inverted_dzc_rescue", False):
        return stats

    ecg_delim = prepare_record_delineation_signal(ecg, sampling_rate, cfg)
    t_list = output_dict.get("T_global_center_idx", [])
    t_source = output_dict.get("t_source", None)

    for cycle_idx, cycle_label in enumerate(cycles):
        prior = template_prior_by_cycle.get(cycle_idx)
        if prior is None or cycle_idx >= len(t_list):
            stats["no_prior"] += 1
            continue
        epoch_i = int(cycle_label)
        if epoch_i < 0 or epoch_i >= len(r_peaks):
            continue
        s_i, q_next = prior.s_i, prior.q_next
        if s_i is None or q_next is None:
            continue

        cur = t_list[cycle_idx]
        if cur is None or not np.isfinite(cur):
            continue

        stats["attempted"] += 1
        morph = normalize_t_morphology_tag(
            record_template.t_morphology
            if record_template is not None and getattr(record_template, "valid", False)
            else "unknown"
        )
        if morph == "inverted_t":
            stats["eligible_morph"] += 1

        prev_t = t_list[cycle_idx - 1] if cycle_idx > 0 else None
        next_t = t_list[cycle_idx + 1] if cycle_idx + 1 < len(t_list) else None
        neighbor = (
            float(prev_t) if prev_t is not None and np.isfinite(prev_t) else None,
            float(next_t) if next_t is not None and np.isfinite(next_t) else None,
        )
        src = None
        if isinstance(t_source, list) and cycle_idx < len(t_source):
            src = t_source[cycle_idx]

        new_t, dec = try_inverted_dzc_rescue(
            ecg=ecg_delim,
            r_idx=int(r_peaks[epoch_i]),
            s_i=int(s_i),
            q_next=int(q_next),
            fs=float(sampling_rate),
            current_t=float(cur),
            current_source=src,
            tmpl=record_template,
            cfg=cfg,
            neighbor_t_samples=neighbor,
        )
        if new_t is None:
            if dec.reason not in ("morphology_not_inverted", "missing_t"):
                stats["gated"] += 1
            continue

        t_list[cycle_idx] = float(new_t)
        if isinstance(t_source, list) and cycle_idx < len(t_source):
            t_source[cycle_idx] = f"inverted_dzc_rescue:{dec.rescued_source}"
        stats["rescued"] += 1

    output_dict["T_global_center_idx"] = t_list
    return stats
