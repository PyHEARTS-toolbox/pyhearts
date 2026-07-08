"""
Template-prior T candidate rescue (Phase 2A).

When per-cycle T is missing or disagrees strongly with the morphology-aware template
landmark, search scored candidates near the landmark and accept if plausible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.delineation_signal import prepare_record_delineation_signal
from pyhearts.processing.record_delineation import MedianBeatTemplate
from pyhearts.processing.t_candidate_scoring import (
    generate_t_candidates,
    nearest_candidate,
    pick_best_t_candidate,
)
from pyhearts.processing.t_landmark_ensemble import pick_ensemble_t_candidate
from pyhearts.processing.t_morphology_routing import (
    morphology_rescue_landmark_global,
    morphology_scoring_weights,
    rescue_candidate_passes_plausibility,
)
from pyhearts.processing.template_prior_windows import TemplatePriorBeatWindows


@dataclass(frozen=True)
class TRescueDecision:
    cycle_idx: int
    applied: bool
    reason: str
    trigger: str
    landmark_sample: Optional[int] = None
    landmark_source: str = ""
    rescued_sample: Optional[int] = None
    rescued_source: str = ""
    per_cycle_sample: Optional[float] = None
    dispute_ms: float = float("nan")


def _finite(val) -> bool:
    return val is not None and np.isfinite(val)


def _ms_to_samples(ms: float, fs: float) -> int:
    return max(1, int(round(ms * fs / 1000.0)))


def _dispute_ms(
    per_cycle_t: Optional[float],
    landmark: Optional[int],
    fs: float,
) -> float:
    if not _finite(per_cycle_t) or landmark is None:
        return float("inf")
    return abs(float(per_cycle_t) - float(landmark)) / fs * 1000.0


def should_attempt_t_rescue(
    per_cycle_t: Optional[float],
    landmark: Optional[int],
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> Tuple[bool, str]:
    if not cfg.record_template_prior_rescue:
        return False, "disabled"
    if not _finite(per_cycle_t):
        return True, "missing_t"
    dispute = _dispute_ms(per_cycle_t, landmark, sampling_rate)
    if dispute > float(cfg.record_template_prior_rescue_dispute_ms):
        return True, "dispute"
    return False, "ok"


def rescue_t_for_beat(
    ecg: np.ndarray,
    *,
    r_idx: int,
    prior: TemplatePriorBeatWindows,
    tmpl: MedianBeatTemplate,
    per_cycle_t: Optional[float],
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    neighbor_t: Optional[Tuple[Optional[int], Optional[int]]] = None,
) -> Tuple[Optional[int], TRescueDecision]:
    """
    Attempt morphology-routed candidate rescue for one beat.

    Returns (global_sample_or_none, decision).
    """
    s_i = prior.s_i
    q_next = prior.q_next
    if s_i is None or q_next is None:
        return None, TRescueDecision(
            cycle_idx=-1, applied=False, reason="no_s_q_anchor", trigger="none"
        )

    landmark, land_src = morphology_rescue_landmark_global(
        int(s_i), int(q_next), tmpl, sampling_rate, cfg
    )
    attempt, trigger = should_attempt_t_rescue(
        per_cycle_t, landmark, sampling_rate, cfg
    )
    dispute = _dispute_ms(per_cycle_t, landmark, sampling_rate)
    base = TRescueDecision(
        cycle_idx=-1,
        applied=False,
        reason="no_attempt",
        trigger=trigger,
        landmark_sample=landmark,
        landmark_source=land_src,
        per_cycle_sample=float(per_cycle_t) if _finite(per_cycle_t) else None,
        dispute_ms=dispute,
    )
    if not attempt:
        return int(per_cycle_t) if _finite(per_cycle_t) else None, base

    if landmark is None:
        return None, TRescueDecision(
            cycle_idx=-1,
            applied=False,
            reason="no_landmark",
            trigger=trigger,
            per_cycle_sample=base.per_cycle_sample,
            dispute_ms=dispute,
        )

    candidates, ctx = generate_t_candidates(
        ecg,
        int(r_idx),
        int(s_i),
        int(q_next),
        sampling_rate,
        tmpl=tmpl,
        cfg=cfg,
        neighbor_t_samples=neighbor_t,
    )
    if not candidates:
        return None, TRescueDecision(
            cycle_idx=-1,
            applied=False,
            reason="no_candidate",
            trigger=trigger,
            landmark_sample=landmark,
            landmark_source=land_src,
            per_cycle_sample=base.per_cycle_sample,
            dispute_ms=dispute,
        )

    radius = _ms_to_samples(cfg.record_template_prior_rescue_radius_ms, sampling_rate)
    near = [c for c in candidates if abs(c.sample_idx - landmark) <= radius]
    if not near:
        near_cand, _ = nearest_candidate(float(landmark), candidates)
        if near_cand is not None:
            near = [near_cand]

    weights = (
        morphology_scoring_weights(tmpl.t_morphology)
        if cfg.record_template_prior_morphology_routing
        and not cfg.record_template_prior_landmark_ensemble
        else None
    )
    if cfg.record_template_prior_landmark_ensemble:
        ranker_enabled = bool(getattr(cfg, "record_template_prior_learned_ranker", False))
        if ranker_enabled:
            from pyhearts.processing.t_candidate_ranker import pick_ranked_t_candidate_from_ensemble

            model = getattr(cfg, "_active_ranker_model", None)
            best, _ = pick_ranked_t_candidate_from_ensemble(
                ecg,
                int(r_idx),
                int(s_i),
                int(q_next),
                sampling_rate,
                tmpl,
                cfg,
                neighbor_t_samples=neighbor_t,
                model=model,
            )
        else:
            best, _ = pick_ensemble_t_candidate(
                ecg,
                int(r_idx),
                int(s_i),
                int(q_next),
                sampling_rate,
                tmpl,
                cfg,
                neighbor_t_samples=neighbor_t,
            )
    else:
        best = pick_best_t_candidate(near or candidates, ctx, weights=weights)
    if best is None:
        return None, TRescueDecision(
            cycle_idx=-1,
            applied=False,
            reason="no_scored_candidate",
            trigger=trigger,
            landmark_sample=landmark,
            landmark_source=land_src,
            per_cycle_sample=base.per_cycle_sample,
            dispute_ms=dispute,
        )

    if not rescue_candidate_passes_plausibility(
        best.rt_ms,
        best.signed_amp,
        tmpl,
        cfg,
        prominence=best.prominence,
        st_baseline=ctx.baseline,
    ):
        return (
            int(per_cycle_t) if _finite(per_cycle_t) else None,
            TRescueDecision(
                cycle_idx=-1,
                applied=False,
                reason="candidate_rejected",
                trigger=trigger,
                landmark_sample=landmark,
                landmark_source=land_src,
                per_cycle_sample=base.per_cycle_sample,
                dispute_ms=dispute,
                rescued_sample=best.sample_idx,
                rescued_source=best.source,
            ),
        )

    if trigger == "dispute" and _finite(per_cycle_t):
        per_d = abs(float(per_cycle_t) - float(landmark))
        resc_d = abs(float(best.sample_idx) - float(landmark))
        if resc_d >= per_d:
            return int(per_cycle_t), TRescueDecision(
                cycle_idx=-1,
                applied=False,
                reason="kept_closer_per_cycle",
                trigger=trigger,
                landmark_sample=landmark,
                landmark_source=land_src,
                per_cycle_sample=base.per_cycle_sample,
                dispute_ms=dispute,
                rescued_sample=best.sample_idx,
                rescued_source=best.source,
            )

    return best.sample_idx, TRescueDecision(
        cycle_idx=-1,
        applied=True,
        reason="rescued",
        trigger=trigger,
        landmark_sample=landmark,
        landmark_source=land_src,
        rescued_sample=best.sample_idx,
        rescued_source=best.source,
        per_cycle_sample=base.per_cycle_sample,
        dispute_ms=dispute,
    )


def apply_template_prior_t_rescue_pass(
    output_dict: Dict,
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    cycles: List[int],
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    template_prior_by_cycle: Dict[int, TemplatePriorBeatWindows],
    record_template: Optional[MedianBeatTemplate],
    cluster_templates: Optional[Dict[int, MedianBeatTemplate]] = None,
    cluster_by_epoch: Optional[Dict[int, int]] = None,
) -> Tuple[Dict[str, int], List[TRescueDecision]]:
    """
    Post-cycle rescue pass: update ``T_global_center_idx`` when rescue accepts.
    """
    stats = {
        "attempted": 0,
        "rescued": 0,
        "missing_trigger": 0,
        "dispute_trigger": 0,
        "no_candidate": 0,
        "candidate_rejected": 0,
        "kept_per_cycle": 0,
    }
    decisions: List[TRescueDecision] = []
    if not cfg.record_template_prior_rescue or not template_prior_by_cycle:
        return stats, decisions

    ecg_delim = prepare_record_delineation_signal(ecg, sampling_rate, cfg)
    t_list = output_dict.get("T_global_center_idx", [])
    r_list = output_dict.get("R_global_center_idx", [])
    t_source = output_dict.get("t_source", None)

    for cycle_idx, cycle_label in enumerate(cycles):
        prior = template_prior_by_cycle.get(cycle_idx)
        if prior is None or cycle_idx >= len(t_list):
            continue

        epoch_i = int(cycle_label)
        if epoch_i < 0 or epoch_i >= len(r_peaks):
            continue

        tmpl = record_template
        if (
            cluster_templates
            and cluster_by_epoch is not None
            and epoch_i in cluster_by_epoch
        ):
            tmpl = cluster_templates.get(cluster_by_epoch[epoch_i], tmpl)
        if tmpl is None or not tmpl.valid:
            continue

        r_idx = int(r_peaks[epoch_i])
        per_cycle_t = t_list[cycle_idx]
        prev_t = t_list[cycle_idx - 1] if cycle_idx > 0 else None
        next_t = t_list[cycle_idx + 1] if cycle_idx + 1 < len(t_list) else None
        rescued, dec = rescue_t_for_beat(
            ecg_delim,
            r_idx=r_idx,
            prior=prior,
            tmpl=tmpl,
            per_cycle_t=float(per_cycle_t) if _finite(per_cycle_t) else None,
            sampling_rate=sampling_rate,
            cfg=cfg,
            neighbor_t=(
                int(prev_t) if _finite(prev_t) else None,
                int(next_t) if _finite(next_t) else None,
            ),
        )
        dec = TRescueDecision(
            cycle_idx=cycle_idx,
            applied=dec.applied,
            reason=dec.reason,
            trigger=dec.trigger,
            landmark_sample=dec.landmark_sample,
            landmark_source=dec.landmark_source,
            rescued_sample=dec.rescued_sample,
            rescued_source=dec.rescued_source,
            per_cycle_sample=dec.per_cycle_sample,
            dispute_ms=dec.dispute_ms,
        )
        decisions.append(dec)

        if dec.trigger == "missing_t":
            stats["missing_trigger"] += 1
        elif dec.trigger == "dispute":
            stats["dispute_trigger"] += 1

        if dec.trigger in ("missing_t", "dispute"):
            stats["attempted"] += 1

        if dec.reason == "no_candidate":
            stats["no_candidate"] += 1
        elif dec.reason == "candidate_rejected":
            stats["candidate_rejected"] += 1

        if dec.applied and rescued is not None:
            t_list[cycle_idx] = float(rescued)
            if isinstance(t_source, list) and cycle_idx < len(t_source):
                t_source[cycle_idx] = f"template_prior_rescue:{dec.rescued_source}"
            stats["rescued"] += 1
        elif _finite(per_cycle_t):
            stats["kept_per_cycle"] += 1

    output_dict["T_global_center_idx"] = t_list
    return stats, decisions
