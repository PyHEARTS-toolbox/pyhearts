"""
Per-beat T landmark ensemble with decomposed candidate scoring.

Generates morphology-diverse landmark hypotheses (peaks, derivative features,
plateau, template projection) and ranks them with soft priors — the template
shifts probability; it does not dictate the answer.

final_score =
  morphology_score
+ timing_prior_score
+ derivative_support
+ template_prior_score
+ prominence_score
+ beat_corr_support
+ local_confidence
+ evidence_score (local shape / neighborhood / stability; unit-scale ablation)
- implausibility_penalties
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.record_delineation import MedianBeatTemplate
from pyhearts.processing.t_candidate_scoring import (
    TBeatCandidateContext,
    TCandidate,
    generate_t_candidates,
    pick_best_t_candidate,
)
from pyhearts.processing.t_landmark_evidence import (
    LandmarkEvidence,
    compute_landmark_evidence,
)
from pyhearts.processing.t_morphology_routing import normalize_t_morphology_tag
from pyhearts.processing.template_prior_window_diagnostics import (
    _beat_template_correlation,
)

ENSEMBLE_SOURCES = frozenset(
    {
        "positive_peak",
        "negative_peak",
        "derivative_zero_crossing",
        "max_curvature",
        "plateau_midpoint",
        "amplitude_plateau_midpoint",
        "refined_apex",
        "refined_dzc",
        "template_projected_apex",
        "template_landmark",
        "rising_edge_onset",
        "post_positive_shoulder",
    }
)

# Morphology → source-type prior bonuses (rule-weighted, not learned).
_SOURCE_MORPHOLOGY_PRIORS: Dict[str, Dict[str, float]] = {
    "normal": {
        "positive_peak": 0.18,
        "refined_apex": 0.16,
        "derivative_zero_crossing": 0.10,
        "refined_dzc": 0.09,
        "max_curvature": 0.06,
        "plateau_midpoint": 0.04,
        "amplitude_plateau_midpoint": 0.05,
        "template_projected_apex": 0.12,
        "template_landmark": 0.10,
        "negative_peak": 0.02,
        "rising_edge_onset": 0.05,
    },
    "inverted_t": {
        "negative_peak": 0.28,
        "refined_apex": 0.12,
        "derivative_zero_crossing": 0.12,
        "refined_dzc": 0.10,
        "max_curvature": 0.08,
        "amplitude_plateau_midpoint": 0.06,
        "template_projected_apex": 0.14,
        "template_landmark": 0.12,
        "positive_peak": -0.05,
    },
    "rising_edge": {
        "rising_edge_onset": 0.22,
        "derivative_zero_crossing": 0.14,
        "refined_dzc": 0.12,
        "positive_peak": 0.10,
        "refined_apex": 0.10,
        "max_curvature": 0.06,
        "template_projected_apex": 0.10,
        "post_positive_shoulder": 0.08,
    },
    "plateau": {
        "plateau_midpoint": 0.24,
        "amplitude_plateau_midpoint": 0.22,
        "derivative_zero_crossing": 0.16,
        "refined_dzc": 0.12,
        "positive_peak": 0.10,
        "refined_apex": 0.08,
        "max_curvature": 0.05,
        "template_projected_apex": 0.10,
    },
    "biphasic_positive_negative": {
        "negative_peak": 0.20,
        "positive_peak": 0.14,
        "refined_apex": 0.12,
        "derivative_zero_crossing": 0.10,
        "refined_dzc": 0.09,
        "amplitude_plateau_midpoint": 0.06,
        "post_positive_shoulder": 0.08,
        "template_projected_apex": 0.10,
        "template_landmark": 0.08,
    },
}


@dataclass(frozen=True)
class TLandmarkEnsembleWeights:
    """Rule-weighted linear ensemble (tune via leave-records-out validation)."""

    timing_prior_per_ms: float = -0.014
    template_prior_per_ms: float = -0.010
    template_prior_cap: float = 0.35
    prominence: float = 0.30
    derivative_support: float = 0.12
    beat_corr_support: float = 0.15
    local_confidence: float = 0.10
    sign_consistency: float = 0.22
    terminal_penalty: float = 0.18
    before_first_pos_per_ms: float = -0.025
    low_prominence_penalty: float = 0.20
    rt_oob_penalty: float = 0.35


@dataclass
class TLandmarkEnsembleScore:
    """Decomposed score for one landmark candidate."""

    sample_idx: int
    source: str
    morphology_score: float = 0.0
    timing_prior_score: float = 0.0
    derivative_support: float = 0.0
    template_prior_score: float = 0.0
    prominence_score: float = 0.0
    beat_corr_support: float = 0.0
    local_confidence: float = 0.0
    evidence_score: float = 0.0
    implausibility_penalty: float = 0.0
    total: float = 0.0
    evidence: Optional[LandmarkEvidence] = None
    features: Dict[str, float] = field(default_factory=dict)

    def as_dict(self, prefix: str = "") -> Dict[str, float]:
        p = prefix
        return {
            f"{p}sample_idx": float(self.sample_idx),
            f"{p}source": self.source,
            f"{p}morphology_score": self.morphology_score,
            f"{p}timing_prior_score": self.timing_prior_score,
            f"{p}derivative_support": self.derivative_support,
            f"{p}template_prior_score": self.template_prior_score,
            f"{p}prominence_score": self.prominence_score,
            f"{p}beat_corr_support": self.beat_corr_support,
            f"{p}local_confidence": self.local_confidence,
            f"{p}evidence_score": self.evidence_score,
            f"{p}implausibility_penalty": self.implausibility_penalty,
            f"{p}total": self.total,
        }


def _primary_source(source: str) -> str:
    return source.split("+")[0].strip()


def _gaussian_timing_score(dist_ms: float, sigma_ms: float) -> float:
    if not np.isfinite(dist_ms):
        return 0.0
    z = float(dist_ms) / max(sigma_ms, 1e-6)
    return float(np.exp(-0.5 * z * z))


def _morphology_source_bonus(source: str, morphology: str, cfg: ProcessCycleConfig) -> float:
    tag = normalize_t_morphology_tag(morphology)
    priors = _SOURCE_MORPHOLOGY_PRIORS.get(tag, _SOURCE_MORPHOLOGY_PRIORS["normal"])
    base = float(priors.get(_primary_source(source), 0.0))
    if tag == "biphasic_positive_negative":
        lobe = getattr(cfg, "record_template_prior_biphasic_rescue_lobe", "negative")
        if lobe == "positive" and _primary_source(source) == "positive_peak":
            base += 0.08
        if lobe == "negative" and _primary_source(source) == "negative_peak":
            base += 0.08
    return base


def score_landmark_candidate(
    candidate: TCandidate,
    ctx: TBeatCandidateContext,
    *,
    template_landmark_rt_ms: Optional[float] = None,
    beat_template_corr: float = float("nan"),
    timing_sigma_ms: float = 45.0,
    template_sigma_ms: float = 55.0,
    weights: Optional[TLandmarkEnsembleWeights] = None,
    cfg: Optional[ProcessCycleConfig] = None,
) -> TLandmarkEnsembleScore:
    """Decomposed ensemble score for one landmark hypothesis."""
    w = weights or TLandmarkEnsembleWeights()
    cfg = cfg or ProcessCycleConfig()
    src = _primary_source(candidate.source)

    morphology_score = _morphology_source_bonus(
        candidate.source, ctx.template_morphology, cfg
    )
    if candidate.sign_matches_template:
        morphology_score += w.sign_consistency * 0.5

    dist_expected = abs(
        float(candidate.features.get("dist_expected_rt_ms", candidate.rt_ms - ctx.expected_rt_ms))
    )
    timing_prior_score = _gaussian_timing_score(dist_expected, timing_sigma_ms)

    template_prior_score = 0.0
    if template_landmark_rt_ms is not None and np.isfinite(template_landmark_rt_ms):
        dist_tpl = abs(candidate.rt_ms - float(template_landmark_rt_ms))
        template_prior_score = min(
            w.template_prior_cap,
            _gaussian_timing_score(dist_tpl, template_sigma_ms),
        )

    derivative_support = 0.0
    if src in ("derivative_zero_crossing", "max_curvature", "rising_edge_onset"):
        derivative_support = w.derivative_support
    elif src == "plateau_midpoint":
        derivative_support = w.derivative_support * 0.5

    prominence_score = w.prominence * min(1.0, candidate.prominence / 0.15)

    curv_norm = min(1.0, candidate.curvature / max(1e-6, candidate.curvature))
    local_confidence = w.local_confidence * (
        0.6 * min(1.0, candidate.prominence / 0.15) + 0.4 * curv_norm
    )

    beat_corr_support = 0.0
    if np.isfinite(beat_template_corr):
        beat_corr_support = w.beat_corr_support * max(0.0, beat_template_corr)

    penalty = 0.0
    if candidate.is_terminal:
        penalty += w.terminal_penalty
    if candidate.before_first_pos and ctx.first_pos_rt_ms is not None:
        penalty += w.before_first_pos_per_ms * max(
            0.0, float(ctx.first_pos_rt_ms) - candidate.rt_ms
        )
        penalty = abs(penalty)
    if not candidate.sign_matches_template:
        penalty += w.sign_consistency * 0.5
    if candidate.prominence < 0.03:
        penalty += w.low_prominence_penalty
    rt_min, rt_max = ctx.t_window_rt_lo_ms, ctx.t_window_rt_hi_ms
    if candidate.rt_ms < rt_min or candidate.rt_ms > rt_max:
        penalty += w.rt_oob_penalty

    total = (
        morphology_score
        + timing_prior_score
        + derivative_support
        + template_prior_score
        + prominence_score
        + beat_corr_support
        + local_confidence
        - penalty
    )

    return TLandmarkEnsembleScore(
        sample_idx=int(candidate.sample_idx),
        source=candidate.source,
        morphology_score=float(morphology_score),
        timing_prior_score=float(timing_prior_score),
        derivative_support=float(derivative_support),
        template_prior_score=float(template_prior_score),
        prominence_score=float(prominence_score),
        beat_corr_support=float(beat_corr_support),
        local_confidence=float(local_confidence),
        implausibility_penalty=float(penalty),
        total=float(total),
        features={
            "rt_ms": float(candidate.rt_ms),
            "signed_amp": float(candidate.signed_amp),
            "prominence": float(candidate.prominence),
            "beat_template_corr": float(beat_template_corr)
            if np.isfinite(beat_template_corr)
            else float("nan"),
            "dist_expected_rt_ms": float(dist_expected),
        },
    )


def score_landmark_ensemble(
    candidates: Sequence[TCandidate],
    ctx: TBeatCandidateContext,
    *,
    ecg: Optional[np.ndarray] = None,
    template_landmark_rt_ms: Optional[float] = None,
    beat_template_corr: float = float("nan"),
    timing_sigma_ms: float = 45.0,
    template_sigma_ms: float = 55.0,
    weights: Optional[TLandmarkEnsembleWeights] = None,
    cfg: Optional[ProcessCycleConfig] = None,
    neighbor_rts_ms: Optional[Sequence[float]] = None,
) -> List[TLandmarkEnsembleScore]:
    cfg = cfg or ProcessCycleConfig()
    scored = [
        score_landmark_candidate(
            c,
            ctx,
            template_landmark_rt_ms=template_landmark_rt_ms,
            beat_template_corr=beat_template_corr,
            timing_sigma_ms=timing_sigma_ms,
            template_sigma_ms=template_sigma_ms,
            weights=weights,
            cfg=cfg,
        )
        for c in candidates
    ]

    use_shape = bool(getattr(cfg, "record_template_prior_evidence_local_shape", False))
    use_nbr = bool(getattr(cfg, "record_template_prior_evidence_neighborhood", False))
    use_stab = bool(getattr(cfg, "record_template_prior_evidence_stability", False))

    if ecg is not None and (use_shape or use_nbr or use_stab) and scored:
        base_totals = {s.sample_idx: s.total for s in scored}

        def _base_score(cand: TCandidate) -> float:
            return float(base_totals.get(int(cand.sample_idx), 0.0))

        updated: List[TLandmarkEnsembleScore] = []
        nbr_tol = float(
            getattr(cfg, "record_template_prior_evidence_neighborhood_tolerance_ms", 50.0)
        )
        nbr_sig = float(
            getattr(cfg, "record_template_prior_evidence_neighborhood_sigma_ms", 45.0)
        )
        nbr_max_pen = float(
            getattr(cfg, "record_template_prior_evidence_neighborhood_max_penalty", 0.15)
        )
        for c, s in zip(candidates, scored):
            ev = compute_landmark_evidence(
                ecg,
                c,
                ctx,
                candidates,
                base_score_fn=_base_score,
                neighbor_rts_ms=neighbor_rts_ms,
                neighborhood_tolerance_ms=nbr_tol,
                neighborhood_sigma_ms=nbr_sig,
            )
            ev_score = ev.enabled_sum(
                shape=use_shape,
                neighborhood=False,
                stability=use_stab,
            )
            # Neighborhood: outlier-only slight down-weight — never smooths T_i.
            if use_nbr:
                ev_score -= nbr_max_pen * ev.neighborhood_penalty()
            feats = dict(s.features)
            feats.update(ev.as_dict())
            feats["neighborhood_penalty"] = ev.neighborhood_penalty()
            updated.append(
                TLandmarkEnsembleScore(
                    sample_idx=s.sample_idx,
                    source=s.source,
                    morphology_score=s.morphology_score,
                    timing_prior_score=s.timing_prior_score,
                    derivative_support=s.derivative_support,
                    template_prior_score=s.template_prior_score,
                    prominence_score=s.prominence_score,
                    beat_corr_support=s.beat_corr_support,
                    local_confidence=s.local_confidence,
                    evidence_score=float(ev_score),
                    implausibility_penalty=s.implausibility_penalty,
                    total=float(s.total + ev_score),
                    evidence=ev,
                    features=feats,
                )
            )
        scored = updated

    scored.sort(key=lambda s: s.total, reverse=True)
    return scored


def filter_ensemble_candidates(candidates: Sequence[TCandidate]) -> List[TCandidate]:
    """Keep canonical ensemble source types."""
    out: List[TCandidate] = []
    for c in candidates:
        primary = _primary_source(c.source)
        if primary in ENSEMBLE_SOURCES:
            out.append(c)
    return out if out else list(candidates)


def _neighbor_rts_ms(
    r_idx: int,
    fs: float,
    neighbor_t_samples: Optional[Tuple[Optional[int], Optional[int]]],
) -> List[float]:
    if not neighbor_t_samples:
        return []
    out: List[float] = []
    for t in neighbor_t_samples:
        if t is not None and np.isfinite(t):
            out.append((float(t) - float(r_idx)) * 1000.0 / fs)
    return out


def pick_ensemble_landmark(
    ecg: np.ndarray,
    r_idx: int,
    s_i: int,
    q_next: int,
    fs: float,
    tmpl: Optional[MedianBeatTemplate],
    cfg: ProcessCycleConfig,
    *,
    neighbor_t_samples: Optional[Tuple[Optional[int], Optional[int]]] = None,
    timing_sigma_ms: float = 45.0,
    template_sigma_ms: float = 55.0,
    weights: Optional[TLandmarkEnsembleWeights] = None,
) -> Tuple[Optional[int], str, List[TLandmarkEnsembleScore]]:
    """
    Generate landmark ensemble and return (sample, source, scored_list).

    Template contributes a soft prior via ``template_prior_score``, not hard routing.
    """
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
        return None, "none", []

    beat_corr = float("nan")
    if tmpl is not None and tmpl.valid:
        beat_corr = _beat_template_correlation(ecg, int(s_i), int(q_next), tmpl)

    template_landmark_rt_ms: Optional[float] = None
    if tmpl is not None and tmpl.valid and tmpl.t_landmark_idx is not None:
        from pyhearts.processing.t_morphology_routing import morphology_rescue_landmark_global

        land_s, _ = morphology_rescue_landmark_global(
            int(s_i), int(q_next), tmpl, fs, cfg
        )
        if land_s is not None:
            template_landmark_rt_ms = (float(land_s) - float(r_idx)) * 1000.0 / fs

    scored = score_landmark_ensemble(
        ensemble,
        ctx,
        ecg=ecg,
        template_landmark_rt_ms=template_landmark_rt_ms,
        beat_template_corr=beat_corr,
        timing_sigma_ms=timing_sigma_ms,
        template_sigma_ms=template_sigma_ms,
        weights=weights,
        cfg=cfg,
        neighbor_rts_ms=_neighbor_rts_ms(int(r_idx), fs, neighbor_t_samples),
    )
    if not scored:
        return None, "none", []

    best = scored[0]
    return int(best.sample_idx), best.source, scored


def pick_ensemble_t_candidate(
    ecg: np.ndarray,
    r_idx: int,
    s_i: int,
    q_next: int,
    fs: float,
    tmpl: Optional[MedianBeatTemplate],
    cfg: ProcessCycleConfig,
    *,
    neighbor_t_samples: Optional[Tuple[Optional[int], Optional[int]]] = None,
    weights: Optional[TLandmarkEnsembleWeights] = None,
) -> Tuple[Optional[TCandidate], List[TLandmarkEnsembleScore]]:
    """Pick best T candidate using ensemble scoring (for rescue / window center)."""
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
        legacy = pick_best_t_candidate(candidates, ctx)
        return legacy, []

    beat_corr = float("nan")
    if tmpl is not None and tmpl.valid:
        beat_corr = _beat_template_correlation(ecg, int(s_i), int(q_next), tmpl)

    template_landmark_rt_ms: Optional[float] = None
    if tmpl is not None and tmpl.valid and tmpl.t_landmark_idx is not None:
        from pyhearts.processing.t_morphology_routing import morphology_rescue_landmark_global

        land_s, _ = morphology_rescue_landmark_global(
            int(s_i), int(q_next), tmpl, fs, cfg
        )
        if land_s is not None:
            template_landmark_rt_ms = (float(land_s) - float(r_idx)) * 1000.0 / fs

    scored = score_landmark_ensemble(
        ensemble,
        ctx,
        ecg=ecg,
        template_landmark_rt_ms=template_landmark_rt_ms,
        beat_template_corr=beat_corr,
        weights=weights,
        cfg=cfg,
        neighbor_rts_ms=_neighbor_rts_ms(int(r_idx), fs, neighbor_t_samples),
    )
    if not scored:
        return None, scored

    if getattr(cfg, "record_template_prior_learned_ranker", False):
        from pyhearts.processing.t_candidate_ranker import pick_ranked_t_candidate
        from pyhearts.processing.t_morphology_routing import normalize_t_morphology_tag

        morph = normalize_t_morphology_tag(
            tmpl.t_morphology if tmpl is not None and tmpl.valid else ctx.template_morphology
        )
        model = getattr(cfg, "_active_ranker_model", None)
        best_c, scored, _ = pick_ranked_t_candidate(
            scored, ensemble, morph, cfg, model=model
        )
        return best_c, scored

    best_s = scored[0]
    best_c = next((c for c in ensemble if c.sample_idx == best_s.sample_idx), ensemble[0])
    return best_c, scored
