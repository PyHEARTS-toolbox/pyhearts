"""
Independent evidence features for T landmark candidates.

Three information sources humans use that point-wise scoring misses:

1. **Local wave shape** — does ±30 ms around the candidate resemble a T apex?
2. **Neighborhood consistency** — soft outlier down-weight only (never smooths ``T_i``).
3. **Candidate stability** — does this candidate win under ±perturbation?

Neighborhood is **not** ``T_i = average(T_{i-1}, T_i, T_{i+1})``. It only scores each
candidate: in-range RT vs neighbors is neutral; far-outlier candidates are slightly
less likely. Beat-to-beat physiological variation is preserved.

Features are computed on a unit scale [0, 1] for ablation (on/off) and future
leave-record-out learning (logistic / regularized GBT), not manual weight tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from pyhearts.processing.t_candidate_scoring import TCandidate, TBeatCandidateContext


@dataclass(frozen=True)
class LandmarkEvidence:
    """Measured evidence (not hand-weighted score components)."""

    local_shape_score: float = float("nan")
    neighborhood_consistency: float = float("nan")
    candidate_stability: float = float("nan")

    def neighborhood_penalty(self) -> float:
        """
        Outlier-only down-weight in [0, 1]. Zero when RT is within normal neighbor variation.

        Never modifies beat timing — used only to slightly penalize outlier candidates.
        """
        if not np.isfinite(self.neighborhood_consistency):
            return 0.0
        return float(np.clip(1.0 - self.neighborhood_consistency, 0.0, 1.0))

    def enabled_sum(self, *, shape: bool, neighborhood: bool, stability: bool) -> float:
        """Unit-scale positive evidence (shape, stability). Neighborhood handled separately."""
        total = 0.0
        n = 0
        if shape and np.isfinite(self.local_shape_score):
            total += float(self.local_shape_score)
            n += 1
        if stability and np.isfinite(self.candidate_stability):
            total += float(self.candidate_stability)
            n += 1
        return total / n if n else 0.0

    def as_dict(self) -> dict:
        return {
            "local_shape_score": self.local_shape_score,
            "neighborhood_consistency": self.neighborhood_consistency,
            "candidate_stability": self.candidate_stability,
        }


def _ms_to_samples(ms: float, fs: float) -> int:
    return max(1, int(round(ms * fs / 1000.0)))


def local_t_apex_shape_score(
    ecg: np.ndarray,
    center_sample: int,
    fs: float,
    baseline: float,
    *,
    half_width_ms: float = 30.0,
) -> float:
    """
    Does the ±half_width_ms window around ``center_sample`` resemble a T apex?

    Combines apex centrality, derivative zero-crossing proximity, curvature
    concentration, and rise/fall symmetry. Returns [0, 1].
    """
    half = _ms_to_samples(half_width_ms, fs)
    lo = max(0, int(center_sample) - half)
    hi = min(len(ecg), int(center_sample) + half + 1)
    if hi - lo < 7:
        return 0.0

    seg = ecg[lo:hi].astype(float, copy=False) - float(baseline)
    center_rel = int(center_sample) - lo
    n = seg.size
    if center_rel < 1 or center_rel >= n - 1:
        return 0.0

    abs_seg = np.abs(seg)
    apex_i = int(np.argmax(abs_seg))
    centrality = 1.0 - min(1.0, abs(apex_i - center_rel) / max(1.0, n * 0.35))

    d1 = np.gradient(seg)
    d2 = np.gradient(d1)
    zero_dists: List[float] = []
    for i in range(1, d1.size):
        if d1[i - 1] > 0 and d1[i] <= 0:
            zero_dists.append(abs(i - center_rel))
        if d1[i - 1] < 0 and d1[i] >= 0:
            zero_dists.append(abs(i - center_rel))
    if zero_dists:
        deriv_score = float(np.exp(-0.5 * (min(zero_dists) / max(2.0, n * 0.15)) ** 2))
    else:
        deriv_score = 0.3 * centrality

    curv_i = int(np.argmax(np.abs(d2[1:-1]))) + 1 if d2.size > 2 else center_rel
    curvature_focus = 1.0 - min(1.0, abs(curv_i - center_rel) / max(1.0, n * 0.35))

    left = seg[:center_rel]
    right = seg[center_rel + 1 :]
    sym_score = 0.5
    if left.size >= 2 and right.size >= 2:
        rise = float(np.max(np.gradient(left))) if left.size else 0.0
        fall = float(np.min(np.gradient(right))) if right.size else 0.0
        denom = max(abs(rise), abs(fall), 1e-6)
        sym_score = 1.0 - min(1.0, abs(abs(rise) - abs(fall)) / denom)

    prom = float(abs_seg[center_rel])
    prom_ref = float(np.max(abs_seg)) or 1e-6
    prominence_at_center = min(1.0, prom / prom_ref)

    score = (
        0.30 * centrality
        + 0.25 * deriv_score
        + 0.20 * curvature_focus
        + 0.15 * sym_score
        + 0.10 * prominence_at_center
    )
    return float(np.clip(score, 0.0, 1.0))


def neighborhood_rt_consistency(
    candidate_rt_ms: float,
    neighbor_rts_ms: Sequence[float],
    *,
    tolerance_ms: float = 50.0,
    sigma_ms: float = 45.0,
) -> float:
    """
    Candidate-level plausibility vs adjacent-beat RT — **not** beat smoothing.

    Returns [0, 1]:
      1.0 — candidate RT within normal beat-to-beat variation of neighbors (neutral)
      <1.0 — only when candidate is a far outlier vs neighbor median (slight down-weight)

    Does **not** set or average ``T_i``. Preserves genuine RT differences (e.g. 242 vs 248 ms).
    """
    vals = [float(v) for v in neighbor_rts_ms if np.isfinite(v)]
    if not vals:
        return float("nan")

    med = float(np.median(vals))
    dev = abs(float(candidate_rt_ms) - med)
    tol = max(float(tolerance_ms), 1e-6)
    if dev <= tol:
        return 1.0

    excess = dev - tol
    sig = max(float(sigma_ms), 1e-6)
    return float(np.exp(-0.5 * (excess / sig) ** 2))


def candidate_perturbation_stability(
    ecg: np.ndarray,
    candidate: TCandidate,
    candidates: Sequence[TCandidate],
    ctx: TBeatCandidateContext,
    base_score_fn: Callable[[TCandidate], float],
    *,
    perturb_ms: float = 5.0,
    shape_half_width_ms: float = 30.0,
) -> float:
    """
    Fraction of ±perturb_ms shifts where ``candidate`` still outscores all rivals.

    Re-evaluates local apex shape at perturbed positions; stable winners score near 1.
    """
    if not candidates:
        return float("nan")

    rivals = [c for c in candidates if int(c.sample_idx) != int(candidate.sample_idx)]
    rival_max = max((base_score_fn(c) for c in rivals), default=float("-inf"))

    delta = _ms_to_samples(perturb_ms, ctx.fs)
    wins = 0
    for shift in (-delta, 0, delta):
        shape = local_t_apex_shape_score(
            ecg,
            int(candidate.sample_idx) + shift,
            ctx.fs,
            ctx.baseline,
            half_width_ms=shape_half_width_ms,
        )
        trial = base_score_fn(candidate) + shape
        if trial >= rival_max - 1e-9:
            wins += 1
    return float(wins / 3.0)


def compute_landmark_evidence(
    ecg: np.ndarray,
    candidate: TCandidate,
    ctx: TBeatCandidateContext,
    candidates: Sequence[TCandidate],
    *,
    base_score_fn: Callable[[TCandidate], float],
    neighbor_rts_ms: Optional[Sequence[float]] = None,
    perturb_ms: float = 5.0,
    shape_half_width_ms: float = 30.0,
    neighborhood_tolerance_ms: float = 50.0,
    neighborhood_sigma_ms: float = 45.0,
) -> LandmarkEvidence:
    """Compute all three evidence features for one candidate."""
    shape = local_t_apex_shape_score(
        ecg,
        int(candidate.sample_idx),
        ctx.fs,
        ctx.baseline,
        half_width_ms=shape_half_width_ms,
    )

    nbr_rts: List[float] = []
    if neighbor_rts_ms:
        nbr_rts.extend(neighbor_rts_ms)
    if ctx.neighbor_rt_ms is not None and np.isfinite(ctx.neighbor_rt_ms):
        nbr_rts.append(float(ctx.neighbor_rt_ms))
    neighborhood = neighborhood_rt_consistency(
        candidate.rt_ms,
        nbr_rts,
        tolerance_ms=neighborhood_tolerance_ms,
        sigma_ms=neighborhood_sigma_ms,
    )

    stability = candidate_perturbation_stability(
        ecg,
        candidate,
        candidates,
        ctx,
        base_score_fn,
        perturb_ms=perturb_ms,
        shape_half_width_ms=shape_half_width_ms,
    )

    return LandmarkEvidence(
        local_shape_score=shape,
        neighborhood_consistency=neighborhood,
        candidate_stability=stability,
    )


def evidence_to_feature_vector(evidence: LandmarkEvidence) -> List[float]:
    """Flat feature vector for offline LOO learning (logistic / GBT)."""
    return [
        evidence.local_shape_score if np.isfinite(evidence.local_shape_score) else 0.0,
        evidence.neighborhood_consistency
        if np.isfinite(evidence.neighborhood_consistency)
        else 0.5,
        evidence.candidate_stability if np.isfinite(evidence.candidate_stability) else 0.5,
    ]


FEATURE_NAMES = (
    "local_shape_score",
    "neighborhood_consistency",
    "candidate_stability",
)
