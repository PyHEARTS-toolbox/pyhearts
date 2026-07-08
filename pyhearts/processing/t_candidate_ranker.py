"""
L2-regularized logistic candidate ranker: P(candidate correct | evidence).

Ranks pre-generated T landmark candidates; does not predict T from raw ECG.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.t_candidate_scoring import TCandidate
from pyhearts.processing.t_candidate_ranking_features import (
    FEATURE_COLUMNS,
    build_candidate_feature_row,
    impute_median,
    rows_to_matrix,
)
from pyhearts.processing.t_landmark_ensemble import TLandmarkEnsembleScore, _primary_source


@dataclass
class CandidateRankerModel:
    """Trained L2 logistic ranker + commit threshold."""

    coef: np.ndarray
    intercept: float
    feature_medians: np.ndarray
    commit_threshold: float = 0.5
    correct_tol_ms: float = 20.0
    feature_columns: Tuple[str, ...] = FEATURE_COLUMNS
    train_stats: Dict[str, float] = field(default_factory=dict)

    def predict_proba_rows(self, rows: Sequence[Dict[str, float]]) -> np.ndarray:
        X, _ = rows_to_matrix(rows)
        X, _ = impute_median(X, self.feature_medians)
        logits = X @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-logits))

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        X, _ = impute_median(X, self.feature_medians)
        logits = X @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-logits))

    def save(self, path: Path) -> None:
        payload = {
            "coef": self.coef.tolist(),
            "intercept": float(self.intercept),
            "feature_medians": self.feature_medians.tolist(),
            "commit_threshold": float(self.commit_threshold),
            "correct_tol_ms": float(self.correct_tol_ms),
            "feature_columns": list(self.feature_columns),
            "train_stats": self.train_stats,
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "CandidateRankerModel":
        data = json.loads(path.read_text())
        return cls(
            coef=np.asarray(data["coef"], dtype=float),
            intercept=float(data["intercept"]),
            feature_medians=np.asarray(data["feature_medians"], dtype=float),
            commit_threshold=float(data.get("commit_threshold", 0.5)),
            correct_tol_ms=float(data.get("correct_tol_ms", 20.0)),
            feature_columns=tuple(data.get("feature_columns", FEATURE_COLUMNS)),
            train_stats=dict(data.get("train_stats", {})),
        )


def train_l2_logistic_ranker(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    class_weight: Optional[str] = "balanced",
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Fit L2-regularized logistic regression; returns (coef, intercept)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_imp, medians = impute_median(X)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_imp)
    clf = LogisticRegression(
        C=C,
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        class_weight=class_weight,
        random_state=0,
    )
    clf.fit(Xs, y)
    # Map coefficients back to raw feature space: logit = (Xs @ w) + b, Xs = (X - mean)/scale
    w = clf.coef_.ravel()
    coef_raw = w / scaler.scale_
    intercept = float(clf.intercept_[0] - np.sum(w * scaler.mean_ / scaler.scale_))
    return coef_raw, intercept, medians


def train_ranker_from_table(
    df,
    *,
    C: float = 1.0,
    commit_threshold: float = 0.5,
    correct_tol_ms: float = 20.0,
) -> CandidateRankerModel:
    """Train from candidate-level DataFrame with FEATURE_COLUMNS + is_correct."""
    rows = [dict(r) for r in df.to_dict(orient="records")]
    X, _ = rows_to_matrix(rows)
    y = df["is_correct"].astype(int).to_numpy()
    coef, intercept, medians = train_l2_logistic_ranker(X, y, C=C)
    return CandidateRankerModel(
        coef=coef,
        intercept=intercept,
        feature_medians=medians,
        commit_threshold=commit_threshold,
        correct_tol_ms=correct_tol_ms,
        train_stats={
            "n_rows": float(len(df)),
            "n_positive": float(y.sum()),
            "positive_rate": float(y.mean()) if len(y) else float("nan"),
        },
    )


def tune_commit_threshold(
    model: CandidateRankerModel,
    df,
    *,
    thresholds: Optional[Sequence[float]] = None,
    min_commit_rate: float = 0.97,
) -> float:
    """
    Tune withhold threshold on training beats (beat-level max prob vs manual).

    Minimizes RT MAE on committed beats while keeping commit rate >= min_commit_rate
    so threshold tuning does not sacrifice T coverage.
    """
    if thresholds is None:
        thresholds = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]

    best_t = 0.05
    best_mae = float("inf")
    fallback_t = 0.05
    fallback_mae = float("inf")
    for t in thresholds:
        mae, commit_rate = _beat_level_mae(model, df, t)
        if not np.isfinite(mae):
            continue
        if commit_rate >= min_commit_rate and mae < best_mae:
            best_mae = mae
            best_t = float(t)
        if mae < fallback_mae:
            fallback_mae = mae
            fallback_t = float(t)
    return best_t if np.isfinite(best_mae) else fallback_t


def _beat_level_mae(
    model: CandidateRankerModel, df, threshold: float
) -> Tuple[float, float]:
    errs = []
    n_beats = 0
    n_committed = 0
    for (_, _), grp in df.groupby(["record_id", "beat_id"]):
        n_beats += 1
        rows = [dict(r) for r in grp.to_dict(orient="records")]
        probs = model.predict_proba_rows(rows)
        j = int(np.argmax(probs))
        if probs[j] < threshold:
            continue
        n_committed += 1
        err = float(grp.iloc[j]["abs_error_ms"])
        if np.isfinite(err):
            errs.append(err)
    commit_rate = float(n_committed / n_beats) if n_beats else 0.0
    mae = float(np.mean(errs)) if errs else float("inf")
    return mae, commit_rate


def predict_beats_from_candidate_table(
    model: CandidateRankerModel,
    df,
    *,
    threshold: Optional[float] = None,
) -> "pd.DataFrame":
    """Beat-level picks from candidate rows (one row per candidate)."""
    import pandas as pd

    thresh = float(threshold if threshold is not None else model.commit_threshold)
    picks = []
    for (_, beat_id), grp in df.groupby(["record_id", "beat_id"]):
        rows = [dict(r) for r in grp.to_dict(orient="records")]
        probs = model.predict_proba_rows(rows)
        j = int(np.argmax(probs))
        base = grp.iloc[0]
        if probs[j] < thresh:
            picks.append(
                {
                    "record_id": base["record_id"],
                    "beat_id": beat_id,
                    "cycle_idx": base["cycle_idx"],
                    "r_sample": base["r_sample"],
                    "manual_rt_ms": base["manual_rt_ms"],
                    "manual_t_sample": base["manual_t_sample"],
                    "ranked_rt_ms": float("nan"),
                    "ranked_candidate_type": None,
                    "max_probability": float(probs[j]),
                    "withheld": True,
                    "abs_error_ms": float("nan"),
                }
            )
            continue
        row = grp.iloc[j]
        picks.append(
            {
                "record_id": row["record_id"],
                "beat_id": beat_id,
                "cycle_idx": row["cycle_idx"],
                "r_sample": row["r_sample"],
                "manual_rt_ms": row["manual_rt_ms"],
                "manual_t_sample": row["manual_t_sample"],
                "ranked_rt_ms": row["candidate_rt_ms"],
                "ranked_candidate_type": row["candidate_type"],
                "max_probability": float(probs[j]),
                "withheld": False,
                "abs_error_ms": float(row["abs_error_ms"]),
            }
        )
    return pd.DataFrame(picks)


def rank_candidates_on_beat(
    *,
    scored: Sequence[TLandmarkEnsembleScore],
    candidates: Sequence[TCandidate],
    morphology_class: str,
    model: CandidateRankerModel,
) -> Tuple[Optional[TCandidate], float, List[float]]:
    """Return (best candidate, max probability, all probs aligned with scored)."""
    if not scored or not candidates:
        return None, 0.0, []
    cand_by_idx = {int(c.sample_idx): c for c in candidates}
    n = len(scored)
    rows = []
    for sc in scored:
        c = cand_by_idx.get(int(sc.sample_idx))
        if c is None:
            rows.append({col: 0.0 for col in FEATURE_COLUMNS})
            continue
        rows.append(
            build_candidate_feature_row(
                score=sc,
                candidate_type=c.source,
                morphology_class=morphology_class,
                candidate_count=n,
                scored_on_beat=scored,
            )
        )
    probs = model.predict_proba_rows(rows)
    j = int(np.argmax(probs))
    best = cand_by_idx.get(int(scored[j].sample_idx))
    return best, float(probs[j]), list(probs)


def apply_learned_ranker_pass(
    output_dict: dict,
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    cycles: List[int],
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    template_prior_by_cycle: dict,
    record_template,
    *,
    ranker_model: Optional[CandidateRankerModel] = None,
) -> dict:
    """
    Per-beat candidate ranking pass: score all candidates, commit if P >= threshold.
    """
    from pyhearts.processing.delineation_signal import prepare_record_delineation_signal
    from pyhearts.processing.t_candidate_scoring import generate_t_candidates
    from pyhearts.processing.t_landmark_ensemble import (
        filter_ensemble_candidates,
        score_landmark_ensemble,
        _neighbor_rts_ms,
    )
    from pyhearts.processing.t_morphology_routing import morphology_rescue_landmark_global
    from pyhearts.processing.template_prior_window_diagnostics import _beat_template_correlation

    stats = {
        "attempted": 0,
        "ranked": 0,
        "withheld": 0,
        "no_candidates": 0,
    }
    if ranker_model is None:
        path = getattr(cfg, "record_template_prior_ranker_model_path", None)
        if path:
            ranker_model = CandidateRankerModel.load(Path(path))
    if ranker_model is None:
        return stats

    ecg_delim = prepare_record_delineation_signal(ecg, sampling_rate, cfg)
    t_list = output_dict.get("T_global_center_idx", [])
    t_source = output_dict.get("t_source", None)

    for cycle_idx, cycle_label in enumerate(cycles):
        prior = template_prior_by_cycle.get(cycle_idx)
        if prior is None or cycle_idx >= len(t_list):
            continue
        epoch_i = int(cycle_label)
        if epoch_i < 0 or epoch_i >= len(r_peaks):
            continue
        s_i, q_next = prior.s_i, prior.q_next
        if s_i is None or q_next is None:
            continue

        stats["attempted"] += 1
        r_idx = int(r_peaks[epoch_i])
        prev_t = t_list[cycle_idx - 1] if cycle_idx > 0 else None
        next_t = t_list[cycle_idx + 1] if cycle_idx + 1 < len(t_list) else None
        neighbor_t = (
            int(prev_t) if prev_t is not None and np.isfinite(prev_t) else None,
            int(next_t) if next_t is not None and np.isfinite(next_t) else None,
        )

        morph = (
            record_template.t_morphology
            if record_template is not None and record_template.valid
            else "normal"
        )

        candidates, ctx = generate_t_candidates(
            ecg_delim,
            r_idx,
            int(s_i),
            int(q_next),
            sampling_rate,
            tmpl=record_template,
            cfg=cfg,
            neighbor_t_samples=neighbor_t,
        )
        ensemble = filter_ensemble_candidates(candidates)
        if not ensemble:
            stats["no_candidates"] += 1
            continue

        beat_corr = (
            _beat_template_correlation(ecg_delim, int(s_i), int(q_next), record_template)
            if record_template is not None and record_template.valid
            else float("nan")
        )
        template_landmark_rt_ms = None
        if record_template is not None and record_template.valid:
            land_s, _ = morphology_rescue_landmark_global(
                int(s_i), int(q_next), record_template, sampling_rate, cfg
            )
            if land_s is not None:
                template_landmark_rt_ms = (float(land_s) - r_idx) * 1000.0 / sampling_rate

        scored = score_landmark_ensemble(
            ensemble,
            ctx,
            ecg=ecg_delim,
            template_landmark_rt_ms=template_landmark_rt_ms,
            beat_template_corr=beat_corr,
            cfg=cfg,
            neighbor_rts_ms=_neighbor_rts_ms(r_idx, sampling_rate, neighbor_t),
        )
        picked, _, prob = pick_ranked_t_candidate(
            scored, ensemble, morph, cfg, model=ranker_model
        )
        if picked is not None and prob >= ranker_model.commit_threshold:
            t_list[cycle_idx] = float(picked.sample_idx)
            if isinstance(t_source, list) and cycle_idx < len(t_source):
                t_source[cycle_idx] = f"template_prior_ranker:{_primary_source(picked.source)}"
            stats["ranked"] += 1
        elif getattr(cfg, "record_template_prior_ranker_withhold_low_confidence", True):
            t_list[cycle_idx] = np.nan
            if isinstance(t_source, list) and cycle_idx < len(t_source):
                t_source[cycle_idx] = None
            stats["withheld"] += 1

    output_dict["T_global_center_idx"] = t_list
    return stats


def pick_ranked_t_candidate(
    scored: Sequence[TLandmarkEnsembleScore],
    candidates: Sequence[TCandidate],
    morphology_class: str,
    cfg: ProcessCycleConfig,
    model: Optional[CandidateRankerModel] = None,
) -> Tuple[Optional[TCandidate], List[TLandmarkEnsembleScore], float]:
    """
    Rank candidates by P(correct); apply commit threshold from cfg or model.

    Returns (candidate or None if withheld, scored list, max probability).
    """
    if model is None:
        path = getattr(cfg, "record_template_prior_ranker_model_path", None)
        if path:
            model = CandidateRankerModel.load(Path(path))
    if model is None:
        raise ValueError("CandidateRankerModel required when learned ranker is enabled")

    thresh = float(
        getattr(cfg, "record_template_prior_ranker_commit_threshold", model.commit_threshold)
        or model.commit_threshold
    )
    best, prob, _ = rank_candidates_on_beat(
        scored=scored,
        candidates=candidates,
        morphology_class=morphology_class,
        model=model,
    )
    if best is None or prob < thresh:
        return None, list(scored), prob
    return best, list(scored), prob


def pick_ranked_t_candidate_from_ensemble(
    ecg: np.ndarray,
    r_idx: int,
    s_i: int,
    q_next: int,
    fs: float,
    tmpl,
    cfg: ProcessCycleConfig,
    *,
    neighbor_t_samples=None,
    model: Optional[CandidateRankerModel] = None,
) -> Tuple[Optional[TCandidate], List[TLandmarkEnsembleScore]]:
    """Generate candidates, score with ensemble evidence, rank with learned model."""
    from pyhearts.processing.t_candidate_scoring import generate_t_candidates
    from pyhearts.processing.t_landmark_ensemble import (
        _neighbor_rts_ms,
        filter_ensemble_candidates,
        score_landmark_ensemble,
    )
    from pyhearts.processing.t_morphology_routing import (
        morphology_rescue_landmark_global,
        normalize_t_morphology_tag,
    )
    from pyhearts.processing.template_prior_window_diagnostics import _beat_template_correlation

    candidates, ctx = generate_t_candidates(
        ecg, int(r_idx), int(s_i), int(q_next), fs, tmpl=tmpl, cfg=cfg,
        neighbor_t_samples=neighbor_t_samples,
    )
    ensemble = filter_ensemble_candidates(candidates)
    if not ensemble:
        return None, []

    morph = normalize_t_morphology_tag(
        tmpl.t_morphology if tmpl is not None and tmpl.valid else "normal"
    )
    beat_corr = (
        _beat_template_correlation(ecg, int(s_i), int(q_next), tmpl)
        if tmpl is not None and tmpl.valid
        else float("nan")
    )
    template_landmark_rt_ms = None
    if tmpl is not None and tmpl.valid and tmpl.t_landmark_idx is not None:
        land_s, _ = morphology_rescue_landmark_global(int(s_i), int(q_next), tmpl, fs, cfg)
        if land_s is not None:
            template_landmark_rt_ms = (float(land_s) - float(r_idx)) * 1000.0 / fs

    scored = score_landmark_ensemble(
        ensemble,
        ctx,
        ecg=ecg,
        template_landmark_rt_ms=template_landmark_rt_ms,
        beat_template_corr=beat_corr,
        cfg=cfg,
        neighbor_rts_ms=_neighbor_rts_ms(int(r_idx), fs, neighbor_t_samples),
    )
    best, scored_list, _ = pick_ranked_t_candidate(
        scored, ensemble, morph, cfg, model=model
    )
    return best, scored_list
