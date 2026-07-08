"""
Pairwise T landmark ranking: P(A closer to manual than B | delta evidence).

Learns from candidate pairs (typically positive_peak vs other landmarks),
then aggregates pairwise win probabilities into a beat-level winner.
Does not predict T from raw ECG.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DELTA_FEATURES: Tuple[str, ...] = (
    "delta_timing_prior_score",
    "delta_template_prior_score",
    "delta_morphology_score",
    "delta_derivative_support",
    "delta_prominence_score",
    "delta_beat_corr_support",
    "delta_local_shape_score",
    "delta_neighborhood_consistency",
    "delta_candidate_stability",
    "delta_local_confidence",
)

BASE_FEATURES: Tuple[str, ...] = (
    "timing_prior_score",
    "template_prior_score",
    "morphology_score",
    "derivative_support",
    "prominence_score",
    "beat_corr_support",
    "local_shape_score",
    "neighborhood_consistency",
    "candidate_stability",
    "local_confidence",
)

# Focused training opponents vs positive_peak (audit / regret structure).
PAIRWISE_OPPONENTS: Tuple[str, ...] = (
    "derivative_zero_crossing",
    "negative_peak",
    "template_projected_apex",
    "rising_edge_onset",
    "max_curvature",
)

REFERENCE_TYPE = "positive_peak"


@dataclass
class PairwiseRankerModel:
    coef: np.ndarray
    intercept: float
    feature_medians: np.ndarray
    feature_columns: Tuple[str, ...] = DELTA_FEATURES
    train_stats: Dict[str, float] = field(default_factory=dict)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_imp = _impute(X, self.feature_medians)
        logits = X_imp @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-logits))

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "coef": self.coef.tolist(),
                    "intercept": float(self.intercept),
                    "feature_medians": self.feature_medians.tolist(),
                    "feature_columns": list(self.feature_columns),
                    "train_stats": self.train_stats,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: Path) -> "PairwiseRankerModel":
        data = json.loads(path.read_text())
        return cls(
            coef=np.asarray(data["coef"], dtype=float),
            intercept=float(data["intercept"]),
            feature_medians=np.asarray(data["feature_medians"], dtype=float),
            feature_columns=tuple(data.get("feature_columns", DELTA_FEATURES)),
            train_stats=dict(data.get("train_stats", {})),
        )


def _impute(X: np.ndarray, medians: np.ndarray) -> np.ndarray:
    out = X.copy()
    for j in range(out.shape[1]):
        bad = ~np.isfinite(out[:, j])
        if np.any(bad):
            out[bad, j] = medians[j]
    return out


def _nanmedian_cols(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X, axis=0)
    return np.where(np.isfinite(med), med, 0.0)


def delta_feature_row(row_a: dict, row_b: dict) -> Dict[str, float]:
    """delta_feature = feature_a - feature_b for pairwise evidence."""
    out: Dict[str, float] = {}
    for base, delta in zip(BASE_FEATURES, DELTA_FEATURES):
        a = float(row_a.get(base, np.nan))
        b = float(row_b.get(base, np.nan))
        if np.isfinite(a) and np.isfinite(b):
            out[delta] = a - b
        else:
            out[delta] = float("nan")
    return out


def build_pairwise_rows_from_candidate_table(
    candidate_df: pd.DataFrame,
    *,
    reference_type: str = REFERENCE_TYPE,
    opponents: Sequence[str] = PAIRWISE_OPPONENTS,
) -> pd.DataFrame:
    """
    One row per (beat, positive_peak candidate, opponent candidate).

    a = reference_type (default positive_peak)
    b = opponent type
    a_beats_b = 1 if a closer to manual than b.
    """
    opponents_set = set(opponents)
    rows: List[dict] = []
    for (record_id, beat_id), grp in candidate_df.groupby(["record_id", "beat_id"]):
        refs = grp[grp["candidate_type"] == reference_type]
        opps = grp[grp["candidate_type"].isin(opponents_set)]
        if refs.empty or opps.empty:
            continue
        cycle_idx = int(grp.iloc[0]["cycle_idx"])
        manual_rt = float(grp.iloc[0]["manual_rt_ms"])
        for _, a in refs.iterrows():
            for _, b in opps.iterrows():
                a_err = float(a["abs_error_ms"])
                b_err = float(b["abs_error_ms"])
                if not (np.isfinite(a_err) and np.isfinite(b_err)):
                    continue
                if a_err == b_err:
                    # ties: skip for training signal (no preference)
                    continue
                deltas = delta_feature_row(a.to_dict(), b.to_dict())
                rows.append(
                    {
                        "record_id": record_id,
                        "beat_id": beat_id,
                        "cycle_idx": cycle_idx,
                        "candidate_a_type": reference_type,
                        "candidate_b_type": str(b["candidate_type"]),
                        "candidate_a_sample": float(a["candidate_sample"]),
                        "candidate_b_sample": float(b["candidate_sample"]),
                        "candidate_a_rt_ms": float(a["candidate_rt_ms"]),
                        "candidate_b_rt_ms": float(b["candidate_rt_ms"]),
                        "manual_rt_ms": manual_rt,
                        "a_abs_error_ms": a_err,
                        "b_abs_error_ms": b_err,
                        "a_beats_b": int(a_err < b_err),
                        **deltas,
                    }
                )
    return pd.DataFrame(rows)


def rows_to_delta_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array([[float(r.get(c, np.nan)) for c in DELTA_FEATURES] for _, r in df.iterrows()], dtype=float)
    y = df["a_beats_b"].astype(int).to_numpy()
    return X, y


def train_pairwise_logistic(
    pair_df: pd.DataFrame,
    *,
    C: float = 1.0,
) -> PairwiseRankerModel:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, y = rows_to_delta_matrix(pair_df)
    medians = _nanmedian_cols(X)
    X_imp = _impute(X, medians)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_imp)
    clf = LogisticRegression(
        C=C,
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=0,
    )
    clf.fit(Xs, y)
    w = clf.coef_.ravel()
    coef_raw = w / scaler.scale_
    intercept = float(clf.intercept_[0] - np.sum(w * scaler.mean_ / scaler.scale_))
    return PairwiseRankerModel(
        coef=coef_raw,
        intercept=intercept,
        feature_medians=medians,
        train_stats={
            "n_pairs": float(len(pair_df)),
            "n_positive": float(y.sum()),
            "positive_rate": float(y.mean()) if len(y) else float("nan"),
        },
    )


def predict_pair_prob(
    model: PairwiseRankerModel,
    row_a: dict,
    row_b: dict,
) -> float:
    deltas = delta_feature_row(row_a, row_b)
    X = np.array([[float(deltas[c]) for c in DELTA_FEATURES]], dtype=float)
    return float(model.predict_proba(X)[0])


def score_candidates_pairwise(
    model: PairwiseRankerModel,
    candidates: Sequence[dict],
) -> Dict[int, float]:
    """
    Aggregate mean pairwise win probability for each candidate index.

    candidate_score[i] = mean_j≠i P(i beats j | delta evidence).
    """
    n = len(candidates)
    if n == 0:
        return {}
    if n == 1:
        return {0: 1.0}

    # Vectorized deltas for all ordered pairs (i, j), i != j
    feat_mat = np.array(
        [[float(c.get(f, np.nan)) for f in BASE_FEATURES] for c in candidates],
        dtype=float,
    )
    rows = []
    pair_idx = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            delta = feat_mat[i] - feat_mat[j]
            rows.append(delta)
            pair_idx.append(i)
    X = np.asarray(rows, dtype=float)
    probs = model.predict_proba(X)
    buckets: Dict[int, List[float]] = {i: [] for i in range(n)}
    for i, p in zip(pair_idx, probs):
        buckets[i].append(float(p))
    return {i: float(np.mean(v)) if v else 0.0 for i, v in buckets.items()}


def pick_beat_winner_pairwise(
    model: PairwiseRankerModel,
    beat_candidates: pd.DataFrame,
) -> Optional[pd.Series]:
    """Choose candidate with highest mean pairwise win probability."""
    if beat_candidates.empty:
        return None
    cands = [r.to_dict() for _, r in beat_candidates.iterrows()]
    scores = score_candidates_pairwise(model, cands)
    best_i = max(scores, key=scores.get)
    out = beat_candidates.iloc[best_i].copy()
    out["pairwise_score"] = scores[best_i]
    return out


def predict_beats_pairwise(
    model: PairwiseRankerModel,
    candidate_df: pd.DataFrame,
) -> pd.DataFrame:
    """Beat-level picks via pairwise aggregation (always commit)."""
    picks: List[dict] = []
    for (record_id, beat_id), grp in candidate_df.groupby(["record_id", "beat_id"]):
        winner = pick_beat_winner_pairwise(model, grp)
        if winner is None:
            continue
        picks.append(
            {
                "record_id": record_id,
                "beat_id": beat_id,
                "cycle_idx": int(winner["cycle_idx"]),
                "r_sample": float(winner["r_sample"]),
                "manual_rt_ms": float(winner["manual_rt_ms"]),
                "manual_t_sample": float(winner.get("manual_t_sample", np.nan)),
                "ranked_rt_ms": float(winner["candidate_rt_ms"]),
                "ranked_candidate_type": str(winner["candidate_type"]),
                "ranked_candidate_sample": float(winner["candidate_sample"]),
                "abs_error_ms": float(winner["abs_error_ms"]),
                "pairwise_score": float(winner["pairwise_score"]),
                "withheld": False,
                "exported_rt_ms": float(winner.get("exported_rt_ms", np.nan))
                if "exported_rt_ms" in winner
                else float("nan"),
                "exported_abs_error_ms": float(winner.get("exported_abs_error_ms", np.nan))
                if "exported_abs_error_ms" in winner
                else float("nan"),
            }
        )
    return pd.DataFrame(picks)
