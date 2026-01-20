"""
Train baseline arrhythmia-burden predictors on the MGH dataset using PyHEARTS outputs.

This script:
  1) Loads per-record labels from mgh_metadata_summary.csv (derived from .ari annotations).
  2) Aggregates per-cycle PyHEARTS features (mgh###_pyhearts.csv) to per-record stats.
  3) Joins HRV + variability metric CSVs if present.
  4) Trains simple baseline classifiers with cross-validation and writes metrics/artifacts.

Recommended labels (binary, from .ari):
  - ventricular_ectopy_present: sym_v > 0
  - supraventricular_ectopy_present: sym_s > 0
  - pacing_present: sym_p > 0
  - junctional_present: sym_j > 0
  - noisy_record: sym_tilde > 0 OR aux_artifact > 0 OR aux_electrocautery > 0
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Column-wise clipping to reduce the impact of extreme values.
    Fits per-column lower/upper quantiles on the training fold only.
    """

    def __init__(self, q_low: float = 0.01, q_high: float = 0.99):
        self.q_low = q_low
        self.q_high = q_high
        self.lo_: np.ndarray | None = None
        self.hi_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: Any = None) -> "Winsorizer":
        X = np.asarray(X, dtype=float)
        # Replace infs with nan for quantile calc
        X = np.where(np.isfinite(X), X, np.nan)
        self.lo_ = np.nanquantile(X, self.q_low, axis=0)
        self.hi_ = np.nanquantile(X, self.q_high, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.lo_ is None or self.hi_ is None:
            raise RuntimeError("Winsorizer not fit() yet.")
        X = np.asarray(X, dtype=float)
        X = np.where(np.isfinite(X), X, np.nan)
        return np.clip(X, self.lo_, self.hi_)


@dataclass(frozen=True)
class LabelSpec:
    name: str
    # columns in metadata CSV used to construct label
    cols: tuple[str, ...]


DEFAULT_LABELS: dict[str, LabelSpec] = {
    "ventricular_ectopy_present": LabelSpec("ventricular_ectopy_present", ("sym_v",)),
    "supraventricular_ectopy_present": LabelSpec("supraventricular_ectopy_present", ("sym_s",)),
    "pacing_present": LabelSpec("pacing_present", ("sym_p",)),
    "junctional_present": LabelSpec("junctional_present", ("sym_j",)),
    "noisy_record": LabelSpec("noisy_record", ("sym_tilde", "aux_artifact", "aux_electrocautery")),
}


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def iqr(x: pd.Series) -> float:
    q = x.quantile([0.25, 0.75])
    return float(q.iloc[1] - q.iloc[0])


def aggregate_pyhearts_cycle_features(pyhearts_csv: Path) -> dict[str, Any]:
    df = pd.read_csv(pyhearts_csv)

    # Keep numeric columns only; drop obvious identifiers
    drop_cols = {"cycle_index"}
    num_df = df.drop(columns=[c for c in df.columns if c in drop_cols], errors="ignore")
    num_df = num_df.select_dtypes(include=[np.number])

    out: dict[str, Any] = {}
    out["n_cycles"] = int(len(df))

    if num_df.shape[1] == 0 or num_df.shape[0] == 0:
        return out

    for col in num_df.columns:
        s = num_df[col]
        # Robust summaries are more stable across noisy clinical records.
        # Guard against all-NaN columns to avoid noisy warnings.
        vals = s.values.astype(float, copy=False)
        finite = np.isfinite(vals)
        if not finite.any():
            out[f"{col}__median"] = np.nan
            out[f"{col}__iqr"] = np.nan
            out[f"{col}__std"] = np.nan
            out[f"{col}__mean"] = np.nan
        else:
            out[f"{col}__median"] = float(np.nanmedian(vals))
            out[f"{col}__iqr"] = iqr(s.dropna()) if s.notna().any() else np.nan
            out[f"{col}__std"] = float(np.nanstd(vals))
            out[f"{col}__mean"] = float(np.nanmean(vals))
        out[f"{col}__missing_frac"] = float(s.isna().mean())

    return out


def build_feature_table(run_dir: Path) -> pd.DataFrame:
    """
    Build one row per record from PyHEARTS outputs inside a run folder.
    """
    pyhearts_files = sorted(run_dir.glob("mgh*_pyhearts.csv"))
    if not pyhearts_files:
        raise SystemExit(f"No mgh*_pyhearts.csv found in {run_dir}")

    rows: list[dict[str, Any]] = []
    for p in pyhearts_files:
        rec = p.name.replace("_pyhearts.csv", "")
        row: dict[str, Any] = {"record": rec}
        row.update(aggregate_pyhearts_cycle_features(p))

        # Join per-record HRV and variability summary metrics if present
        hrv = safe_read_csv(run_dir / f"{rec}_hrv_metrics.csv")
        if hrv is not None and len(hrv) == 1:
            for k, v in hrv.iloc[0].to_dict().items():
                row[f"hrv__{k}"] = v

        var = safe_read_csv(run_dir / f"{rec}_variability_metrics.csv")
        if var is not None and len(var) == 1:
            for k, v in var.iloc[0].to_dict().items():
                row[f"var__{k}"] = v

        rows.append(row)

    feat = pd.DataFrame(rows)
    return feat


def build_labels(meta_csv: Path, label_names: list[str]) -> pd.DataFrame:
    meta = pd.read_csv(meta_csv)
    if "record" not in meta.columns:
        raise SystemExit(f"metadata CSV missing 'record' column: {meta_csv}")

    out = meta[["record"]].copy()

    for ln in label_names:
        if ln not in DEFAULT_LABELS:
            raise SystemExit(f"Unknown label '{ln}'. Available: {sorted(DEFAULT_LABELS.keys())}")
        spec = DEFAULT_LABELS[ln]
        for c in spec.cols:
            if c not in meta.columns:
                # missing symbol columns should be treated as all-zero
                out[c] = 0
        # construct label
        if ln == "noisy_record":
            # any evidence of noise
            cols = [c for c in spec.cols if c in meta.columns]
            if cols:
                out[ln] = (meta[cols].fillna(0).sum(axis=1) > 0).astype(int)
            else:
                out[ln] = 0
        else:
            c = spec.cols[0]
            out[ln] = (meta[c].fillna(0) > 0).astype(int) if c in meta.columns else 0

        # also store burden rate if possible (count / n_annotations)
        if spec.cols and spec.cols[0].startswith("sym_") and "n_annotations" in meta.columns:
            c0 = spec.cols[0]
            denom = meta["n_annotations"].replace({0: np.nan})
            if c0 in meta.columns:
                out[f"{ln}__rate"] = (meta[c0].fillna(0) / denom).astype(float)
            else:
                out[f"{ln}__rate"] = np.nan

    # Keep a couple of demographic covariates available for optional adjustment
    for c in ["age_raw", "sex_raw"]:
        if c in meta.columns:
            out[c] = meta[c]

    return out


def sex_to_numeric(s: pd.Series) -> pd.Series:
    # Encode M/F; unknown -> NaN
    up = s.astype(str).str.strip().str.upper()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[up == "M"] = 1.0
    out[up == "F"] = 0.0
    return out


def age_to_numeric(s: pd.Series) -> pd.Series:
    # Extract first integer from age_raw
    out = pd.Series(np.nan, index=s.index, dtype=float)
    for i, v in s.items():
        m = re.search(r"\\d+", str(v))
        if m:
            out.loc[i] = float(int(m.group(0)))
    return out


def _class_weights(y: np.ndarray) -> np.ndarray:
    # Balanced sample weights (like class_weight='balanced')
    n = len(y)
    pos = max(1, int(y.sum()))
    neg = max(1, int(n - y.sum()))
    w_pos = n / (2.0 * pos)
    w_neg = n / (2.0 * neg)
    return np.where(y == 1, w_pos, w_neg).astype(float)


def _make_estimator(*, model: str, seed: int) -> Pipeline:
    if model == "logreg":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("winsor", Winsorizer(q_low=0.01, q_high=0.99)),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if model == "hgb":
        # Tree-based baseline: no scaling, robust to feature magnitudes.
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("winsor", Winsorizer(q_low=0.01, q_high=0.99)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        random_state=seed,
                        max_depth=3,
                        learning_rate=0.1,
                        max_iter=300,
                    ),
                ),
            ]
        )
    raise SystemExit("Unknown model. Use --model logreg or --model hgb")


def train_cv(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    folds: int,
    model: str,
    return_coef: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    # columns: numeric only; we will impute+scale
    Xn = X.select_dtypes(include=[np.number]).copy()
    # Replace infinities and drop completely empty columns
    Xn.replace([np.inf, -np.inf], np.nan, inplace=True)
    Xn = Xn.dropna(axis=1, how="all")

    est = _make_estimator(model=model, seed=seed)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    aucs: list[float] = []
    aprs: list[float] = []
    coef_rows: list[dict[str, Any]] = []

    yv = y.values.astype(int)
    for tr, te in skf.split(Xn, yv):
        sw = _class_weights(yv[tr])
        # Some estimators don't accept sample_weight; pipe it only when supported.
        try:
            est.fit(Xn.iloc[tr], yv[tr], clf__sample_weight=sw)
        except TypeError:
            est.fit(Xn.iloc[tr], yv[tr])
        p = est.predict_proba(Xn.iloc[te])[:, 1]
        aucs.append(float(roc_auc_score(yv[te], p)))
        aprs.append(float(average_precision_score(yv[te], p)))

        if return_coef and model == "logreg":
            clf = est.named_steps["clf"]
            coefs = clf.coef_.ravel()
            for name, val in zip(Xn.columns.tolist(), coefs):
                coef_rows.append({"feature": name, "coef": float(val)})

    metrics = {
        "n": int(len(yv)),
        "pos": int(yv.sum()),
        "neg": int(len(yv) - yv.sum()),
        "roc_auc_mean": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
        "auprc_mean": float(np.mean(aprs)),
        "auprc_std": float(np.std(aprs)),
    }
    coef_df = None
    if return_coef and coef_rows:
        coef_df = pd.DataFrame(coef_rows).groupby("feature", as_index=False).agg(
            coef_mean=("coef", "mean"),
            coef_std=("coef", "std"),
            coef_abs_mean=("coef", lambda x: float(np.mean(np.abs(x)))),
        )
        coef_df = coef_df.sort_values("coef_abs_mean", ascending=False)

    return metrics, coef_df


def _topk_mask(values: np.ndarray, frac: float) -> tuple[np.ndarray, float, int]:
    idx = np.where(np.isfinite(values))[0]
    vals = values[idx]
    if len(vals) == 0:
        return np.zeros(len(values), dtype=bool), float("nan"), 0
    k = int(np.ceil(frac * len(vals)))
    k = max(1, min(k, len(vals)))
    order = np.argsort(vals)  # ascending
    top_idx_local = order[-k:]
    thr = float(np.min(vals[top_idx_local]))
    mask = np.zeros(len(values), dtype=bool)
    mask[idx[top_idx_local]] = True
    return mask, thr, k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        type=Path,
        default=Path("results/mgh_all_20260118_173512"),
        help="Results run directory containing mgh*_pyhearts.csv and per-record metrics",
    )
    ap.add_argument(
        "--metadata_csv",
        type=Path,
        default=Path("results/mgh_all_20260118_173512/mgh_metadata_summary.csv"),
        help="Metadata/annotation summary CSV generated from .hea/.ari",
    )
    ap.add_argument(
        "--labels",
        type=str,
        default="ventricular_ectopy_present,supraventricular_ectopy_present,pacing_present",
        help=f"Comma-separated labels from: {','.join(sorted(DEFAULT_LABELS.keys()))}",
    )
    ap.add_argument(
        "--high_burden_target",
        type=str,
        default="",
        help="If set, creates a new binary label for high burden (top-k) based on a *_rate column. Options: pvc, sv, pacing",
    )
    ap.add_argument(
        "--high_burden_top_frac",
        type=float,
        default=0.2,
        help="Top fraction for high burden label (e.g., 0.2 for top 20%%). Uses rank-based top-k to avoid tie ambiguity.",
    )
    ap.add_argument("--include_demographics", action="store_true", help="Include age/sex as covariates")
    ap.add_argument("--model", type=str, default="hgb", help="Model: hgb (default) or logreg")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    meta_csv: Path = args.metadata_csv
    label_names = [s.strip() for s in args.labels.split(",") if s.strip()]

    feat = build_feature_table(run_dir)
    lab = build_labels(meta_csv, label_names)

    # Optional: create a high-burden label from the per-record rate columns
    hb_target = args.high_burden_target.strip().lower()
    hb_label_name = ""
    hb_threshold = None
    if hb_target:
        target_to_rate = {
            "pvc": "ventricular_ectopy_present__rate",
            "sv": "supraventricular_ectopy_present__rate",
            "pacing": "pacing_present__rate",
        }
        if hb_target not in target_to_rate:
            raise SystemExit("high_burden_target must be one of: pvc, sv, pacing")
        rate_col = target_to_rate[hb_target]
        if rate_col not in lab.columns:
            raise SystemExit(f"Missing required rate column in labels table: {rate_col}")
        vals = lab[rate_col].to_numpy(dtype=float)
        mask, thr, k = _topk_mask(vals, float(args.high_burden_top_frac))
        hb_label_name = f"high_{hb_target}_burden_top{int(round(100*args.high_burden_top_frac))}"
        lab[hb_label_name] = mask.astype(int)
        hb_threshold = thr
        label_names = [hb_label_name]

    # Optional demographic covariates
    if args.include_demographics:
        if "age_raw" in lab.columns:
            lab["age_years"] = age_to_numeric(lab["age_raw"])
        if "sex_raw" in lab.columns:
            lab["sex_male"] = sex_to_numeric(lab["sex_raw"])

    df = feat.merge(lab, on="record", how="inner")
    if df.empty:
        raise SystemExit("No overlap between run features and metadata records.")

    # IMPORTANT: prevent label leakage.
    # Only use features originating from PyHEARTS outputs (feat), plus optional demographic covariates.
    feat_cols = [c for c in feat.columns if c != "record"]

    # Save modeling table
    out_tag = hb_label_name if hb_label_name else "mgh_arrhythmia"
    out_table = run_dir / f"{out_tag}_modeling_table.csv"
    df.to_csv(out_table, index=False)

    metrics: dict[str, Any] = {"n_records": int(len(df)), "labels": {}, "labels_rate_cols": {}}
    if hb_label_name:
        metrics["high_burden"] = {
            "label": hb_label_name,
            "top_frac": float(args.high_burden_top_frac),
            "threshold_rate_min_in_topk": float(hb_threshold) if hb_threshold is not None else None,
        }

    # Train each label separately
    for ln in label_names:
        y = df[ln].astype(int)
        if y.nunique() < 2:
            metrics["labels"][ln] = {"error": "Label has <2 classes after join; cannot train."}
            continue

        feature_cols = feat_cols.copy()
        if args.include_demographics:
            for c in ["age_years", "sex_male"]:
                if c in df.columns and c not in feature_cols:
                    feature_cols.append(c)

        X = df[feature_cols]
        m, coef_df = train_cv(
            X,
            y,
            seed=args.seed,
            folds=args.folds,
            model=args.model,
            return_coef=(args.model == "logreg"),
        )
        metrics["labels"][ln] = m
        if coef_df is not None:
            coef_path = run_dir / f"{out_tag}__{ln}__logreg_feature_coeffs.csv"
            coef_df.to_csv(coef_path, index=False)

        rate_col = f"{ln}__rate"
        if rate_col in df.columns:
            metrics["labels_rate_cols"][ln] = {
                "rate_col": rate_col,
                "rate_median": float(np.nanmedian(df[rate_col].values)),
                "rate_iqr": float(np.nanpercentile(df[rate_col].values, 75) - np.nanpercentile(df[rate_col].values, 25)),
            }

    out_metrics = run_dir / f"{out_tag}_model_metrics.json"
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Wrote modeling table: {out_table}")
    print(f"Wrote metrics: {out_metrics}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()


