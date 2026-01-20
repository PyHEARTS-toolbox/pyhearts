"""
Train baseline predictors of cardiovascular disease (CVD) phenotypes on MGH using PyHEARTS features.

MGHDB provides free-text diagnoses/procedures in WFDB headers. This script converts those
into coarse phenotype labels (regex rules), joins with aggregated PyHEARTS features from a run
folder, and trains an interpretable baseline (logistic regression) with cross-validation.

Outputs:
  - modeling table CSV (features + labels)
  - metrics JSON
  - coefficient ranking CSV (mean coef across folds)
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def iqr(x: pd.Series) -> float:
    q = x.quantile([0.25, 0.75])
    return float(q.iloc[1] - q.iloc[0])


def aggregate_pyhearts_cycle_features(pyhearts_csv: Path) -> dict[str, Any]:
    df = pd.read_csv(pyhearts_csv)
    drop_cols = {"cycle_index"}
    num_df = df.drop(columns=[c for c in df.columns if c in drop_cols], errors="ignore")
    num_df = num_df.select_dtypes(include=[np.number])

    out: dict[str, Any] = {"n_cycles": int(len(df))}
    if num_df.shape[1] == 0 or num_df.shape[0] == 0:
        return out

    for col in num_df.columns:
        s = num_df[col]
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
    pyhearts_files = sorted(run_dir.glob("mgh*_pyhearts.csv"))
    if not pyhearts_files:
        raise SystemExit(f"No mgh*_pyhearts.csv found in {run_dir}")

    rows: list[dict[str, Any]] = []
    for p in pyhearts_files:
        rec = p.name.replace("_pyhearts.csv", "")
        row: dict[str, Any] = {"record": rec}
        row.update(aggregate_pyhearts_cycle_features(p))

        hrv = safe_read_csv(run_dir / f"{rec}_hrv_metrics.csv")
        if hrv is not None and len(hrv) == 1:
            for k, v in hrv.iloc[0].to_dict().items():
                row[f"hrv__{k}"] = v

        var = safe_read_csv(run_dir / f"{rec}_variability_metrics.csv")
        if var is not None and len(var) == 1:
            for k, v in var.iloc[0].to_dict().items():
                row[f"var__{k}"] = v

        rows.append(row)

    return pd.DataFrame(rows)


def age_to_numeric(s: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=s.index, dtype=float)
    for i, v in s.items():
        m = re.search(r"\\d+", str(v))
        if m:
            out.loc[i] = float(int(m.group(0)))
    return out


def sex_to_numeric(s: pd.Series) -> pd.Series:
    up = s.astype(str).str.strip().str.upper()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[up == "M"] = 1.0
    out[up == "F"] = 0.0
    return out


def _class_weights(y: np.ndarray) -> np.ndarray:
    n = len(y)
    pos = max(1, int(y.sum()))
    neg = max(1, int(n - y.sum()))
    w_pos = n / (2.0 * pos)
    w_neg = n / (2.0 * neg)
    return np.where(y == 1, w_pos, w_neg).astype(float)


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Column-wise clipping to reduce impact of extreme values (robust scaling helper).
    Fits per-column lower/upper quantiles on training fold only.
    """

    def __init__(self, q_low: float = 0.01, q_high: float = 0.99):
        self.q_low = q_low
        self.q_high = q_high
        self.lo_: np.ndarray | None = None
        self.hi_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: Any = None) -> "Winsorizer":
        X = np.asarray(X, dtype=float)
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
class Phenotype:
    name: str
    pattern: str


PHENOTYPES: dict[str, Phenotype] = {
    # Sub-phenotypes
    "cad_or_cabg": Phenotype("cad_or_cabg", r"\b(?:coronary artery disease|cabg|coronary artery bypass)\b"),
    "valve_disease_or_replacement": Phenotype(
        "valve_disease_or_replacement",
        r"\b(?:aortic valve replacement|mitral (?:stenosis|regurgitation)|valve replacement|valvular)\b",
    ),
    "congenital_hd": Phenotype(
        "congenital_hd",
        r"\b(?:congenital heart disease|ventricular septal defect|\basd\b|\bpda\b)\b",
    ),
    "aaa_or_aortic_aneurysm": Phenotype(
        "aaa_or_aortic_aneurysm",
        r"\b(?:abdominal aortic aneurysm|aortic aneurysm|\baaa\b)\b",
    ),
    # Umbrella: any of the above major CVD-ish groups (cardiac + vascular)
    "any_cvd_umbrella": Phenotype(
        "any_cvd_umbrella",
        r"|".join(
            [
                r"\b(?:coronary artery disease|cabg|coronary artery bypass)\b",
                r"\b(?:aortic valve replacement|mitral (?:stenosis|regurgitation)|valve replacement|valvular)\b",
                r"\b(?:congenital heart disease|ventricular septal defect|\basd\b|\bpda\b)\b",
                r"\b(?:abdominal aortic aneurysm|aortic aneurysm|\baaa\b)\b",
            ]
        ),
    ),
}


def build_labels(meta_csv: Path, phenotype_key: str) -> pd.DataFrame:
    meta = pd.read_csv(meta_csv)
    if "record" not in meta.columns:
        raise SystemExit(f"metadata CSV missing 'record' column: {meta_csv}")

    dx = meta.get("diagnoses_raw", pd.Series([""] * len(meta))).fillna("")
    dx_fb = meta.get("diagnoses_fallback_raw", pd.Series([""] * len(meta))).fillna("")
    dx_all = (dx + "; " + dx_fb).str.lower()

    if phenotype_key not in PHENOTYPES:
        raise SystemExit(f"Unknown phenotype '{phenotype_key}'. Options: {sorted(PHENOTYPES.keys())}")

    pat = PHENOTYPES[phenotype_key].pattern
    y = dx_all.str.contains(pat, regex=True).astype(int)

    out = pd.DataFrame(
        {
            "record": meta["record"],
            "label": y,
            "diagnoses_raw": meta.get("diagnoses_raw", ""),
            "age_years": age_to_numeric(meta["age_raw"]) if "age_raw" in meta.columns else np.nan,
            "sex_male": sex_to_numeric(meta["sex_raw"]) if "sex_raw" in meta.columns else np.nan,
        }
    )
    return out


def train_cv_logreg(X: pd.DataFrame, y: pd.Series, *, seed: int, folds: int) -> tuple[dict[str, Any], pd.DataFrame]:
    Xn = X.select_dtypes(include=[np.number]).copy()
    Xn.replace([np.inf, -np.inf], np.nan, inplace=True)
    Xn = Xn.dropna(axis=1, how="all")

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("winsor", Winsorizer(q_low=0.01, q_high=0.99)),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear", random_state=seed)),
        ]
    )

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    aucs: list[float] = []
    aprs: list[float] = []
    coef_rows: list[dict[str, Any]] = []

    yv = y.values.astype(int)
    for tr, te in skf.split(Xn, yv):
        sw = _class_weights(yv[tr])
        pipe.fit(Xn.iloc[tr], yv[tr], clf__sample_weight=sw)
        p = pipe.predict_proba(Xn.iloc[te])[:, 1]
        aucs.append(float(roc_auc_score(yv[te], p)))
        aprs.append(float(average_precision_score(yv[te], p)))

        coefs = pipe.named_steps["clf"].coef_.ravel()
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

    coef_df = (
        pd.DataFrame(coef_rows)
        .groupby("feature", as_index=False)
        .agg(
            coef_mean=("coef", "mean"),
            coef_std=("coef", "std"),
            coef_abs_mean=("coef", lambda x: float(np.mean(np.abs(x)))),
        )
        .sort_values("coef_abs_mean", ascending=False)
    )

    return metrics, coef_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, default=Path("results/mgh_all_20260118_173512"))
    ap.add_argument(
        "--metadata_csv",
        type=Path,
        default=Path("results/mgh_all_20260118_173512/mgh_metadata_summary.csv"),
    )
    ap.add_argument(
        "--phenotype",
        type=str,
        default="any_cvd_umbrella",
        help=f"Phenotype key: {', '.join(sorted(PHENOTYPES.keys()))}",
    )
    ap.add_argument(
        "--label_col",
        type=str,
        default="",
        help="If set, use this boolean/int column from metadata_csv as the label (e.g. narr_lvh, narr_afib). Overrides --phenotype.",
    )
    ap.add_argument(
        "--drop_features",
        type=str,
        default="",
        help="Comma-separated feature columns to drop from training (e.g. n_cycles). Applied to PyHEARTS feature columns only.",
    )
    ap.add_argument("--include_demographics", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min_pos", type=int, default=15, help="Fail fast if too few positives after join")
    args = ap.parse_args()

    feat = build_feature_table(args.run_dir)
    if args.label_col.strip():
        meta = pd.read_csv(args.metadata_csv)
        col = args.label_col.strip()
        if col not in meta.columns:
            raise SystemExit(f"label_col '{col}' not found in metadata CSV.")
        lab = pd.DataFrame(
            {
                "record": meta["record"],
                "label": meta[col].fillna(False).astype(bool).astype(int),
                "diagnoses_raw": meta.get("diagnoses_raw", ""),
                "age_years": age_to_numeric(meta["age_raw"]) if "age_raw" in meta.columns else np.nan,
                "sex_male": sex_to_numeric(meta["sex_raw"]) if "sex_raw" in meta.columns else np.nan,
            }
        )
        phenotype_name = col
    else:
        lab = build_labels(args.metadata_csv, args.phenotype)
        phenotype_name = args.phenotype

    df = feat.merge(lab, on="record", how="inner")
    if df.empty:
        raise SystemExit("No overlap between run features and metadata records.")

    y = df["label"].astype(int)
    if int(y.sum()) < int(args.min_pos):
        raise SystemExit(f"Too few positives for {phenotype_name}: pos={int(y.sum())} < min_pos={args.min_pos}")

    # Prevent leakage: features only from PyHEARTS + (optional) demographics
    drop_set = {s.strip() for s in args.drop_features.split(",") if s.strip()}
    feature_cols = [c for c in feat.columns if c != "record" and c not in drop_set]
    if args.include_demographics:
        feature_cols += ["age_years", "sex_male"]

    X = df[feature_cols]
    metrics, coef_df = train_cv_logreg(X, y, seed=args.seed, folds=args.folds)

    out_tag = f"mgh_cvd_{phenotype_name}" + ("_adj_demo" if args.include_demographics else "")
    out_table = args.run_dir / f"{out_tag}_modeling_table.csv"
    out_metrics = args.run_dir / f"{out_tag}_metrics.json"
    out_coefs = args.run_dir / f"{out_tag}_logreg_feature_coeffs.csv"

    df.to_csv(out_table, index=False)
    out_metrics.write_text(json.dumps({"phenotype": phenotype_name, **metrics}, indent=2), encoding="utf-8")
    coef_df.to_csv(out_coefs, index=False)

    print(f"Wrote: {out_table}")
    print(f"Wrote: {out_metrics}")
    print(f"Wrote: {out_coefs}")
    print(json.dumps({"phenotype": phenotype_name, **metrics}, indent=2))


if __name__ == "__main__":
    main()


