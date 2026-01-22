#!/usr/bin/env python3
"""
AutoAge age-prediction robustness analyses.

Runs:
- Accuracy reporting (MAE, RMSE, R²; CV + held-out test)
- Quality filtering (minimum analyzable cycles; minimum mean fit R²; optional max mean RMSE)
- Lead/device adjustment:
    AutoAge does not provide explicit ECG lead labels in this pipeline; we include `Device` as a proxy covariate.
- Feature-set ablations:
    - full
    - rr_only (RR_interval_ms aggregate features)
    - morphology_only (excludes global interval features like RR/PP/PR/QRS/QT/QTc/ST)

Outputs:
  results/age_prediction_autoage_robustness_<timestamp>/
    - robustness_results.json
    - robustness_results.csv
    - AUTOAGE_AGE_PREDICTION_ROBUSTNESS.md
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent


def find_latest_autoage_results_dir(project_root: Path) -> Path:
    candidates = sorted(project_root.glob("results/autoage_*"))
    if not candidates:
        raise ValueError("No autoage results directory found under results/ (results/autoage_*).")
    return candidates[-1]


def load_subject_info(subject_info_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(subject_info_csv)
    df["subject"] = df["ID"].astype(int).astype(str).str.zfill(4)

    def mid(age_range: str) -> float | None:
        if pd.isna(age_range):
            return None
        m = re.match(r"\s*(\d+)\s*-\s*(\d+)\s*$", str(age_range))
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            return (lo + hi) / 2.0
        try:
            return float(age_range)
        except Exception:
            return None

    df["age"] = df["age_range"].apply(mid).astype(float)

    def sex_norm(s) -> str | None:
        if pd.isna(s):
            return None
        s = str(s).strip().lower()
        if s in {"male", "m", "1"}:
            return "M"
        if s in {"female", "f", "0"}:
            return "F"
        return None

    # subject_info_clean.csv has both Sex and sex columns; prefer the cleaned `sex`
    sex_col = "sex" if "sex" in df.columns else ("Sex" if "Sex" in df.columns else None)
    df["sex_norm"] = df[sex_col].apply(sex_norm) if sex_col else None

    return df.set_index("subject")


def aggregate_features_per_subject(results_dir: Path, subject: str) -> pd.Series | None:
    """
    Mirrors analysis_scripts/predict_age_from_features_autoage.py aggregation.
    """
    output_file = results_dir / f"{subject}_output.csv"
    if not output_file.exists():
        return None

    try:
        df = pd.read_csv(output_file)
        if len(df) == 0:
            return None

        # Keep cycle_trend; exclude index-like columns
        exclude_cols = [
            "R_global_center_idx",
            "P_global_center_idx",
            "Q_global_center_idx",
            "S_global_center_idx",
            "T_global_center_idx",
            "P_global_le_idx",
            "P_global_ri_idx",
            "Q_global_le_idx",
            "Q_global_ri_idx",
            "R_global_le_idx",
            "R_global_ri_idx",
            "S_global_le_idx",
            "S_global_ri_idx",
            "T_global_le_idx",
            "T_global_ri_idx",
            "P_fwhm_global_le_idx",
            "P_fwhm_global_ri_idx",
            "Q_fwhm_global_le_idx",
            "Q_fwhm_global_ri_idx",
            "R_fwhm_global_le_idx",
            "R_fwhm_global_ri_idx",
            "S_fwhm_global_le_idx",
            "S_fwhm_global_ri_idx",
            "T_fwhm_global_le_idx",
            "T_fwhm_global_ri_idx",
            "P_center_idx",
            "P_le_idx",
            "P_ri_idx",
            "Q_center_idx",
            "Q_le_idx",
            "Q_ri_idx",
            "R_center_idx",
            "R_le_idx",
            "R_ri_idx",
            "S_center_idx",
            "S_le_idx",
            "S_ri_idx",
            "T_center_idx",
            "T_le_idx",
            "T_ri_idx",
            "P_fwhm_le_idx",
            "P_fwhm_ri_idx",
            "Q_fwhm_le_idx",
            "Q_fwhm_ri_idx",
            "R_fwhm_le_idx",
            "R_fwhm_ri_idx",
            "S_fwhm_le_idx",
            "S_fwhm_ri_idx",
            "T_fwhm_le_idx",
            "T_fwhm_ri_idx",
            "P_gauss_center",
            "Q_gauss_center",
            "R_gauss_center",
            "S_gauss_center",
            "T_gauss_center",
        ]

        aggregated: dict[str, float] = {}
        for col in df.columns:
            if col in exclude_cols:
                continue
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(values) == 0:
                aggregated[f"{col}_mean"] = np.nan
                aggregated[f"{col}_std"] = np.nan
                aggregated[f"{col}_median"] = np.nan
                aggregated[f"{col}_min"] = np.nan
                aggregated[f"{col}_max"] = np.nan
                aggregated[f"{col}_count"] = 0.0
            else:
                aggregated[f"{col}_mean"] = float(values.mean())
                aggregated[f"{col}_std"] = float(values.std())
                aggregated[f"{col}_median"] = float(values.median())
                aggregated[f"{col}_min"] = float(values.min())
                aggregated[f"{col}_max"] = float(values.max())
                aggregated[f"{col}_count"] = float(len(values))

        aggregated["n_cycles"] = float(len(df))
        aggregated["r_squared_mean"] = float(pd.to_numeric(df.get("r_squared"), errors="coerce").mean()) if "r_squared" in df.columns else np.nan
        aggregated["r_squared_std"] = float(pd.to_numeric(df.get("r_squared"), errors="coerce").std()) if "r_squared" in df.columns else np.nan
        aggregated["rmse_mean"] = float(pd.to_numeric(df.get("rmse"), errors="coerce").mean()) if "rmse" in df.columns else np.nan

        return pd.Series(aggregated, name=subject)
    except Exception:
        return None


def build_dataset(subject_info: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    output_files = list(results_dir.glob("*_output.csv"))
    subjects = [f.stem.replace("_output", "") for f in output_files]
    rows: list[pd.Series] = []

    for s in subjects:
        if s not in subject_info.index:
            continue
        age = subject_info.loc[s, "age"]
        if not np.isfinite(age):
            continue
        feats = aggregate_features_per_subject(results_dir, s)
        if feats is None:
            continue
        feats["age"] = float(age)
        feats["sex"] = subject_info.loc[s, "sex_norm"]
        # covariates
        for cov in ["Device", "Length", "BMI"]:
            feats[cov] = float(subject_info.loc[s, cov]) if cov in subject_info.columns and pd.notna(subject_info.loc[s, cov]) else np.nan
        feats["subject"] = s
        rows.append(feats)

    if not rows:
        raise ValueError("No subjects with both age and features found.")
    return pd.DataFrame(rows).set_index("subject")


def apply_quality_filters(
    df: pd.DataFrame,
    *,
    min_cycles: int = 0,
    min_r2_mean: float | None = None,
    max_rmse_mean: float | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if min_cycles > 0 and "n_cycles" in out.columns:
        out = out[out["n_cycles"] >= float(min_cycles)]
    if min_r2_mean is not None and "r_squared_mean" in out.columns:
        out = out[out["r_squared_mean"] >= float(min_r2_mean)]
    if max_rmse_mean is not None and "rmse_mean" in out.columns:
        out = out[out["rmse_mean"] <= float(max_rmse_mean)]
    return out


def feature_subset_cols(df: pd.DataFrame, subset: str) -> list[str]:
    forbidden = {"age", "sex", "Device", "Length", "BMI", "subject"}
    cols = [c for c in df.columns if c not in forbidden]

    if subset == "full":
        return cols

    if subset == "rr_only":
        return [c for c in cols if c.startswith("RR_interval_ms_")]

    if subset == "morphology_only":
        drop_re = re.compile(
            "|".join(
                [
                    r"^RR_interval_ms_",
                    r"^PP_interval_ms_",
                    r"^PR_interval_ms_",
                    r"^PR_segment_ms_",
                    r"^QRS_interval_ms_",
                    r"^ST_segment_ms_",
                    r"^ST_interval_ms_",
                    r"^QT_interval_ms_",
                    r"^QTc_",
                ]
            )
        )
        return [c for c in cols if not drop_re.search(c)]

    raise ValueError(f"Unknown subset={subset!r}. Expected: full | rr_only | morphology_only")


@dataclass
class RunConfig:
    name: str
    subset: str  # full | rr_only | morphology_only
    min_cycles: int = 0
    min_r2_mean: float | None = None
    max_rmse_mean: float | None = None
    include_sex: bool = True
    include_device: bool = False
    include_length: bool = False
    include_bmi: bool = False
    n_features_select: int = 100  # 0 disables selection
    seed: int = 42


def evaluate_models(X: np.ndarray, y: np.ndarray, seed: int) -> dict[str, dict]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=seed)

    models = {
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_samples_split=5,
            random_state=seed,
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    results: dict[str, dict] = {}
    for name, model in models.items():
        cv_mae = -cross_val_score(model, Xs, y, cv=cv, scoring="neg_mean_absolute_error")
        cv_r2 = cross_val_score(model, Xs, y, cv=cv, scoring="r2")

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results[name] = {
            "cv_mae_mean": float(cv_mae.mean()),
            "cv_mae_std": float(cv_mae.std()),
            "cv_r2_mean": float(cv_r2.mean()),
            "cv_r2_std": float(cv_r2.std()),
            "test_mae": float(mean_absolute_error(y_test, pred)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "test_r2": float(r2_score(y_test, pred)),
        }
    return results


def run_config(df: pd.DataFrame, cfg: RunConfig) -> dict:
    dff = apply_quality_filters(df, min_cycles=cfg.min_cycles, min_r2_mean=cfg.min_r2_mean, max_rmse_mean=cfg.max_rmse_mean)
    y = dff["age"].values.astype(float)

    feat_cols = feature_subset_cols(dff, cfg.subset)
    X_df = dff[feat_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    blocks = [X_df.values]
    names = list(feat_cols)

    if cfg.include_sex:
        sex = dff["sex"].astype(str).values
        sex_enc = np.array([1.0 if s == "M" else 0.0 if s == "F" else 0.5 for s in sex], dtype=float).reshape(-1, 1)
        blocks.append(sex_enc)
        names.append("sex_M")

    def add_numeric(col: str):
        arr = pd.to_numeric(dff[col], errors="coerce")
        arr = arr.replace([np.inf, -np.inf], np.nan)
        arr = arr.fillna(arr.median())
        blocks.append(arr.values.astype(float).reshape(-1, 1))
        names.append(col)

    if cfg.include_device and "Device" in dff.columns:
        add_numeric("Device")
    if cfg.include_length and "Length" in dff.columns:
        add_numeric("Length")
    if cfg.include_bmi and "BMI" in dff.columns:
        add_numeric("BMI")

    X = np.column_stack(blocks)

    k = int(cfg.n_features_select)
    if k > 0 and X.shape[1] > k:
        variances = np.var(X, axis=0)
        valid = variances > 1e-8
        Xf = X[:, valid]
        yf = y
        names_f = [n for n, v in zip(names, valid) if v]
        k2 = min(k, Xf.shape[1])
        sel = SelectKBest(f_regression, k=k2)
        X = sel.fit_transform(Xf, yf)
        names = [names_f[i] for i in sel.get_support(indices=True)]

    results = evaluate_models(X, y, seed=cfg.seed)
    # Select "best model" by cross-validation MAE (not by the held-out test set),
    # so the reported selection criterion matches manuscript wording.
    best = min(results.keys(), key=lambda m: results[m]["cv_mae_mean"])

    return {
        "config": asdict(cfg),
        "n_subjects": int(len(dff)),
        "age_mean": float(dff["age"].mean()),
        "age_std": float(dff["age"].std()),
        "n_features_used": int(X.shape[1]),
        "best_model": best,
        "best_test_mae": results[best]["test_mae"],
        "best_test_r2": results[best]["test_r2"],
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AutoAge age prediction robustness analyses.")
    p.add_argument("--results-dir", type=Path, default=None, help="AutoAge processed outputs dir (contains *_output.csv).")
    p.add_argument(
        "--subject-info-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "autoage" / "subject_info_clean.csv",
        help="AutoAge subject info CSV.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir or find_latest_autoage_results_dir(PROJECT_ROOT)
    subject_info = load_subject_info(args.subject_info_csv)
    df = build_dataset(subject_info, results_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "results" / f"age_prediction_autoage_robustness_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[RunConfig] = [
        # Baseline (no quality filtering)
        RunConfig(name="baseline_full", subset="full", include_device=False),
        RunConfig(name="baseline_full_plus_device", subset="full", include_device=True),
        RunConfig(name="baseline_rr_only", subset="rr_only", include_device=False, n_features_select=0),
        RunConfig(name="baseline_rr_only_plus_device", subset="rr_only", include_device=True, n_features_select=0),
        RunConfig(name="baseline_morphology_only", subset="morphology_only", include_device=False),
        RunConfig(name="baseline_morphology_only_plus_device", subset="morphology_only", include_device=True),
        # Quality filters (moderate)
        RunConfig(name="qf1_full_plus_device", subset="full", min_cycles=50, min_r2_mean=0.75, include_device=True),
        RunConfig(name="qf1_rr_only_plus_device", subset="rr_only", min_cycles=50, min_r2_mean=0.75, include_device=True, n_features_select=0),
        RunConfig(name="qf1_morphology_only_plus_device", subset="morphology_only", min_cycles=50, min_r2_mean=0.75, include_device=True),
        # Quality filters (stricter)
        RunConfig(name="qf2_full_plus_device", subset="full", min_cycles=100, min_r2_mean=0.80, include_device=True),
        RunConfig(name="qf2_rr_only_plus_device", subset="rr_only", min_cycles=100, min_r2_mean=0.80, include_device=True, n_features_select=0),
        RunConfig(name="qf2_morphology_only_plus_device", subset="morphology_only", min_cycles=100, min_r2_mean=0.80, include_device=True),
    ]

    outputs: list[dict] = []
    for cfg in runs:
        res = run_config(df, cfg)
        outputs.append(res)
        print(
            f"{cfg.name:35s} n={res['n_subjects']:4d}  best={res['best_model']:15s}  "
            f"MAE={res['best_test_mae']:.2f}  R2={res['best_test_r2']:.3f}"
        )

    (out_dir / "robustness_results.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")

    rows: list[dict] = []
    for o in outputs:
        cfg = o["config"]
        best = o["best_model"]
        rows.append(
            {
                "name": cfg["name"],
                "subset": cfg["subset"],
                "min_cycles": cfg["min_cycles"],
                "min_r2_mean": cfg["min_r2_mean"],
                "include_device": cfg["include_device"],
                "n_subjects": o["n_subjects"],
                "best_model": best,
                "best_test_mae": o["best_test_mae"],
                "best_test_r2": o["best_test_r2"],
                "cv_mae_mean": o["results"][best]["cv_mae_mean"],
                "cv_r2_mean": o["results"][best]["cv_r2_mean"],
            }
        )

    pd.DataFrame(rows).to_csv(out_dir / "robustness_results.csv", index=False)

    md_lines = [
        f"# AutoAge age prediction robustness ({ts})",
        "",
        f"- results_dir: `{results_dir}`",
        f"- n_subjects (raw): {len(df)}",
        "",
        "## Key note on lead adjustment",
        "AutoAge does not provide explicit ECG lead identifiers in this pipeline; we include `Device` as a proxy covariate (acquisition device/channel differences).",
        "",
        "## Results (best model per setting)",
        "",
    ]
    for r in rows:
        md_lines.append(
            f"- **{r['name']}**: n={r['n_subjects']}, best={r['best_model']}, "
            f"test MAE={r['best_test_mae']:.2f}, test R²={r['best_test_r2']:.3f}, "
            f"CV MAE={r['cv_mae_mean']:.2f}, CV R²={r['cv_r2_mean']:.3f}"
        )
    (out_dir / "AUTOAGE_AGE_PREDICTION_ROBUSTNESS.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\nSaved: {out_dir}")


if __name__ == "__main__":
    main()


