#!/usr/bin/env python3
"""LUDB held-out morphology validation against a frozen config lockfile.

Primary non-circular evaluation for PyHEARTS public defaults. Metrics match the
QTDB peak-reference protocol (no new metrics after seeing results):

* unconditional sensitivity at ±40 ms and ±150 ms (and full Se(tolerance) curve)
* conditional signed error: bias (median) + scatter (σ, IQR)

Lead policy: limb-preferred (II when present). Cardiologist annotations are a
reference with known inter-observer variability, not absolute ground truth.

Example
-------
::

    # 1) Freeze + commit BEFORE any scoring (already done for v1)
    # 2) Primary run
    PYTHONPATH=. python3 validation/run_ludb_heldout.py \\
        --frozen validation/config_frozen_v1.json \\
        --data-dir data/ludb \\
        --out-dir validation/output/ludb_heldout_v1

    # 3) Sensitivity sweep ONLY after primary numbers are locked
    PYTHONPATH=. python3 validation/run_ludb_heldout.py \\
        --frozen validation/config_frozen_v1.json \\
        --data-dir data/ludb \\
        --out-dir validation/output/ludb_sensitivity_v1 \\
        --sensitivity
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy.optimize import linear_sum_assignment

from pyhearts import PyHEARTS, load_wfdb_signal
from pyhearts._morphology.config import ProcessCycleConfig as MorphCfg
from pyhearts.config import ProcessCycleConfig as PubCfg
from pyhearts.core.analyzer import PIPELINE_VERSION
from pyhearts.version import __version__

WAVES = ("P", "R", "T")
SYMBOL = {"P": "p", "R": "N", "T": "t"}
# Pre-registered tolerances (do not invent new ones after seeing results).
TOLERANCES_MS = (20.0, 40.0, 80.0, 150.0)
PRIMARY_TOLS_MS = (40.0, 150.0)
SE_CURVE_MS = np.arange(5.0, 155.0, 5.0)

# Sensitivity factors around frozen knobs (±25%, ±50%).
SENSITIVITY_FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5)

LUDB_LEAD_EXT = {
    "i": "i",
    "ii": "ii",
    "iii": "iii",
    "avr": "avr",
    "avl": "avl",
    "avf": "avf",
    "v1": "v1",
    "v2": "v2",
    "v3": "v3",
    "v4": "v4",
    "v5": "v5",
    "v6": "v6",
}


def load_frozen(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _filter_known(cls: type, payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {f.name: f for f in fields(cls)}
    out: dict[str, Any] = {}
    for k, v in payload.items():
        if k not in allowed:
            continue
        typ = str(allowed[k].type)
        if isinstance(v, list) and ("Tuple" in typ or "tuple" in typ):
            v = tuple(v)
        out[k] = v
    return out


def morph_cfg_from_frozen(frozen: dict[str, Any], **overrides: Any) -> MorphCfg:
    base = MorphCfg(**_filter_known(MorphCfg, frozen["core_config"]))
    if overrides:
        return replace(base, **overrides)
    return base


def pub_cfg_from_frozen(frozen: dict[str, Any]) -> PubCfg | None:
    raw = frozen.get("t_config")
    if raw is None:
        return None
    return PubCfg(**_filter_known(PubCfg, raw))


def build_analyzer(
    fs: float,
    frozen: dict[str, Any],
    *,
    core_overrides: dict[str, Any] | None = None,
) -> PyHEARTS:
    """Rebuild analyzer from lockfile (not from live preset factories)."""
    core = morph_cfg_from_frozen(frozen, **(core_overrides or {}))
    analyzer = PyHEARTS(
        sampling_rate=float(fs),
        species=None,
        verbose=False,
        cfg=core,
    )
    # Public analyzer wraps morphology; restore dual-config behavior from freeze.
    analyzer.species = frozen["analyzer"].get("species", "human")
    analyzer._core_cfg = analyzer.cfg
    t_cfg = pub_cfg_from_frozen(frozen)
    if t_cfg is not None:
        analyzer._t_cfg = replace(t_cfg, version=PIPELINE_VERSION)
    analyzer.apply_record_t = bool(frozen["analyzer"].get("apply_record_t", True))
    analyzer.pipeline_version = frozen.get("pipeline_version", PIPELINE_VERSION)
    return analyzer


def peak_samples(ann, wave: str) -> np.ndarray:
    sym = SYMBOL[wave]
    samples = np.asarray(ann.sample, dtype=float)
    symbols = np.asarray(ann.symbol)
    return samples[symbols == sym]


def nearest_manual_wave(
    samples: np.ndarray,
    r_sample: float,
    *,
    before_ms: float | None,
    after_ms: float | None,
    fs: float,
) -> float:
    values = np.asarray(samples, dtype=float)
    if before_ms is not None:
        lo = r_sample - before_ms * fs / 1000.0
        values = values[(values < r_sample) & (values >= lo)]
    if after_ms is not None:
        hi = r_sample + after_ms * fs / 1000.0
        values = values[(values > r_sample) & (values <= hi)]
    if values.size == 0:
        return np.nan
    return float(values[np.argmin(np.abs(values - r_sample))])


def manual_beats_from_ann(ann, fs: float) -> pd.DataFrame:
    p = peak_samples(ann, "P")
    r = peak_samples(ann, "R")
    t = peak_samples(ann, "T")
    rows = []
    for i, r_sample in enumerate(r):
        rows.append(
            {
                "manual_beat_index": i,
                "manual_R_sample": float(r_sample),
                "manual_P_sample": nearest_manual_wave(
                    p, float(r_sample), before_ms=400.0, after_ms=None, fs=fs
                ),
                "manual_T_sample": nearest_manual_wave(
                    t, float(r_sample), before_ms=None, after_ms=700.0, fs=fs
                ),
            }
        )
    return pd.DataFrame(rows)


def match_r_anchors(
    manual: pd.DataFrame,
    detected: pd.DataFrame,
    fs: float,
    max_error_ms: float = 150.0,
) -> dict[int, int]:
    manual_r = manual["manual_R_sample"].to_numpy(dtype=float)
    detected_r = pd.to_numeric(
        detected.get("R_global_center_idx"), errors="coerce"
    ).to_numpy(dtype=float)
    valid = np.flatnonzero(np.isfinite(detected_r))
    if manual_r.size == 0 or valid.size == 0:
        return {}
    cost = np.abs(manual_r[:, None] - detected_r[valid][None, :])
    mi, di = linear_sum_assignment(cost)
    lim = max_error_ms * fs / 1000.0
    return {
        int(a): int(valid[b])
        for a, b in zip(mi, di)
        if cost[a, b] <= lim
    }


def ludb_ann_ext(lead_name: str) -> str | None:
    key = str(lead_name).strip().lower().replace(" ", "")
    return LUDB_LEAD_EXT.get(key)


def list_ludb_records(data_dir: Path) -> list[str]:
    """Return record ids; supports flat or PhysioNet ``data/<id>`` layout."""
    heas = list(data_dir.glob("*.hea")) + list(data_dir.glob("data/*.hea"))
    return sorted({p.stem for p in heas if p.stem.isdigit()}, key=lambda s: int(s))


def record_path(data_dir: Path, record: str) -> Path:
    for base in (data_dir / record, data_dir / "data" / record):
        if Path(f"{base}.hea").exists():
            return base
    raise FileNotFoundError(f"LUDB record {record} not found under {data_dir}")



def score_record(
    record: str,
    data_dir: Path,
    frozen: dict[str, Any],
    *,
    core_overrides: dict[str, Any] | None = None,
) -> pd.DataFrame:
    path = record_path(data_dir, record)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ecg, fs, lead_idx, lead_name = load_wfdb_signal(path, policy="limb_preferred")
        ext = ludb_ann_ext(lead_name)
        if ext is None or not Path(f"{path}.{ext}").exists():
            # Fall back to lead II annotation if present, else first available.
            for candidate in ("ii", "i", "v5", "v2"):
                if Path(f"{path}.{candidate}").exists():
                    rec = wfdb.rdrecord(str(path))
                    names = [str(n) for n in rec.sig_name]
                    lower = [n.lower() for n in names]
                    if candidate in lower:
                        lead_idx = lower.index(candidate)
                        lead_name = names[lead_idx]
                        ecg = np.asarray(rec.p_signal[:, lead_idx], dtype=float)
                        fs = float(rec.fs)
                        ext = candidate
                        break
            else:
                return pd.DataFrame()
        ann = wfdb.rdann(str(path), ext)
        manual = manual_beats_from_ann(ann, fs)
        analyzer = build_analyzer(fs, frozen, core_overrides=core_overrides)
        features, _ = analyzer.analyze_ecg(ecg)

    matches = match_r_anchors(manual, features, fs)
    rows = []
    for manual_i, mrow in manual.iterrows():
        detected_i = matches.get(int(manual_i))
        out: dict[str, Any] = {
            "record": record,
            "fs": fs,
            "lead_index": lead_idx,
            "lead_name": lead_name,
            "ann_ext": ext,
            "manual_beat_index": int(mrow["manual_beat_index"]),
            "detected_cycle_index": detected_i,
            "r_anchor_matched": detected_i is not None,
            "signal_inverted": bool(getattr(analyzer, "signal_inverted", False)),
        }
        for wave in WAVES:
            manual_sample = float(mrow[f"manual_{wave}_sample"])
            detected_sample = np.nan
            if detected_i is not None:
                detected_sample = pd.to_numeric(
                    pd.Series([features.iloc[detected_i].get(f"{wave}_global_center_idx")]),
                    errors="coerce",
                ).iloc[0]
            signed = (
                (float(detected_sample) - manual_sample) * 1000.0 / fs
                if np.isfinite(manual_sample) and np.isfinite(detected_sample)
                else np.nan
            )
            out[f"manual_{wave}_sample"] = manual_sample
            out[f"detected_{wave}_sample"] = (
                float(detected_sample) if np.isfinite(detected_sample) else np.nan
            )
            out[f"{wave}_signed_err_ms"] = signed
            out[f"{wave}_abs_err_ms"] = abs(signed) if np.isfinite(signed) else np.nan
            out[f"{wave}_manual_present"] = bool(np.isfinite(manual_sample))
            out[f"{wave}_detected"] = bool(np.isfinite(detected_sample))
        rows.append(out)
    return pd.DataFrame(rows)


def unconditional_sensitivity(beats: pd.DataFrame, wave: str, tol_ms: float) -> dict:
    present = beats[f"{wave}_manual_present"]
    n_ref = int(present.sum())
    errs = beats.loc[present, f"{wave}_abs_err_ms"]
    n_hit = int((errs <= tol_ms).fillna(False).sum())
    return {
        "wave": wave,
        "tol_ms": float(tol_ms),
        "n_ref": n_ref,
        "n_hit": n_hit,
        "sensitivity": n_hit / n_ref if n_ref else np.nan,
    }


def conditional_localization(beats: pd.DataFrame, wave: str) -> dict:
    err = beats[f"{wave}_signed_err_ms"].dropna().to_numpy(dtype=float)
    if err.size == 0:
        return {
            "wave": wave,
            "n": 0,
            "bias_median_ms": np.nan,
            "scatter_sigma_ms": np.nan,
            "scatter_iqr_ms": np.nan,
            "median_abs_ms": np.nan,
            "mean_abs_ms": np.nan,
        }
    return {
        "wave": wave,
        "n": int(err.size),
        "bias_median_ms": float(np.median(err)),
        "scatter_sigma_ms": float(np.std(err, ddof=1)) if err.size > 1 else np.nan,
        "scatter_iqr_ms": float(np.subtract(*np.percentile(err, [75, 25]))),
        "median_abs_ms": float(np.median(np.abs(err))),
        "mean_abs_ms": float(np.mean(np.abs(err))),
    }


def summarize_primary(beats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    se_rows = []
    for wave in WAVES:
        for tol in TOLERANCES_MS:
            se_rows.append(unconditional_sensitivity(beats, wave, tol))
    se = pd.DataFrame(se_rows)

    loc = pd.DataFrame([conditional_localization(beats, w) for w in WAVES])

    curve_rows = []
    for wave in WAVES:
        for tol in SE_CURVE_MS:
            row = unconditional_sensitivity(beats, wave, float(tol))
            curve_rows.append(row)
    curves = pd.DataFrame(curve_rows)
    return se, loc, curves


def run_corpus(
    data_dir: Path,
    frozen: dict[str, Any],
    *,
    max_records: int | None = None,
    core_overrides: dict[str, Any] | None = None,
) -> pd.DataFrame:
    records = list_ludb_records(data_dir)
    if max_records is not None:
        records = records[:max_records]
    if not records:
        raise FileNotFoundError(f"No LUDB .hea records under {data_dir}")

    tables: list[pd.DataFrame] = []
    for i, record in enumerate(records, 1):
        scored = score_record(
            record, data_dir, frozen, core_overrides=core_overrides
        )
        if scored.empty:
            print(f"[{i}/{len(records)}] {record}: SKIP (no annotation)")
            continue
        tables.append(scored)
        print(
            f"[{i}/{len(records)}] {record}: "
            f"manual_R={len(scored)} matched_R={int(scored.r_anchor_matched.sum())} "
            f"lead={scored.lead_name.iloc[0]}"
        )
    if not tables:
        raise RuntimeError("No LUDB records scored")
    return pd.concat(tables, ignore_index=True)


def plot_se_curves(curves: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = {"P": "#2ca02c", "R": "#1f77b4", "T": "#d62728"}
    for wave in WAVES:
        df = curves[curves.wave == wave]
        ax.plot(df["tol_ms"], df["sensitivity"], lw=2, label=wave, color=colors[wave])
    for tol in PRIMARY_TOLS_MS:
        ax.axvline(tol, color="0.5", ls=":", lw=1)
    ax.set_xlabel("Match tolerance (ms)")
    ax.set_ylabel("Unconditional sensitivity")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, 155)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_sweep(sweep: pd.DataFrame, out_path: Path) -> None:
    knobs = sweep["knob"].unique()
    fig, axes = plt.subplots(len(knobs), 1, figsize=(8.0, 2.8 * len(knobs)), sharex=False)
    if len(knobs) == 1:
        axes = [axes]
    colors = {"P": "#2ca02c", "R": "#1f77b4", "T": "#d62728"}
    for ax, knob in zip(axes, knobs):
        sub = sweep[sweep.knob == knob]
        for wave in WAVES:
            w = sub[sub.wave == wave]
            ax.plot(w["factor"], w["sensitivity_40ms"], "o-", lw=1.8, label=wave, color=colors[wave])
        ax.axvline(1.0, color="0.4", ls="--", lw=1)
        ax.set_ylabel("Uncond. Se @ ±40 ms")
        ax.set_title(f"Sensitivity to {knob}")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, loc="best")
    axes[-1].set_xlabel("Factor relative to frozen value")
    fig.suptitle(
        "LUDB held-out sensitivity sweep (one-at-a-time; primary metric Se@40 ms)",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_primary_outputs(
    out_dir: Path,
    frozen_path: Path,
    frozen: dict[str, Any],
    beats: pd.DataFrame,
    se: pd.DataFrame,
    loc: pd.DataFrame,
    curves: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    beats.to_csv(out_dir / "ludb_beats.csv", index=False)
    se.to_csv(out_dir / "ludb_unconditional_se.csv", index=False)
    loc.to_csv(out_dir / "ludb_conditional_localization.csv", index=False)
    curves.to_csv(out_dir / "ludb_se_tolerance_curve.csv", index=False)
    plot_se_curves(
        curves,
        out_dir / "ludb_se_tolerance_curve.png",
        title=(
            f"Se(tolerance) on LUDB held-out — {frozen['validation_id']} "
            f"(package {__version__})"
        ),
    )
    primary = se[se.tol_ms.isin(PRIMARY_TOLS_MS)].copy()
    meta = {
        "role": "held_out_primary",
        "corpus": "LUDB",
        "frozen_config": str(frozen_path),
        "validation_id": frozen["validation_id"],
        "arm": frozen["arm"],
        "package_version": __version__,
        "frozen_git_sha": frozen.get("git_sha"),
        "frozen_at_utc": frozen.get("frozen_at_utc"),
        "n_records": int(beats["record"].nunique()),
        "n_manual_beats": int(len(beats)),
        "lead_policy": "limb_preferred",
        "metrics_locked": [
            "unconditional_Se_at_40_and_150_ms",
            "conditional_signed_error_bias_and_scatter",
            "Se_tolerance_curve",
        ],
        "primary_se": primary.to_dict(orient="records"),
        "conditional_localization": loc.to_dict(orient="records"),
        "protocol_note": (
            "Config was frozen and committed before this run. LUDB was not used "
            "for parameter selection. QTDB/AA/PTB-XL are not held-out for this claim."
        ),
    }
    (out_dir / "ludb_heldout_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print("\n=== Unconditional Se (primary tols) ===")
    print(
        primary.pivot_table(index="wave", columns="tol_ms", values="sensitivity")
        .reindex(WAVES)
        .round(4)
        .to_string()
    )
    print("\n=== Conditional localization (bias / scatter) ===")
    print(loc.set_index("wave").reindex(WAVES).round(3).to_string())


def sensitivity_overrides(
    frozen: dict[str, Any],
    knob: str,
    factor: float,
) -> dict[str, Any]:
    knobs = frozen["knobs_for_sensitivity_sweep"]
    if knob == "rpeak_prominence_multiplier":
        return {"rpeak_prominence_multiplier": float(knobs[knob]) * factor}
    if knob == "epoch_corr_thresh":
        val = float(knobs[knob]) * factor
        return {"epoch_corr_thresh": float(np.clip(val, 0.05, 0.99))}
    if knob == "bound_factor":
        val = float(knobs[knob]) * factor
        return {"bound_factor": float(np.clip(val, 0.05, 0.95))}
    if knob == "snr_mad_multiplier":
        base = knobs[knob]
        return {
            "snr_mad_multiplier": {
                k: float(v) * factor for k, v in base.items()
            }
        }
    raise ValueError(knob)


def run_sensitivity(
    data_dir: Path,
    frozen: dict[str, Any],
    out_dir: Path,
    *,
    max_records: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / "PRIMARY_LOCKED.json"
    # Require explicit acknowledgement that primary was reported first.
    primary_meta = Path("validation/output/ludb_heldout_v1/ludb_heldout_meta.json")
    if not primary_meta.exists():
        raise SystemExit(
            "Sensitivity sweep refused: primary LUDB results not found at "
            f"{primary_meta}. Run the primary held-out evaluation first and leave "
            "those numbers locked before sweeping."
        )
    rows = []
    knob_names = [
        "rpeak_prominence_multiplier",
        "epoch_corr_thresh",
        "snr_mad_multiplier",
        "bound_factor",
    ]
    for knob in knob_names:
        for factor in SENSITIVITY_FACTORS:
            overrides = sensitivity_overrides(frozen, knob, factor)
            print(f"\n--- sweep {knob} x{factor} overrides={overrides} ---")
            beats = run_corpus(
                data_dir,
                frozen,
                max_records=max_records,
                core_overrides=overrides,
            )
            for wave in WAVES:
                se40 = unconditional_sensitivity(beats, wave, 40.0)
                se150 = unconditional_sensitivity(beats, wave, 150.0)
                loc = conditional_localization(beats, wave)
                rows.append(
                    {
                        "knob": knob,
                        "factor": float(factor),
                        "override_json": json.dumps(overrides),
                        "wave": wave,
                        "sensitivity_40ms": se40["sensitivity"],
                        "sensitivity_150ms": se150["sensitivity"],
                        "n_ref": se40["n_ref"],
                        "bias_median_ms": loc["bias_median_ms"],
                        "scatter_sigma_ms": loc["scatter_sigma_ms"],
                    }
                )
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out_dir / "ludb_sensitivity_sweep.csv", index=False)
    plot_sensitivity_sweep(sweep, out_dir / "ludb_sensitivity_sweep.png")
    meta = {
        "role": "held_out_sensitivity_sweep",
        "corpus": "LUDB",
        "validation_id": frozen["validation_id"],
        "primary_meta": str(primary_meta),
        "factors": list(SENSITIVITY_FACTORS),
        "knobs": knob_names,
        "primary_metric": "unconditional_Se_at_40_ms",
        "n_records_cap": max_records,
    }
    (out_dir / "ludb_sensitivity_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    lock.write_text(
        json.dumps(
            {
                "note": "Sweep executed after primary lock.",
                "primary_meta": str(primary_meta),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nWrote sweep to {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--frozen",
        type=Path,
        default=Path("validation/config_frozen_v1.json"),
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/ludb"))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("validation/output/ludb_heldout_v1"),
    )
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run one-at-a-time sensitivity sweep (only after primary results exist).",
    )
    p.add_argument(
        "--arm",
        choices=("public_default", "paper_era_morphology"),
        default=None,
        help="Optional sanity check that frozen arm matches expectation.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frozen = load_frozen(args.frozen)
    if args.arm is not None and frozen.get("arm") != args.arm:
        raise SystemExit(
            f"Frozen arm {frozen.get('arm')!r} does not match requested {args.arm!r}"
        )
    if not args.data_dir.exists():
        raise SystemExit(
            f"LUDB data dir missing: {args.data_dir}. "
            "Download with: python -c \"import wfdb; wfdb.dl_database('ludb', 'data/ludb')\""
        )

    if args.sensitivity:
        run_sensitivity(
            args.data_dir,
            frozen,
            args.out_dir,
            max_records=args.max_records,
        )
        return

    beats = run_corpus(args.data_dir, frozen, max_records=args.max_records)
    se, loc, curves = summarize_primary(beats)
    write_primary_outputs(args.out_dir, args.frozen, frozen, beats, se, loc, curves)
    print(f"\nWrote primary held-out results to {args.out_dir}")


if __name__ == "__main__":
    main()
