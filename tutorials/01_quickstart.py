"""
Tutorial 01: PyHEARTS quickstart (new users)

Goal:
  - Run PyHEARTS end-to-end on a small ECG signal
  - Understand the two main return objects:
      (1) output_df: per-beat features (one row per cardiac cycle)
      (2) epochs_df: segmented cycles in long-form (easy to plot)

Run:
  python3 tutorials/01_quickstart.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Allow running this tutorial from a repo clone without installing the package:
# `python3 tutorials/01_quickstart.py`
#
# If you installed PyHEARTS (pip/conda), this is harmless.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pyhearts import PyHEARTS  # noqa: E402


@dataclass(frozen=True)
class SyntheticECGParams:
    sampling_rate_hz: int = 250
    duration_s: float = 12.0
    heart_rate_bpm: float = 60.0
    rr_jitter_s: float = 0.01
    noise_std_mv: float = 0.01
    baseline_wander_mv: float = 0.03
    baseline_wander_hz: float = 0.33
    t_wave_amp_mv: float = 0.20


def tutorial_params(demo: str) -> SyntheticECGParams:
    """
    demo="clean" is tuned to be a stable, beginner-friendly run.
    demo="noisy" approximates the earlier tutorial settings that could over-detect R peaks.
    """
    if demo == "clean":
        return SyntheticECGParams()
    if demo == "noisy":
        return SyntheticECGParams(
            rr_jitter_s=0.03,
            noise_std_mv=0.02,
            baseline_wander_mv=0.08,
            t_wave_amp_mv=0.35,
        )
    raise ValueError(f"Unknown demo mode: {demo!r}")


def _gaussian(x: np.ndarray, mu: float, sigma: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def make_synthetic_ecg(params: SyntheticECGParams, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """
    Make a simple, physiologically-plausible synthetic ECG in **mV**.

    Notes:
      - This is intentionally simple: a sum of Gaussians (P, Q, R, S, T) repeated over time.
      - For real data, you would load a 1D array and ensure it's in mV.
    """
    rng = np.random.default_rng(seed)
    sr = params.sampling_rate_hz
    n = int(params.duration_s * sr)
    t = np.arange(n) / sr

    # Beat times with small jitter
    rr_mean_s = 60.0 / params.heart_rate_bpm
    beat_times: list[float] = []
    cur = 0.8  # start after a brief lead-in
    while cur < params.duration_s - 0.8:
        beat_times.append(cur)
        cur += max(0.35, rr_mean_s + rng.normal(0.0, params.rr_jitter_s))

    # Wave offsets (seconds) relative to R
    p_off, q_off, r_off, s_off, t_off = (-0.16, -0.04, 0.0, 0.04, 0.25)

    # Widths (seconds)
    p_w, q_w, r_w, s_w, t_w = (0.025, 0.012, 0.010, 0.012, 0.050)

    # Amplitudes (mV)
    p_a, q_a, r_a, s_a, t_a = (0.12, -0.12, 1.00, -0.25, params.t_wave_amp_mv)

    ecg = np.zeros_like(t, dtype=float)
    for r_time in beat_times:
        ecg += _gaussian(t, r_time + p_off, p_w, p_a)
        ecg += _gaussian(t, r_time + q_off, q_w, q_a)
        ecg += _gaussian(t, r_time + r_off, r_w, r_a)
        ecg += _gaussian(t, r_time + s_off, s_w, s_a)
        ecg += _gaussian(t, r_time + t_off, t_w, t_a)

    # Baseline wander + measurement noise
    ecg += params.baseline_wander_mv * np.sin(2 * np.pi * params.baseline_wander_hz * t)
    ecg += rng.normal(0.0, params.noise_std_mv, size=ecg.size)

    # For sanity-checking R-peaks later, return the expected R times as well.
    return ecg, t, np.asarray(beat_times, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="PyHEARTS Tutorial 01: quickstart")
    parser.add_argument(
        "--demo",
        choices=["clean", "noisy"],
        default="clean",
        help="Choose the synthetic ECG difficulty (default: clean).",
    )
    parser.add_argument(
        "--sensitivity",
        choices=["standard", "high", "maximum"],
        default=None,
        help="Override PyHEARTS sensitivity (default depends on demo).",
    )
    args = parser.parse_args()

    params = tutorial_params(args.demo)
    raw_ecg_mv, t, expected_r_times_s = make_synthetic_ecg(params)

    # 1) Create the analyzer (new users: start with a preset + sensitivity)
    default_sensitivity = "standard" if args.demo == "clean" else "high"
    sensitivity = args.sensitivity or default_sensitivity
    hearts = PyHEARTS(
        sampling_rate=float(params.sampling_rate_hz),
        species="human",
        sensitivity=sensitivity,
        verbose=False,
        plot=False,
    )

    # 2) (Optional but recommended) preprocess the raw signal
    ecg_filtered = hearts.preprocess_signal(
        raw_ecg_mv,
        highpass_cutoff=0.5,
        lowpass_cutoff=40.0,
        filter_order=2,
        notch_frequency=None,
        quality_factor=None,
        poly_degree=None,
    )
    if ecg_filtered is None:
        raise RuntimeError("Preprocessing failed (got None).")

    # 3) Run the full pipeline
    #    - analyze_ecg expects a *preprocessed* signal (in mV)
    #    - pass raw_ecg for more robust polarity detection (recommended when available)
    output_df, epochs_df = hearts.analyze_ecg(ecg_filtered, raw_ecg=raw_ecg_mv)

    print("\n=== Returned objects ===")
    print(f"output_df: {type(output_df)} shape={output_df.shape}")
    print(f"epochs_df: {type(epochs_df)} shape={epochs_df.shape}")

    # Quick R-peak sanity check: compare detected peaks to the synthetic ground truth.
    r_peaks = getattr(hearts, "r_peak_indices", None)
    if isinstance(r_peaks, np.ndarray) and r_peaks.size > 0:
        detected_r_times_s = t[r_peaks]
        # For each expected R time, compute nearest detected R time delta (seconds)
        nearest_abs_err_s = []
        for rt in expected_r_times_s:
            nearest_abs_err_s.append(float(np.min(np.abs(detected_r_times_s - rt))))
        nearest_abs_err_s_arr = np.asarray(nearest_abs_err_s)
        print("\n=== R-peak sanity check (synthetic ground truth) ===")
        print(f"expected beats: {expected_r_times_s.size}")
        print(f"detected R-peaks: {r_peaks.size}")
        print(f"median |timing error|: {np.median(nearest_abs_err_s_arr) * 1000.0:.1f} ms")
        print(f"max |timing error|: {np.max(nearest_abs_err_s_arr) * 1000.0:.1f} ms")
    else:
        print("\n=== R-peak sanity check ===")
        print("No detected R-peaks found on this run.")

    # 4) Inspect per-cycle features
    print("\n=== output_df preview (first 5 rows) ===")
    with pd.option_context("display.max_columns", 12, "display.width", 120):
        print(output_df.head())

    headline_cols = [
        "RR_interval_ms",
        "PR_interval_ms",
        "QRS_interval_ms",
        "QT_interval_ms",
        "QTc_Bazett_ms",
        "R_gauss_height",
        "r_squared",
        "rmse",
    ]
    existing_headline_cols = [c for c in headline_cols if c in output_df.columns]
    print("\n=== A few headline columns (if present) ===")
    print(existing_headline_cols)

    if existing_headline_cols:
        with pd.option_context("display.max_columns", 20, "display.width", 120):
            print(output_df[existing_headline_cols].describe(include="all"))

    # 5) NaNs are normal: they typically mean a wave wasn't confidently detected/fitted on that beat.
    if not output_df.empty:
        nan_rate = output_df.isna().mean().sort_values(ascending=False)
        print("\n=== Columns with the most missing values (top 10) ===")
        print(nan_rate.head(10))

        if "r_squared" in output_df.columns:
            good_fit_frac = float((output_df["r_squared"] > 0.9).mean(skipna=True))
            print(f"\nFit quality: fraction of beats with r_squared > 0.9: {good_fit_frac:.2%}")

    # 6) Plotting: raw vs filtered, detected R-peaks, and one example epoch
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)

        # (a) Raw vs filtered
        axes[0].plot(t, raw_ecg_mv, lw=1.0, alpha=0.8, label="raw (mV)")
        axes[0].plot(t, ecg_filtered, lw=1.0, alpha=0.8, label="filtered (mV)")
        axes[0].set_title("Synthetic ECG (raw vs filtered)")
        axes[0].set_xlabel("time (s)")
        axes[0].set_ylabel("amplitude (mV)")
        axes[0].legend(loc="upper right")

        # (b) Filtered with detected R-peaks
        axes[1].plot(t, ecg_filtered, lw=1.0, label="filtered (mV)")
        if isinstance(r_peaks, np.ndarray) and r_peaks.size > 0:
            axes[1].scatter(t[r_peaks], ecg_filtered[r_peaks], s=18, c="red", label="Detected R-peaks")
        # Overlay expected (synthetic) R times for comparison
        for i, rt in enumerate(expected_r_times_s):
            axes[1].axvline(rt, color="k", lw=0.6, alpha=0.15, label="Expected R time" if i == 0 else None)
        axes[1].set_title("Detected R-peaks")
        axes[1].set_xlabel("time (s)")
        axes[1].set_ylabel("amplitude (mV)")
        axes[1].legend(loc="upper right")

        # (c) One segmented cycle (epoch)
        if not epochs_df.empty and "cycle" in epochs_df.columns:
            cycle_id = int(epochs_df["cycle"].unique()[0])
            one = epochs_df.loc[epochs_df["cycle"] == cycle_id].sort_values("signal_x")
            axes[2].plot(one["signal_x"].to_numpy() * 1000.0, one["signal_y"].to_numpy(), lw=1.0)
            axes[2].axvline(0.0, color="k", lw=1.0, alpha=0.5)
            axes[2].set_title(f"Example epoch (cycle={cycle_id}) — x-axis is time relative to R")
            axes[2].set_xlabel("time (ms, relative to R)")
            axes[2].set_ylabel("detrended amplitude (mV)")
        else:
            axes[2].text(0.01, 0.5, "epochs_df is empty (no cycles retained).", transform=axes[2].transAxes)
            axes[2].set_axis_off()

        fig.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\nPlotting skipped (matplotlib error): {e}")


if __name__ == "__main__":
    main()


