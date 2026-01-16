"""
Smoke test runner for PyHEARTS on local mouse baseline .mat files.

Goal: verify end-to-end processing still works after toolbox changes:
- Load .mat (data/labels/isi)
- Extract ECG channel (prefer label containing "ECG")
- Compute sampling rate from isi (ms or s)
- Optionally crop to a time window
- Run PyHEARTS.analyze_ecg(species="mouse")
- Save per-file outputs + a summary CSV

Usage (example):
  python3 analysis_scripts/smoke_test_mouse_baseline.py \
    --data-dir /Users/morganfitzgerald/Documents/pyhearts/data/mouse_baseline \
    --max-files 3 --seconds 30 --sensitivity high --notch 50
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import scipy.io

# Ensure local package import works when running this script from any CWD
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pyhearts.core.fit import PyHEARTS  # noqa: E402


def _coerce_str_list(arr: object) -> list[str]:
    if arr is None:
        return []
    flat = np.array(arr).ravel()
    out: list[str] = []
    for x in flat:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", "ignore"))
        else:
            out.append(str(x))
    return out


def _is_resource_fork(path: Path) -> bool:
    # macOS "._" resource fork artifacts
    return path.name.startswith("._")


def _iter_mat_files(data_dir: Path) -> Iterable[Path]:
    for p in sorted(data_dir.rglob("*.mat")):
        if _is_resource_fork(p):
            continue
        yield p


@dataclass(frozen=True)
class MatECG:
    file_path: Path
    ecg: np.ndarray
    fs_hz: float
    label: str
    units: str


def load_mouse_mat_ecg(file_path: Path, *, prefer_label_contains: str = "ECG") -> MatECG:
    mat = scipy.io.loadmat(str(file_path))

    if "data" not in mat:
        raise KeyError("Missing 'data' key in .mat")

    data = np.asarray(mat["data"])
    if data.ndim != 2 or data.shape[1] < 1:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    labels = _coerce_str_list(mat.get("labels"))
    units = _coerce_str_list(mat.get("units"))

    ecg_col = 0
    if labels:
        # Prefer an ECG-ish label (e.g. "ECG100C")
        upper = [s.upper() for s in labels]
        key = prefer_label_contains.upper()
        hits = [i for i, s in enumerate(upper) if key in s]
        if hits:
            ecg_col = hits[0]

    ecg_label = labels[ecg_col] if ecg_col < len(labels) else f"col{ecg_col}"
    ecg_units = units[ecg_col] if ecg_col < len(units) else ""

    ecg = np.asarray(data[:, ecg_col], dtype=float)

    isi = mat.get("isi", None)
    if isi is None:
        raise KeyError("Missing 'isi' key in .mat (cannot compute sampling rate)")
    isi_val = float(np.asarray(isi).ravel()[0])

    isi_units = _coerce_str_list(mat.get("isi_units"))
    unit = isi_units[0].lower() if isi_units else "ms"

    if unit in {"ms", "msec", "millisecond", "milliseconds"}:
        fs_hz = 1000.0 / isi_val
    elif unit in {"s", "sec", "second", "seconds"}:
        fs_hz = 1.0 / isi_val
    else:
        # fallback: most of these exports are ms
        fs_hz = 1000.0 / isi_val

    if not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError(f"Invalid sampling rate computed from isi={isi_val} ({unit}): fs={fs_hz}")

    return MatECG(file_path=file_path, ecg=ecg, fs_hz=fs_hz, label=ecg_label, units=ecg_units)


def _crop_signal(ecg: np.ndarray, fs: float, start_sec: float, seconds: Optional[float]) -> np.ndarray:
    start = int(max(0.0, start_sec) * fs)
    if seconds is None:
        return ecg[start:]
    n = int(max(0.0, seconds) * fs)
    return ecg[start : start + n]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        type=str,
        default="/Users/morganfitzgerald/Documents/pyhearts/data/mouse_baseline",
        help="Directory containing mouse baseline .mat files.",
    )
    ap.add_argument("--max-files", type=int, default=3, help="Max .mat files to process.")
    ap.add_argument("--start-sec", type=float, default=0.0, help="Crop start time (seconds).")
    ap.add_argument("--seconds", type=float, default=30.0, help="Crop duration (seconds). Use 0 for full.")
    ap.add_argument("--sensitivity", type=str, default="high", choices=["standard", "high", "maximum"])
    ap.add_argument("--highpass", type=float, default=0.0, help="High-pass cutoff (Hz). Use 0 to disable.")
    ap.add_argument("--lowpass", type=float, default=0.0, help="Low-pass cutoff (Hz). Use 0 to disable.")
    ap.add_argument("--notch", type=float, default=0.0, help="Notch frequency (Hz). Use 0 to disable.")
    ap.add_argument("--q", type=float, default=30.0, help="Notch quality factor (Q).")
    ap.add_argument("--filter-order", type=int, default=4, help="Butterworth filter order.")
    ap.add_argument(
        "--poly-degree",
        type=int,
        default=0,
        help="Polynomial detrend degree. Use 0 to disable.",
    )
    ap.add_argument("--out-dir", type=str, default="", help="Output directory (default: results/mouse_smoke_<ts>).")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data-dir not found: {data_dir}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / f"mouse_smoke_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    seconds: Optional[float]
    seconds = None if args.seconds == 0 else float(args.seconds)

    hp = None if args.highpass == 0 else float(args.highpass)
    lp = None if args.lowpass == 0 else float(args.lowpass)
    notch = None if args.notch == 0 else float(args.notch)
    poly = None if args.poly_degree == 0 else int(args.poly_degree)

    rows: list[dict] = []

    files = list(_iter_mat_files(data_dir))[: max(0, args.max_files)]
    if not files:
        print(f"No .mat files found under {data_dir}")
        return 2

    print(f"Found {len(files)} file(s). Writing outputs to {out_dir}")

    for fp in files:
        rec = load_mouse_mat_ecg(fp)

        raw = _crop_signal(rec.ecg, rec.fs_hz, args.start_sec, seconds)
        analyzer = PyHEARTS(
            sampling_rate=rec.fs_hz,
            species="mouse",
            sensitivity=args.sensitivity,
            verbose=False,
            plot=False,
        )

        # High-pass filters at extremely low normalized cutoffs can be numerically unstable
        # at very high sampling rates (e.g., fs=10kHz, hp=0.5Hz). Prefer polynomial detrending.
        if hp is not None and (hp / (0.5 * rec.fs_hz)) < 1e-3:
            print(
                f"[WARN] {fp.stem}: highpass={hp}Hz is very low relative to fs={rec.fs_hz:.1f}Hz; "
                "skipping high-pass to avoid filtfilt instability. Use --highpass 0 to silence."
            )
            hp_effective = None
        else:
            hp_effective = hp

        filt = analyzer.preprocess_signal(
            raw,
            highpass_cutoff=hp_effective,
            filter_order=args.filter_order if (hp_effective is not None or lp is not None) else None,
            lowpass_cutoff=lp,
            notch_frequency=notch,
            quality_factor=args.q if notch is not None else None,
            poly_degree=poly,
        )
        if filt is None:
            raise RuntimeError(f"preprocess_signal returned None for {fp}")

        out_df, epochs_df = analyzer.analyze_ecg(filt, raw_ecg=raw)

        file_id = fp.stem
        file_dir = out_dir / file_id
        file_dir.mkdir(parents=True, exist_ok=True)

        out_csv = file_dir / f"{file_id}_features.csv"
        epochs_csv = file_dir / f"{file_id}_epochs.csv"

        out_df.to_csv(out_csv, index=False)
        epochs_df.to_csv(epochs_csv, index=False)

        n_r = getattr(analyzer, "r_peak_indices", np.array([]))
        n_r = int(len(n_r)) if n_r is not None else 0
        n_cycles = int(out_df.shape[0])

        row = {
            "file": str(fp),
            "file_id": file_id,
            "fs_hz": float(rec.fs_hz),
            "label": rec.label,
            "units": rec.units,
            "samples_used": int(raw.size),
            "seconds_used": float(raw.size / rec.fs_hz),
            "r_peaks": n_r,
            "cycles_features": n_cycles,
            "features_csv": str(out_csv),
            "epochs_csv": str(epochs_csv),
        }
        rows.append(row)

        print(
            f"[OK] {file_id}: fs={rec.fs_hz:.1f} Hz, samples={raw.size}, "
            f"r_peaks={n_r}, cycles={n_cycles}"
        )

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "SUMMARY.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


