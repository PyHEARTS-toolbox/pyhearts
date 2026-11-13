"""
Wave-boundary layer: P, QRS and T onsets/offsets for PyHEARTS beats.

Prototype (opt-in). Gaussian fits remain the morphology model; this layer
places clinical boundaries with established single-lead methods, anchored on
PyHEARTS peak locations:

* Multiscale wavelet transform (quadratic-spline prototype, "algorithme a
  trous") computed at 250 Hz, the design rate of Martinez et al. (2004),
  IEEE TBME 51(4):570-581. QRS boundaries at scale 2^2; P and T boundaries at
  scale 2^4. Boundaries are where the wavelet modulus falls below a fixed
  fraction of the outermost significant modulus maximum, or reaches a local
  minimum, whichever comes first. Fractions are the published values
  (QRS on 0.05/0.07, QRS off 0.125/0.71, P on/off 0.5/0.9, T on/off 0.25/0.4).
* QRS is delineated as one complex (first to last significant slope),
  not per Q/R/S component.
* Alternative T end: area indicator of Zhang et al. (2006), IEEE TBME
  53(12):2544-2552, which is baseline-invariant and threshold-free.
* Missing boundaries stay NaN; nothing is imputed from neighboring beats.

Template guidance (per-beat featurization is preserved): each beat is
delineated on its own; a rolling median template of +/-10 neighboring beats is
delineated with the same rules; for P and T, a beat boundary further than a
tolerance (P 40 ms, T 60 ms) from the template boundary is replaced by the
position that best aligns the beat to the template locally (cross-correlation).
Beat-to-beat values therefore still come from each beat's own waveform.

Development: thresholds are the published values. Two implementation choices
(QRS slope maxima must form a chain outward from R with gaps <= 40 ms; P/T
bounded by the slope maxima nearest the wave peak) were selected on PTB-XL
fold 9 against GE 12SL / Philips Uni-G intervals. P-slope selection
(p_nearest_frac) and template mode were selected on QTDB (annotator 1,
channel 0). LUDB and PTB-XL fold 10 were not used for development.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, resample_poly

WT_FS = 250.0  # design rate of the quadratic-spline filter bank


@dataclass(frozen=True)
class BoundaryParams:
    qrs_scale: int = 2          # 2^k
    pt_scale: int = 4           # T-wave scale
    p_scale: int = 4            # P-wave scale
    edge_fallback: str = "limit"  # "limit" | "nan": value when no threshold/minimum before the search limit
    qrs_rule: str = "noise_chain"  # "noise_chain" (boundary-layer-2) | "legacy" (boundary-layer-1)
    qrs_search_ms: float = 150.0
    gamma_qrs: float = 0.10     # significant-maximum fraction for QRS (Martinez gamma), offset side
    gamma_qrs_on: float = 0.04  # significant-maximum fraction before R (onset side)
    qrs_noise_k: float = 3.0    # maxima must also exceed qrs_noise_k x median |W2| around the beat
    qrs_noise_win_s: float = 1.5
    qrs_anchor_ms: float = 50.0  # QRS slope reference: max |W2| within this distance of R
    qrs_first_rule: str = "chain"   # legacy rule only: "earliest" | "chain"
    chain_gap_ms: float = 40.0  # legacy rule only
    chain_gap_on_ms: float = 40.0   # max gap between successive slope maxima, onset side
    chain_gap_off_ms: float = 60.0  # max gap between successive slope maxima, offset side
    pt_max_rule: str = "nearest"  # "largest" | "nearest" (nearest chosen on PTB-XL fold 9)
    pt_nearest_frac: float = 0.3  # "nearest", T wave: smallest maximum kept, as fraction of side max
    p_nearest_frac: float = 0.05  # "nearest", P wave (chosen on QTDB; avoids QRS spill-over at 2^4)
    xi_qrs_on: tuple = (0.05, 0.07)    # (positive, negative) maximum
    xi_qrs_off: tuple = (0.5, 0.125)   # (positive, negative) last maximum; chosen on QTDB + PTB-XL fold 9
    xi_p_on: float = 0.5
    xi_p_off: float = 0.9
    xi_t_on: float = 0.25
    xi_t_off: float = 0.4
    pt_half_window_ms: float = 150.0
    p_half_window_ms: float = 100.0   # P wave: max distance of each P slope maximum from the P peak (QTDB + PTB-XL fold 9)
    p_max_lead_ms: float = 250.0      # P peak must lie within this distance before QRS onset, else no P boundaries
                                      # (a P peak farther back is a T/U wave or AF/flutter activity; QTDB + PTB-XL fold 9)
    zhang_window_ms: float = 128.0
    zhang_search_ms: float = 300.0
    # Template guidance (per-beat values kept; the local template constrains them)
    template_mode: str = "hybrid"  # off | reject | align | hybrid (hybrid chosen on QTDB)
    template_k: int = 10         # neighbors each side in the rolling median template
    template_bounds: tuple = ("P_on", "P_off", "T_on", "T_off")  # QRS stays single-beat (QTDB)
    template_pre_ms: float = 400.0
    template_post_ms: float = 700.0
    tol_ms: tuple = (("P_on", 40), ("P_off", 40), ("QRS_on", 20), ("QRS_off", 20), ("T_on", 60), ("T_off", 60))
    align_half_ms: tuple = (("P", 50), ("QRS", 30), ("T", 80))


BOUNDARY_LAYER_VERSION = "boundary-layer-3"
DEFAULT_PARAMS = BoundaryParams()
# boundary-layer-1 QRS rule, for reproducing earlier results
LEGACY_QRS_PARAMS = dict(qrs_rule="legacy", qrs_search_ms=120.0, gamma_qrs=0.06, xi_qrs_off=(0.125, 0.71),
                         p_half_window_ms=150.0, p_max_lead_ms=1e9)
# boundary-layer-2 P rule (no P search limit), for reproducing earlier results
LEGACY_P_PARAMS = dict(p_half_window_ms=150.0, p_max_lead_ms=1e9)


def atrous_wavelet(x: np.ndarray, n_scales: int = 5) -> np.ndarray:
    """
    Quadratic-spline dyadic WT (Mallat & Zhong; Martinez et al. 2004).

    Row k is scale 2^(k+1). Each scale is computed with its equivalent
    (cascaded) filter, centered so that a symmetric wave's peak falls on the
    zero crossing between its two modulus maxima.
    """
    x = np.asarray(x, float)
    out = []
    lp = np.array([1.0])
    for k in range(n_scales):
        z = np.zeros(2**k - 1)
        g = np.r_[2.0, z, -2.0]
        h = np.r_[1 / 8, z, 3 / 8, z, 3 / 8, z, 1 / 8]
        eq = np.convolve(lp, g)
        c = (eq.size - 1) // 2
        out.append(np.convolve(x, eq, mode="full")[c : c + x.size])
        lp = np.convolve(lp, h)
    return np.vstack(out)


def _maxima(absw: np.ndarray, lo: int, hi: int, frac: float) -> np.ndarray:
    lo, hi = max(lo, 1), min(hi, absw.size - 1)
    if hi - lo < 3:
        return np.array([], int)
    seg = absw[lo:hi]
    pk, _ = find_peaks(seg, height=frac * seg.max() if seg.max() > 0 else np.inf)
    return pk + lo


def _walk(w: np.ndarray, start: int, step: int, thr: float, limit: int) -> Optional[int]:
    """From start, move by step until |w| < thr or |w| hits a local minimum."""
    absw = np.abs(w)
    i = start
    while 0 < i < w.size - 1 and abs(i - start) < limit:
        if absw[i] < thr:
            return i
        if absw[i + step] > absw[i] and absw[i - step] >= absw[i]:  # local minimum of |w|
            return i
        i += step
    return None


def _qrs_bounds(w2: np.ndarray, r: int, p: BoundaryParams) -> tuple[float, float]:
    if p.qrs_rule == "legacy":
        return _qrs_bounds_legacy(w2, r, p)
    return _qrs_bounds_noise_chain(w2, r, p)


def _qrs_bounds_noise_chain(w2: np.ndarray, r: int, p: BoundaryParams) -> tuple[float, float]:
    """
    QRS onset/offset from the 2^2 scale (boundary-layer-2).

    Significant slope maxima must exceed a fraction of the QRS slope (gamma,
    separately before and after R) and a multiple of the local median |W2|, so
    noise maxima cannot extend the complex. Maxima are chained outward from R
    while successive maxima are close; the outermost ones start the threshold
    walk (Martinez et al. 2004).
    """
    absw = np.abs(w2); n = w2.size
    h = int(round(p.qrs_anchor_ms * WT_FS / 1000))
    lo, hi = max(1, r - h), min(n - 1, r + h + 1)
    if hi - lo < 3 or not (0 < r < n - 1):
        return np.nan, np.nan
    M = absw[lo:hi].max()
    nw = int(p.qrs_noise_win_s * WT_FS)
    noise = float(np.median(absw[max(0, r - nw):min(n, r + nw)]))
    floor = p.qrs_noise_k * noise
    thr_off = max(p.gamma_qrs * M, floor)
    thr_on = max(p.gamma_qrs_on * M, floor)
    s = int(round(p.qrs_search_ms * WT_FS / 1000))
    lo, hi = max(1, r - s), min(n - 1, r + s)
    pk, _ = find_peaks(absw[lo:hi], height=min(thr_on, thr_off)); pk = pk + lo

    def chain(seq, gap):  # seq ordered outward from R
        kept = [r]
        for m in seq:
            if abs(m - kept[-1]) > gap:
                break
            kept.append(int(m))
        return kept

    pre = pk[(pk < r) & (absw[pk] >= thr_on)][::-1]
    post = pk[(pk > r) & (absw[pk] >= thr_off)]
    nf = chain(pre, int(round(p.chain_gap_on_ms * WT_FS / 1000)))[-1]
    nl = chain(post, int(round(p.chain_gap_off_ms * WT_FS / 1000)))[-1]
    xi = p.xi_qrs_on[0] if w2[nf] > 0 else p.xi_qrs_on[1]
    i = _walk(w2, nf, -1, xi * absw[nf], s)
    on = float(i) if i is not None else np.nan
    xi = p.xi_qrs_off[0] if w2[nl] > 0 else p.xi_qrs_off[1]
    i = _walk(w2, nl, +1, xi * absw[nl], s)
    off = float(i) if i is not None else np.nan
    return on, off


def _qrs_bounds_legacy(w2: np.ndarray, r: int, p: BoundaryParams) -> tuple[float, float]:
    n = int(round(p.qrs_search_ms * WT_FS / 1000))
    absw = np.abs(w2)
    lo, hi = max(1, r - n), min(absw.size - 1, r + n)
    ref = absw[lo:hi].max() if hi > lo else 0.0
    pk, _ = find_peaks(absw[lo:hi], height=p.gamma_qrs * ref if ref > 0 else np.inf)
    pk = pk + lo
    pre, post = pk[pk <= r], pk[pk >= r]
    if p.qrs_first_rule == "chain":
        gap = int(round(p.chain_gap_ms * WT_FS / 1000))
        def chain(seq):  # seq ordered outward from R
            kept = []
            for m in seq:
                if kept and abs(m - kept[-1]) > gap:
                    break
                if not kept and abs(m - r) > gap:
                    break
                kept.append(m)
            return kept
        pre = np.array(chain(list(pre[::-1]))[::-1], int)
        post = np.array(chain(list(post)), int)
    on = off = np.nan
    if pre.size:
        nf = int(pre[0])
        xi = p.xi_qrs_on[0] if w2[nf] > 0 else p.xi_qrs_on[1]
        i = _walk(w2, nf, -1, xi * absw[nf], n)
        on = float(i) if i is not None else np.nan
    if post.size:
        nl = int(post[-1])
        xi = p.xi_qrs_off[0] if w2[nl] > 0 else p.xi_qrs_off[1]
        i = _walk(w2, nl, +1, xi * absw[nl], n)
        off = float(i) if i is not None else np.nan
    return on, off


def _side_max(absw: np.ndarray, a: int, b: int, toward_peak_last: bool, p: BoundaryParams,
              frac: Optional[float] = None) -> Optional[int]:
    """Slope maximum bounding a wave on one side of its peak, within [a, b)."""
    if b - a < 3:
        return None
    seg = absw[a:b]
    if p.pt_max_rule == "largest":
        return a + int(np.argmax(seg))
    f = p.pt_nearest_frac if frac is None else frac
    pk, props = find_peaks(seg, height=f * seg.max() if seg.max() > 0 else np.inf)
    if not pk.size:
        return a + int(np.argmax(seg))
    return a + int(pk[-1] if toward_peak_last else pk[0])


def _wave_bounds(w4: np.ndarray, peak: int, xi_on: float, xi_off: float,
                 lo_lim: int, hi_lim: int, p: BoundaryParams, frac: Optional[float] = None,
                 half_ms: Optional[float] = None) -> tuple[float, float]:
    n = int(round((p.pt_half_window_ms if half_ms is None else half_ms) * WT_FS / 1000))
    absw = np.abs(w4)
    on = off = np.nan
    lo = max(lo_lim, peak - n)
    hi = min(hi_lim, peak + n)
    nf = _side_max(absw, lo, peak, True, p, frac)
    if nf is not None:
        i = _walk(w4, nf, -1, xi_on * absw[nf], nf - lo + 1)
        on = float(i) if i is not None else (float(lo) if p.edge_fallback == "limit" else np.nan)
    nl = _side_max(absw, peak, hi, False, p, frac)
    if nl is not None:
        i = _walk(w4, nl, +1, xi_off * absw[nl], hi - nl + 1)
        off = float(i) if i is not None else (float(hi) if p.edge_fallback == "limit" else np.nan)
    return on, off


def _zhang_t_end(s: np.ndarray, t_peak: int, hi_lim: int, p: BoundaryParams) -> float:
    """Zhang et al. (2006) area indicator: T end = argmax_k sum_{j=k-w+1..k} (s_j - s_k)."""
    w = int(round(p.zhang_window_ms * WT_FS / 1000))
    hi = min(hi_lim, t_peak + int(round(p.zhang_search_ms * WT_FS / 1000)), s.size - 1)
    if hi - t_peak < w // 2:
        return np.nan
    pol = 1.0 if s[t_peak] >= np.median(s[max(0, t_peak - w):hi]) else -1.0
    ks = np.arange(t_peak + 1, hi + 1)
    c = np.r_[0.0, np.cumsum(s)]
    starts = np.maximum(ks - w + 1, 0)
    area = (c[ks + 1] - c[starts]) - (ks - starts + 1) * s[ks]
    return float(ks[int(np.argmax(pol * area))])


_BOUNDS = ["P_on", "P_off", "QRS_on", "QRS_off", "T_on", "T_off"]


def _template_bounds(t: np.ndarray, r0: int, p_off: float, t_off: float, p: BoundaryParams) -> dict:
    """Delineate one template waveform (R at r0) with the same rules as single beats."""
    pad = 64
    tp = np.r_[np.full(pad, t[0]), t, np.full(pad, t[-1])]
    W = atrous_wavelet(tp, n_scales=5)
    w2, w4 = W[p.qrs_scale - 1], W[p.pt_scale - 1]
    wp = W[p.p_scale - 1]
    r = r0 + pad
    res = dict.fromkeys(_BOUNDS, np.nan)
    res["QRS_on"], res["QRS_off"] = _qrs_bounds(w2, r, p)
    if np.isfinite(p_off):
        pk = int(round(r + p_off)); hi = int(res["QRS_on"]) if np.isfinite(res["QRS_on"]) else r
        if pad < pk < hi and hi - pk <= p.p_max_lead_ms * WT_FS / 1000:
            res["P_on"], res["P_off"] = _wave_bounds(wp, pk, p.xi_p_on, p.xi_p_off, pad, hi, p, p.p_nearest_frac,
                                                     p.p_half_window_ms)
    if np.isfinite(t_off):
        tk = int(round(r + t_off)); lo = int(res["QRS_off"]) if np.isfinite(res["QRS_off"]) else r
        if lo < tk < tp.size - pad:
            res["T_on"], res["T_off"] = _wave_bounds(w4, tk, p.xi_t_on, p.xi_t_off, lo, tp.size - pad, p)
    return {b: v - r for b, v in res.items()}  # offsets from R (samples)


def _align(s: np.ndarray, t: np.ndarray, center_s: int, center_t: int, half: int, maxlag: int) -> Optional[int]:
    """Lag (samples) maximizing correlation of beat patch vs template patch around a boundary."""
    if center_t - half < 0 or center_t + half > t.size:
        return None
    tp = t[center_t - half : center_t + half]
    tp = tp - tp.mean()
    if not np.any(tp):
        return None
    best, best_lag = -np.inf, None
    for lag in range(-maxlag, maxlag + 1):
        a, b = center_s + lag - half, center_s + lag + half
        if a < 0 or b > s.size:
            continue
        bp = s[a:b] - s[a:b].mean()
        den = np.sqrt((bp**2).sum() * (tp**2).sum())
        if den <= 0:
            continue
        c = float((bp * tp).sum() / den)
        if c > best:
            best, best_lag = c, lag
    return best_lag


def _template_guide(s, R, P, T, first: pd.DataFrame, p: BoundaryParams) -> pd.DataFrame:
    """Second pass: rolling-median template per beat; constrain each beat's boundaries."""
    pre = int(round(p.template_pre_ms * WT_FS / 1000)); post = int(round(p.template_post_ms * WT_FS / 1000))
    ok = np.isfinite(R)
    idx = np.where(ok)[0][np.argsort(R[ok])]
    if idx.size < 3:
        return first
    L = pre + post
    M = np.full((idx.size, L), np.nan)
    for j, i in enumerate(idx):
        r = int(round(R[i])); a, b = r - pre, r + post
        seg = np.full(L, np.nan)
        lo, hi = max(a, 0), min(b, s.size)
        seg[lo - a : hi - a] = s[lo:hi]
        M[j] = seg
    tol = {b: int(round(v * WT_FS / 1000)) for b, v in p.tol_ms}
    half = {w: int(round(v * WT_FS / 1000)) for w, v in p.align_half_ms}
    out = first.copy()
    for j, i in enumerate(idx):
        nb = slice(max(0, j - p.template_k), min(idx.size, j + p.template_k + 1))
        t = np.nanmedian(M[nb], axis=0)
        if np.isnan(t).any():
            t = pd.Series(t).interpolate(limit_direction="both").to_numpy()
        nbi = idx[nb]
        p_off = np.nanmedian(P[nbi] - R[nbi]) if np.isfinite(P[nbi]).any() else np.nan
        t_off = np.nanmedian(T[nbi] - R[nbi]) if np.isfinite(T[nbi]).any() else np.nan
        tb = _template_bounds(t, pre, p_off, t_off, p)
        r = R[i]
        for b in p.template_bounds:
            if not np.isfinite(tb[b]):
                if p.template_mode in ("reject",):
                    out.at[out.index[i], b] = np.nan
                continue
            expected = r + tb[b]
            fp = first.iloc[i][b]
            within = np.isfinite(fp) and abs(fp - expected) <= tol[b]
            if p.template_mode == "reject":
                if not within:
                    out.at[out.index[i], b] = np.nan
            elif p.template_mode in ("align", "hybrid"):
                if p.template_mode == "hybrid" and within:
                    continue
                w = b.split("_")[0]
                lag = _align(s, t, int(round(expected)), int(round(pre + tb[b])), half[w], tol[b])
                out.at[out.index[i], b] = expected + lag if lag is not None else np.nan
    return out


def delineate_boundaries(
    ecg: np.ndarray,
    fs: float,
    beats: pd.DataFrame,
    params: BoundaryParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """
    Boundaries for each PyHEARTS beat.

    Parameters
    ----------
    ecg
        The filtered signal PyHEARTS analyzed (same sample grid as ``beats``).
    fs
        Sampling rate (Hz).
    beats
        PyHEARTS feature table; uses ``R_global_center_idx`` and, when present,
        ``P_global_center_idx`` and ``T_global_center_idx``.

    Returns
    -------
    DataFrame aligned to ``beats`` rows with sample indices
    ``P_on, P_off, QRS_on, QRS_off, T_on, T_off, T_off_zhang`` (original fs)
    and intervals in ms ``PR_interval_ms, QRS_interval_ms, QT_interval_ms,
    QT_zhang_ms, P_duration_ms``.
    """
    up, down = int(round(WT_FS)), int(round(fs))
    g = np.gcd(up, down)
    s = resample_poly(np.asarray(ecg, float), up // g, down // g)
    W = atrous_wavelet(s, n_scales=5)
    w2, w4 = W[params.qrs_scale - 1], W[params.pt_scale - 1]
    wp = W[params.p_scale - 1]
    k = WT_FS / fs  # original-sample -> 250 Hz sample

    def col(name):
        return pd.to_numeric(beats[name], errors="coerce").to_numpy(float) if name in beats else np.full(len(beats), np.nan)

    R, P, T = col("R_global_center_idx"), col("P_global_center_idx"), col("T_global_center_idx")
    order = np.argsort(np.where(np.isfinite(R), R, np.inf))
    rows = [None] * len(beats)
    prev_t_off = 0
    for pos, i in enumerate(order):
        res = dict.fromkeys(["P_on", "P_off", "QRS_on", "QRS_off", "T_on", "T_off", "T_off_zhang"], np.nan)
        if np.isfinite(R[i]):
            r = int(round(R[i] * k))
            nxt = order[pos + 1] if pos + 1 < len(order) else None
            next_r = int(round(R[nxt] * k)) if nxt is not None and np.isfinite(R[nxt]) else s.size - 1
            q_on, q_off = _qrs_bounds(w2, r, params)
            res["QRS_on"], res["QRS_off"] = q_on, q_off
            if np.isfinite(P[i]):
                pk = int(round(P[i] * k))
                hi = int(q_on) if np.isfinite(q_on) else r
                if prev_t_off < pk < hi and hi - pk <= params.p_max_lead_ms * WT_FS / 1000:
                    res["P_on"], res["P_off"] = _wave_bounds(wp, pk, params.xi_p_on, params.xi_p_off, prev_t_off, hi,
                                                             params, params.p_nearest_frac, params.p_half_window_ms)
            if np.isfinite(T[i]):
                tk = int(round(T[i] * k))
                lo = int(q_off) if np.isfinite(q_off) else r
                hi = next_r - int(0.1 * WT_FS)
                if lo < tk < hi:
                    res["T_on"], res["T_off"] = _wave_bounds(w4, tk, params.xi_t_on, params.xi_t_off, lo, hi, params)
                    res["T_off_zhang"] = _zhang_t_end(s, tk, hi, params)
            if np.isfinite(res["T_off"]):
                prev_t_off = int(res["T_off"])
        rows[i] = res

    out = pd.DataFrame(rows, index=beats.index,
                       columns=["P_on", "P_off", "QRS_on", "QRS_off", "T_on", "T_off", "T_off_zhang"], dtype=float)
    if params.template_mode != "off" and len(out):
        out = _template_guide(s, R * k, P * k, T * k, out, params)
    for c in out.columns:  # back to original sample grid
        out[c] = out[c] / k
    ms = 1000.0 / fs
    out["PR_interval_ms"] = (out["QRS_on"] - out["P_on"]) * ms
    out["QRS_interval_ms"] = (out["QRS_off"] - out["QRS_on"]) * ms
    out["QT_interval_ms"] = (out["T_off"] - out["QRS_on"]) * ms
    out["QT_zhang_ms"] = (out["T_off_zhang"] - out["QRS_on"]) * ms
    out["P_duration_ms"] = (out["P_off"] - out["P_on"]) * ms
    return out
