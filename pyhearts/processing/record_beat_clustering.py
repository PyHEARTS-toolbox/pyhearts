"""
Cluster S→Q beat segments for per-cluster template priors (Phase 2D).

Clusters are used only to build window priors — never to force peak replacement.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from pyhearts.config import ProcessCycleConfig
from pyhearts.processing.delineation_signal import prepare_record_delineation_signal
from pyhearts.processing.record_delineation import (
    MedianBeatTemplate,
    _find_q_before_r,
    _find_s_after_r,
    _resample_segment,
    build_stpq_beat_template,
    delineate_record_template,
    finalize_stpq_median_template,
)


def extract_stpq_segments(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
) -> List[Tuple[int, np.ndarray, int, int]]:
    """
    Extract resampled S(i)→Q(i+1) segments.

    Returns list of (epoch_index, segment, s_i, q_next) in beat order.
    """
    ecg_work = prepare_record_delineation_signal(ecg, sampling_rate, cfg)
    r_peaks = np.asarray(r_peaks, dtype=int)
    segments: List[Tuple[int, np.ndarray, int, int]] = []
    lens: List[int] = []

    for i in range(len(r_peaks) - 1):
        r_i = int(r_peaks[i])
        s_i = _find_s_after_r(ecg_work, r_i, sampling_rate, cfg)
        q_next = _find_q_before_r(ecg_work, int(r_peaks[i + 1]), sampling_rate, cfg)
        if s_i is None or q_next is None or q_next <= int(s_i) + 3:
            continue
        seg = ecg_work[int(s_i) : int(q_next)]
        if seg.size < 8:
            continue
        segments.append((i, seg.astype(float, copy=False), int(s_i), int(q_next)))
        lens.append(int(seg.size))

    if not segments:
        return []

    target_len = int(np.median(lens))
    target_len = max(8, target_len)
    out: List[Tuple[int, np.ndarray, int, int]] = []
    for epoch_i, seg, s_i, q_next in segments:
        out.append((epoch_i, _resample_segment(seg, target_len), s_i, q_next))
    return out


def _kmeans_labels(x: np.ndarray, k: int, *, seed: int = 0, max_iter: int = 30) -> np.ndarray:
    """Simple k-means on row vectors."""
    n = x.shape[0]
    if n <= k:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(n, size=k, replace=False)
    centers = x[init_idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dists = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = x[mask].mean(axis=0)
    return labels


def cluster_epoch_indices(
    segments: List[Tuple[int, np.ndarray, int, int]],
    k: int,
    *,
    seed: int = 0,
) -> Dict[int, int]:
    """
    Map beat ``epoch_index`` → cluster_id (0..k-1).

    Uses k-means on z-scored resampled S→Q segments.
    """
    if not segments or k <= 1:
        return {epoch_i: 0 for epoch_i, _, _, _ in segments}

    mat = np.vstack([s for _, s, _, _ in segments])
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True)
    sd[sd < 1e-9] = 1.0
    z = (mat - mu) / sd
    labels = _kmeans_labels(z, min(k, mat.shape[0]), seed=seed)
    return {segments[i][0]: int(labels[i]) for i in range(len(segments))}


def stpq_clusters_are_heterogeneous(
    segments: List[Tuple[int, np.ndarray, int, int]],
    k: int,
    *,
    min_centroid_separation: float = 1.25,
    seed: int = 0,
) -> bool:
    """
    True when k-means finds meaningfully separated S→Q morphology clusters.

    Homogeneous records should keep the single record-level template for windows.
    """
    if not segments or k <= 1 or len(segments) < 2 * k:
        return False

    mat = np.vstack([s for _, s, _, _ in segments])
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True)
    sd[sd < 1e-9] = 1.0
    z = (mat - mu) / sd
    labels = _kmeans_labels(z, min(k, mat.shape[0]), seed=seed)

    centroids: List[np.ndarray] = []
    within_scales: List[float] = []
    for j in range(min(k, mat.shape[0])):
        mask = labels == j
        if not np.any(mask):
            return False
        cluster_z = z[mask]
        centroid = cluster_z.mean(axis=0)
        centroids.append(centroid)
        within_scales.append(float(np.linalg.norm(cluster_z - centroid)))

    if len(centroids) < 2:
        return False

    between = float(np.linalg.norm(centroids[0] - centroids[1]))
    within = float(np.mean(within_scales)) if within_scales else 0.0
    return between >= min_centroid_separation * max(within, 1e-6)


def build_cluster_templates(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    sampling_rate: float,
    cfg: ProcessCycleConfig,
    cluster_by_epoch: Dict[int, int],
    *,
    manual_ann_ext: Optional[str] = None,
) -> Dict[int, MedianBeatTemplate]:
    """
    Build one delineated STPQ template per cluster from member beats' S→Q segments.
    """
    segments = extract_stpq_segments(ecg, r_peaks, sampling_rate, cfg)
    if not segments:
        return {}

    ecg_work = prepare_record_delineation_signal(ecg, sampling_rate, cfg)
    raw = build_stpq_beat_template(ecg, r_peaks, sampling_rate, cfg)
    r_peaks_arr = np.asarray(r_peaks, dtype=int)
    r_amps = [
        abs(float(ecg_work[int(r)]))
        for r in r_peaks_arr
        if 0 <= int(r) < ecg_work.size
    ]
    mean_r = float(np.median(r_amps)) if r_amps else 1.0

    by_cluster: Dict[int, List[np.ndarray]] = {}
    cluster_anchors: Dict[int, List[Tuple[int, int, int]]] = {}
    for epoch_i, seg, s_i, q_next in segments:
        cid = cluster_by_epoch.get(epoch_i, 0)
        by_cluster.setdefault(cid, []).append(seg)
        r_i = int(r_peaks_arr[epoch_i]) if epoch_i < len(r_peaks_arr) else 0
        cluster_anchors.setdefault(cid, []).append((int(s_i), int(q_next), r_i))

    templates: Dict[int, MedianBeatTemplate] = {}
    for cid, segs in by_cluster.items():
        if len(segs) < max(3, cfg.record_delineation_min_beats // 2):
            continue
        stack = np.vstack(segs)
        template = np.median(stack, axis=0) if cfg.record_template_aggregate != "mean" else np.mean(stack, axis=0)
        stub = finalize_stpq_median_template(
            template,
            cfg,
            sampling_rate,
            pre_r_samples=raw.pre_r_samples,
            median_rr_samples=raw.median_rr_samples,
            n_beats=len(segs),
            mean_r_amplitude=mean_r,
            beat_anchors=cluster_anchors.get(cid),
            ecg_work=ecg_work,
        )
        templates[cid] = delineate_record_template(
            stub, sampling_rate, cfg, manual_ann_ext=manual_ann_ext
        )
    return templates
